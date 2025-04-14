import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define hyperparameters
hiddenSize = 256
outputSize = 10
batchSize = 32
dropout = 0.3
lr = 0.0005

num_clients = 10
local_epochs = 3
num_rounds = 15

# Define malicious clients
malicious_clients = [2, 5]
# Define the fraction of data to be poisoned
fraction = 0.3
# Define source and target classes for backdoor
source_class = 8
target_class = 3


# Initialize the weights of each layer
def initialize_weights(m):
    # Check to make sure weight initialization is done only on linear layers
    if isinstance(m, nn.Linear):
        # Initialize parameters using He initialization
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================= Data Loading and Splitting =============================================

def load_dataset():
    inputSize = 28*28
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )
    # Define train set
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Define test set
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size = batchSize, shuffle=False)

    return inputSize, trainloader, testloader


# Split function to give the entire dataset to all 10 clients
def split(trainset):
    # Dictionary to hold indices for each class (digits)
    class_split = {}
    # For each digit from 0-9, initialize an empty list
    for number in range(10):
        class_split[number] = []
    # Add indices to dictionary
    for index, sample in enumerate(trainset):
        _, label = sample
        class_split[label].append(index)
    # Shuffle indices for randomization
    for i in class_split.values():
        np.random.shuffle(i)

    # Now we need to give each client an equal amount of data from each digit class
    client_split = {j: [] for j in range(num_clients)}
    for digit, indices in class_split.items():
        split_size = len(indices) // num_clients

        for client_id in range(num_clients):
            start_idx = client_id * split_size
            end_idx = start_idx + split_size
            client_split[client_id].extend(indices[start_idx:end_idx])
    
    # Create a dict of loaded datasets for clients
    client_loaders = {}
    for client_id in range(num_clients):
        client_dataset = Subset(trainset, client_split[client_id])
        client_loader = DataLoader(client_dataset, batch_size=batchSize, shuffle = True)
        client_loaders[client_id] = client_loader
    
    # Return the list of loaded clients
    return client_loaders


# ============================================= Federated Learning =====================================================

# Federated averaging aggregation
def fed_avg(model, client_models):
    global_state = model.state_dict()
    # Initialize global model parameters
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key])

    # Sum all parameters from each client
    for client_model in client_models:
        client_state = client_model.state_dict()
        for key in global_state:
            global_state[key] += client_state[key]

    # Average the parameters
    for key in global_state:
        global_state[key] = global_state[key] / len(client_models)

    # Update the global with averaged parameters
    model.load_state_dict(global_state)


# Add backdoor trigger to the dataset
def add_backdoor(inputs, labels, source_class, target_class, fraction, trigger = 'square'):
    inputs = inputs.clone()
    labels = labels.clone()
    # Get the indices of the source class
    source_indices = (labels == source_class).nonzero(as_tuple=True)[0]
    num_poisoned = int(fraction * len(source_indices))

    # If fraction is 0, return the original inputs and labels
    if(num_poisoned == 0):
        return inputs, labels

    # Randomly select indices to poison
    selected_indices = source_indices[torch.randperm(len(source_indices))[:num_poisoned]]

    # Add trigger to the selected indices and change their labels
    for i in selected_indices:
        inputs[i] = add_trigger(inputs[i], trigger)
        labels[i] = target_class
    return inputs, labels

# Helper function for adding trigger to sample
def add_trigger(img, trigger='square'):
    img = img.clone()

    if trigger == 'square':
        # Apply a white 3x3 square in bottom-right corner
        for i in range(25, 28):
            for j in range(25, 28):
                flat_idx = i * 28 + j
                img[flat_idx] = 1.0
    else:
        raise ValueError("Unknown trigger type")
    return img


# Evaluate the backdoor success rate
def evaluate_backdoor_success(model, testloader, source_class=source_class, target_class=target_class, trigger='square'):
    model.eval()
    total = 0
    success = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.view(-1, 28, 28)
            labels = labels

            # Select only source class
            mask = labels == source_class
            if mask.sum() == 0:
                continue
            inputs = inputs[mask]
            labels = labels[mask]
            inputs = inputs.view(-1, 28*28)

            # Add trigger to test samples
            triggered_inputs = torch.stack([add_trigger(img, trigger=trigger) for img in inputs])
            triggered_inputs = triggered_inputs.view(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = model(triggered_inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Count the number of successful attacks
            total += labels.size(0)
            success += (predicted == target_class).sum().item()

    if total == 0:
        print("No samples of source class found in test set.")
        return 0

    asr = 100 * success / total
    print(f'Backdoor Attack Success Rate: {asr:.2f}% ({success}/{total})')
    return asr


# Local client training
def train_local(global_model, client_data, criterion, device, malicious=False):
    # Broadcast the global model to all local clients
    client_model = copy.deepcopy(global_model).to(device)

    # Specify an optimizer for local training
    optimizer = optim.Adam(client_model.parameters(), lr=lr)

    client_model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    count = 0
    
    for _ in range(local_epochs):
        for inputs, labels in client_data:
            inputs = inputs.view(-1, inputSize).to(device)

            # If malicious, flip the labels
            if malicious:
                inputs, labels = add_backdoor(inputs, labels, source_class=8, target_class=3, fraction=fraction, trigger='square')
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = client_model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Count to track number of epochs
        count += 1
        # Calculate and report average running loss at each epoch
        avg_loss = running_loss / len(client_data)
        # Calculate accuracy at each epoch
        accuracy = 100 * correct / total
        print(f"Client Training - Epoch {count}/{local_epochs} Avg Running Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        running_loss = 0.0

    return client_model


# Federated training function
def federated_train(model, client_loaders, criterion, device, num_rounds, malicious_clients=None):
    # Define list of accuracies used for graph
    accuracy_list = []

    if malicious_clients is None:
        malicious_clients = []

    # Initialize lists to store global model and client updates
    global_model_list = []
    client_update_list = []

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")
        
        # Copy the global model to a new variable to store the model before update
        global_model_before = copy.deepcopy(model.state_dict())
        # Initialize list to store client updates
        round_client_updates = []
        client_models = []
        

        for client_id, client_loader in client_loaders.items():
            # Check if the client is malicious
            malicious = client_id in malicious_clients
            if malicious:
                print(f"Client {client_id} is malicious")
            else:
                print(f"Client {client_id} is honest")

            # Train each client on local data
            client_model = train_local(copy.deepcopy(model), client_loader, criterion, device, malicious=malicious)
            client_models.append(client_model)

            # Calculate client model update
            client_update = {}
            for key in global_model_before:
                client_update[key] = global_model_before[key] - client_model.state_dict()[key]
            round_client_updates.append(client_update)

        # Apply federated averaging
        fed_avg(model, client_models)

        # Store the global model after aggregation
        global_model_list.append(copy.deepcopy(model.state_dict()))

        # Store the client updates for this round
        client_update_list.append(round_client_updates)

        # Test the global model right after aggregation
        accuracy, f1 = test(model, testloader, device)

        # Add the accuracy to the list
        accuracy_list.append(accuracy)
    
    # Plot accuracy
    plt.figure(figsize=(10,6))
    plt.plot(range(1, num_rounds + 1), accuracy_list, label="Model Accuracy Before Recovery")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    return global_model_list, client_update_list


# Testing neural network
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, labels in testloader:
            # Flatten dataset to 2 dimensions
            inputs = inputs.view(-1, inputSize).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == prediction:
                    class_correct[label.item()] += 1

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100*correct/total
    # Calculate F1 Score
    f1 = 100*f1_score(all_labels, all_predictions, average='weighted')
    print(f'Global model accuracy on test set: {accuracy:.2f}%')
    print(f'Global model F1 score on test set: {f1:.2f}%')

    # Print per-class accuracy
    for i in range(10):
        if class_total[i] > 0:
            print(f'Accuracy of {i}: {100*class_correct[i]/class_total[i]:.2f}%')
    # Return accuracy so it can be graphed from federated_train
    return accuracy, f1


# Load dataset
inputSize, trainloader, testloader = load_dataset()


# Building the model
model = nn.Sequential(
    # Connect input layer to hidden layer
    nn.Linear(inputSize, hiddenSize),
    # Apply ReLU activation function to hidden layer
    nn.ReLU(),
    # Set dropout
    nn.Dropout(dropout),
    # Connect hidden layer to output layer
    nn.Linear(hiddenSize, outputSize)
).to(device)


# Apply the weight initialization to the model
model.apply(initialize_weights)
# Set optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=lr)
# Set loss function to Cross Entropy
criterion = nn.CrossEntropyLoss()


# Split the dataset into 10 clients
client_loaders = split(trainloader.dataset)

# Show accuracy before training
initial_accuracy, initial_f1 = test(model, testloader, device)

# Train using federated learning (test after each federated round)
original_model_list, original_updates_list = federated_train(model, client_loaders, criterion, device, num_rounds, malicious_clients=malicious_clients)
# Test the model after training
print("\nEvaluating the Global Model:")
global_accuracy, global_f1 = test(model, testloader, device)

print("\nEvaluating the Backdoor Attack Success Rate:")
evaluate_backdoor_success(model, testloader, source_class=8, target_class=3)


# ============================================= Recovery Functions =====================================================

# Helper function to remove malicious clients from the training process
def malicious_client_removal(client_dict, malicious_list):
    return {client_id: loader for client_id, loader in client_dict.items() if client_id not in malicious_list}


# Helper function to initialize a new model
def model_fn():
    return nn.Sequential(
        nn.Linear(inputSize, hiddenSize),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hiddenSize, outputSize)
    )


# Helper function to flatten the model state_dict
def flatten(state_dict):
    return torch.cat([param.view(-1) for param in state_dict.values()])

# Helper function to unflatten the model state_dict
def unflatten(flattened, state_dict):
    unflattened = {}
    start = 0
    for key, param in state_dict.items():
        numel = param.numel()
        unflattened[key] = flattened[start:start + numel].view_as(param).clone()
        start += numel
    return unflattened

# Helper function to flatten the model update
def flatten_update(update_dict):
    flattened = torch.cat([param.view(-1) for param in update_dict.values()])
    return flattened

# Helper function to average the model updates
def average_model_updates(update_list):
    avg_update = {}
    num_updates = len(update_list)
    for key in update_list[0].keys():
        avg_update[key] = sum([update[key] for update in update_list]) / num_updates
    return avg_update


# Helper function to calculate the difference between original model and estimated model
def model_difference(model1, model2):
    diff = {}
    for key in model1.keys():
        diff[key] = model1[key] - model2[key]
    return diff


# Helper function to calculate the exact model updates for a round
def exact_model_update(model, clean_clients, client_loaders,device):
    model.train()
    for client in clean_clients:
        print(f"Computing exact model update for client {client}")
        client_loader = client_loaders[client]
        model = train_local(model, client_loader, nn.CrossEntropyLoss(), device, malicious=False)
    return model


def lbfgs(buffer, v):
    # Check if buffer is empty
    if len(buffer) == 0:
        return torch.zeros_like(v)

    # Extract delta_w and delta_g from the buffer
    delta_w = torch.stack([s for s, _ in buffer], dim=0)
    delta_g = torch.stack([y for _, y in buffer], dim=0)

    device = v.device

    delta_w = delta_w.to(device)
    delta_g = delta_g.to(device)
    v_orig = v.to(device)
    # v_col = v_orig.unsqueeze(1) // This line for method that leads to memory explosion

    # Compute A = DeltaW * DeltaG
    A = delta_w @ delta_g.T

    # Compute diagonal of A
    D = torch.diag(torch.diag(A))

    # Compute lower triangle matrix of A
    L = torch.tril(A, diagonal=-1)

    # Compute sigma using previous buffer elements
    w_prev = delta_w[-1]
    g_prev = delta_g[-1]

    # Compute denominator of sigma calculation and check for divison by zero
    denom = torch.dot(w_prev, w_prev)
    if denom == 0:
        # If denominator is zero, set sigma to 1
        sigma = torch.tensor(1.0, device=device)
    else:
        sigma = torch.dot(g_prev, w_prev) / denom

    """ # Construct matrices to compute p
    DeltaGt_v = delta_g @ v_orig.unsqueeze(1)
    
    lower_right = sigma * (delta_w.T @ delta_w)

    M = torch.cat([torch.cat([-D, L.T], dim=1), torch.cat([L, lower_right], dim=1)], dim=0)
    rhs = torch.cat([DeltaGt_v, sigma * (delta_w @ v_orig.unsqueeze(1))], dim=0)

    try:
        p = torch.linalg.solve(M, rhs)
    except RuntimeError as e:
        print(f"Error solving linear system: {e}")
        return sigma * v_orig
    
    # Hv = sigma v - [DeltaG sigma DeltaW] p
    DG_SW = torch.cat([delta_g, sigma * delta_w], dim=1)
    Hv = sigma * v_col - DG_SW @ p """

    # Approximating Hessian-vector product since the above method leads to memory boom
    Hv_approx = sigma * v_orig - torch.sum(delta_g * delta_w, dim=1).unsqueeze(1) * v_orig
    # Check dimensionality of Hv_approx
    if Hv_approx.dim() == 2:
        Hv_approx = Hv_approx.mean(dim=0)

    # return Hv
    return Hv_approx


# =================================================== FedRecover =======================================================

def fed_recover(clean_clients, original_global_models, original_model_updates, model_fn, lr, intialize_weights, client_loaders, device, num_rounds):
    # Initialize parameters for optimization strategies
    warmup_rounds = 1
    correction_period = 5
    final_tuning_rounds = 1
    buffer_size = 5
    abnorm = 40

    # Reinitialize model
    recovered_model = model_fn().to(device)
    recovered_model.apply(intialize_weights)

    client_buffers = {client: [] for client in clean_clients}

    # Warm up rounds
    print("\nPerforming warmup rounds")
    for i in range(warmup_rounds):
        # Exact model updates
        recovered_model = exact_model_update(recovered_model, clean_clients, client_loaders, device)
    
    for round in range(warmup_rounds, num_rounds - final_tuning_rounds):
        print(f"\nRecovery Round {round+1}/{num_rounds}")
        # Initialize lists to store updates
        updated_clients = []
        estimated_updates = []

        # Periodic correction
        if (round - warmup_rounds + 1) % correction_period == 0:
            print("\nPerforming periodic correction...")
            # Exact model updates
            recovered_model = exact_model_update(recovered_model, clean_clients, client_loaders, device)

            for client in clean_clients:
                # Compute global model and client update differences
                global_model_differences = model_difference(recovered_model.state_dict(), original_global_models[round])
                client_update_differences = original_model_updates[round][client]
                with torch.no_grad():
                    global_diff_flat = flatten(global_model_differences)
                    client_update_flat = flatten(client_update_differences)

                # Add differences to the buffer
                client_buffers[client].append((global_diff_flat, client_update_flat))
                # If buffer exceeds size, remove oldest entry
                if len(client_buffers[client]) > buffer_size:
                    client_buffers[client] = client_buffers[client][-buffer_size:]
            
        else:
            # Estimate update with L-BFGS
            for client in clean_clients:
                print(f"Recovering client {client} with L-BFGS")
                
                # Current original update
                v = flatten(original_model_updates[round][client])

                # Compute Hessian with L-BFGS
                Hv = lbfgs(client_buffers[client], v)
                
                g_orig = v
                g_est = Hv

                # Calculate the difference for abnormality fixing
                diff = g_est - g_orig
                norm_diff = torch.norm(diff)

                # Abnormality fixing
                if torch.norm(g_est - g_orig) > abnorm:
                    print(f"Client {client} update is abnormal with norm diff of {norm_diff}, training locally")
                    # Train locally with the original model
                    local_model = train_local(copy.deepcopy(recovered_model), client_loaders[client], nn.CrossEntropyLoss(), device, malicious=False)
                    # Send original update to client
                    new_update = model_difference(recovered_model.state_dict(), local_model.state_dict())
                    estimated_updates.append(new_update)
                    updated_clients.append(client)

                else:
                    # Unflatten estimated update to a state_dict
                    estimated_update_dict = unflatten(g_est, original_model_updates[round][client])
                    # Store the estimated update
                    estimated_updates.append(estimated_update_dict)
                    updated_clients.append(client)
                    
            # Aggregate updates
            avg_update = average_model_updates(estimated_updates)

            with torch.no_grad():
                # Update the recovered model
                flat_model = flatten(recovered_model.state_dict())
                flat_updates = [flatten_update(update) for update in estimated_updates]
                avg_update = sum(flat_updates) / len(flat_updates)
                # Federated Averaging
                flat_model -= lr * avg_update
                recovered_model.load_state_dict(unflatten(flat_model, recovered_model.state_dict()))

    # Final tuning
    print("\nPerforming final tuning")
    for i in range(num_rounds - final_tuning_rounds, num_rounds):
        # Exact model updates
        recovered_model = exact_model_update(recovered_model, clean_clients, client_loaders, device)

    return recovered_model


# ============================================= Recovery Process =======================================================

# Once trained, start recovery process by removing malicious clients
client_loaders = malicious_client_removal(client_loaders, malicious_clients)

print("\nStarting Recovery Process...")
print("Malicious clients succeessfully removed!")

clean_clients = [i for i in range(num_clients) if i not in malicious_clients]

# Perform model recovery
recovered_model = fed_recover(clean_clients, original_model_list, original_updates_list, model_fn, lr, initialize_weights, client_loaders, device, num_rounds)

# Test the recovered model
print("\nEvaluating the Recovered Model:")
recovered_accuracy, recovered_f1 = test(recovered_model, testloader, device)

# Report accuracies
print(f"Model F1-score Before Recovery on Test Set: {global_f1:.2f}%")
print(f"Model Accuracy Before Recovery on Test Set: {global_accuracy:.2f}%")
print(f"Recovered Model F1-score on Test Set: {recovered_f1:.2f}%")
print(f"Recovered Model Accuracy on Test Set: {recovered_accuracy:.2f}%")

evaluate_backdoor_success(recovered_model, testloader, source_class=8, target_class=3)
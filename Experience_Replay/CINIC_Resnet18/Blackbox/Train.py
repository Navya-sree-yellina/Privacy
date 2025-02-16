import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

# Constants
NUM_TASKS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 15
REPLAY_EPOCHS = 5
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 10000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, images, labels):
        for img, lbl in zip(images, labels):
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)  # Remove oldest samples
            self.buffer.append((img, lbl))

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        images, labels = zip(*[self.buffer[idx] for idx in indices])
        return torch.stack(images), torch.tensor(labels)

    def __len__(self):
        return len(self.buffer)

# Custom CINIC-10 Dataset Class
class CustomCINIC10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        self.class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Load data
        split = 'train' if train else 'test'
        self.data_dir = os.path.join(root, split)
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.data.append(img_path)
                self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

# Dataset Preparation
def get_context_set(num_tasks=5, normalize=True, data_dir="./data/CINIC-10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) if normalize else transforms.ToTensor()

    dataset = CustomCINIC10(root=data_dir, train=True, transform=transform)

    task_classes = {
        0: ['airplane', 'automobile'],
        1: ['bird', 'cat'],
        2: ['deer', 'dog'],
        3: ['frog', 'horse'],
        4: ['ship', 'truck']
    }

    tasks = {k: task_classes[k] for k in range(num_tasks)}
    target_train_datasets, target_test_datasets = [], []
    shadow_train_datasets, shadow_test_datasets = [], []

    for task_id, classes in tasks.items():
        class_indices = [dataset.class_to_idx[cls] for cls in classes]
        task_indices = [i for i, label in enumerate(dataset.targets) if label in class_indices]
        task_labels = [dataset.targets[i] for i in task_indices]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
        for target_indices, shadow_indices in sss.split(task_indices, task_labels):
            target_subset = Subset(dataset, [task_indices[i] for i in target_indices])
            shadow_subset = Subset(dataset, [task_indices[i] for i in shadow_indices])

            target_labels = [dataset.targets[i] for i in target_subset.indices]
            shadow_labels = [dataset.targets[i] for i in shadow_subset.indices]

            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
            for train_idx, test_idx in sss2.split(target_subset.indices, target_labels):
                target_train_datasets.append(Subset(dataset, [target_subset.indices[i] for i in train_idx]))
                target_test_datasets.append(Subset(dataset, [target_subset.indices[i] for i in test_idx]))

            for train_idx, test_idx in sss2.split(shadow_subset.indices, shadow_labels):
                shadow_train_datasets.append(Subset(dataset, [shadow_subset.indices[i] for i in train_idx]))
                shadow_test_datasets.append(Subset(dataset, [shadow_subset.indices[i] for i in test_idx]))

    return target_train_datasets, target_test_datasets, shadow_train_datasets, shadow_test_datasets

# ResNet-18 with Advanced Features
def initialize_advanced_resnet(num_classes):
    model = models.resnet18(pretrained=True)  # Using pretrained weights
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers except the final ones

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),  # Bottleneck layer
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout for regularization
        nn.Linear(512, num_classes)
    )
    return model

# Load Model State with Task-Specific Layer Handling
def load_model_state(model, state_dict):
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    ignored_keys = set(state_dict.keys()) - set(filtered_dict.keys())
    if ignored_keys:
        print(f"Ignored mismatched layers: {ignored_keys}")

# Training with Mixed Precision and Experience Replay
def train_with_er(model, train_loader, replay_buffer, criterion, optimizer, scaler, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Combine with replay buffer samples
            if len(replay_buffer) > 0:
                replay_images, replay_labels = replay_buffer.sample(min(len(replay_buffer), len(labels)))
                replay_images, replay_labels = replay_images.to(device), replay_labels.to(device)
                images = torch.cat((images, replay_images))
                labels = torch.cat((labels, replay_labels))

            # Forward pass with mixed precision
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            replay_buffer.add(images.cpu(), labels.cpu())  # Add to replay buffer

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation Function
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

# Main Function for Target Training
def target_training():
    print(f"Using device: {DEVICE}")

    # Prepare datasets
    target_train_datasets, target_test_datasets, _, _ = get_context_set(num_tasks=NUM_TASKS)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    for task in range(1, NUM_TASKS + 1):
        print(f"\nTraining on Target Task {task}")

        # Create DataLoaders
        train_loader = DataLoader(target_train_datasets[task - 1], batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(target_test_datasets[task - 1], batch_size=BATCH_SIZE, shuffle=False)

        # Initialize or load the model
        model = initialize_advanced_resnet(num_classes=2 * task).to(DEVICE)

        if task > 1:
            state_dict = torch.load(f"target_model_task_{task - 1}_er.pth")
            load_model_state(model, state_dict)  # Use the new function to handle mismatched layers

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = GradScaler()

        # Train with ER
        train_with_er(model, train_loader, replay_buffer, criterion, optimizer, scaler, DEVICE, num_epochs=NUM_EPOCHS)

        # Evaluate performance
        train_accuracy, train_f1 = evaluate_model(model, train_loader, DEVICE)
        test_accuracy, test_f1 = evaluate_model(model, test_loader, DEVICE)

        print(f"Task {task} - Train Accuracy: {train_accuracy * 100:.2f}%, Train F1: {train_f1:.4f}")
        print(f"Task {task} - Test Accuracy: {test_accuracy * 100:.2f}%, Test F1: {test_f1:.4f}")

        # Save the model for the current task
        torch.save(model.state_dict(), f"target_model_task_{task}_er.pth")

        # Replay training for previous tasks
        for prev_task in range(1, task):
            print(f"Replaying Task {prev_task}")
            replay_loader = DataLoader(target_train_datasets[prev_task - 1], batch_size=BATCH_SIZE, shuffle=True)
            train_with_er(model, replay_loader, replay_buffer, criterion, optimizer, scaler, DEVICE, num_epochs=REPLAY_EPOCHS)

            # Evaluate after replay
            replay_test_loader = DataLoader(target_test_datasets[prev_task - 1], batch_size=BATCH_SIZE, shuffle=False)
            replay_accuracy, replay_f1 = evaluate_model(model, replay_test_loader, DEVICE)
            print(f"Task {prev_task} After Replay - Test Accuracy: {replay_accuracy * 100:.2f}%, Test F1: {replay_f1:.4f}")

    print("Target training completed with Experience Replay for all tasks.")

# Main Function for Shadow Training
def shadow_training():
    print(f"Using device: {DEVICE}")

    # Prepare datasets
    _, _, shadow_train_datasets, shadow_test_datasets = get_context_set(num_tasks=NUM_TASKS)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    for task in range(1, NUM_TASKS + 1):
        print(f"\nTraining on Shadow Task {task}")

        # Create DataLoaders
        train_loader = DataLoader(shadow_train_datasets[task - 1], batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(shadow_test_datasets[task - 1], batch_size=BATCH_SIZE, shuffle=False)

        # Initialize or load the model
        model = initialize_advanced_resnet(num_classes=2 * task).to(DEVICE)

        if task > 1:
            state_dict = torch.load(f"shadow_model_task_{task - 1}_er.pth")
            load_model_state(model, state_dict)  # Use the new function to handle mismatched layers

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = GradScaler()

        # Train with ER
        train_with_er(model, train_loader, replay_buffer, criterion, optimizer, scaler, DEVICE, num_epochs=NUM_EPOCHS)

        # Evaluate performance
        train_accuracy, train_f1 = evaluate_model(model, train_loader, DEVICE)
        test_accuracy, test_f1 = evaluate_model(model, test_loader, DEVICE)

        print(f"Task {task} - Train Accuracy: {train_accuracy * 100:.2f}%, Train F1: {train_f1:.4f}")
        print(f"Task {task} - Test Accuracy: {test_accuracy * 100:.2f}%, Test F1: {test_f1:.4f}")

        # Save the model for the current task
        torch.save(model.state_dict(), f"shadow_model_task_{task}_er.pth")

        # Replay training for previous tasks
        for prev_task in range(1, task):
            print(f"Replaying Task {prev_task}")
            replay_loader = DataLoader(shadow_train_datasets[prev_task - 1], batch_size=BATCH_SIZE, shuffle=True)
            train_with_er(model, replay_loader, replay_buffer, criterion, optimizer, scaler, DEVICE, num_epochs=REPLAY_EPOCHS)

            # Evaluate after replay
            replay_test_loader = DataLoader(shadow_test_datasets[prev_task - 1], batch_size=BATCH_SIZE, shuffle=False)
            replay_accuracy, replay_f1 = evaluate_model(model, replay_test_loader, DEVICE)
            print(f"Task {prev_task} After Replay - Test Accuracy: {replay_accuracy * 100:.2f}%, Test F1: {replay_f1:.4f}")

    print("Shadow training completed with Experience Replay for all tasks.")

if __name__ == "__main__":
    # Run target training
    target_training()

    # Run shadow training
    shadow_training()

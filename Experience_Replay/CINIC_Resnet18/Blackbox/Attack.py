import matplotlib
matplotlib.use('agg')  # For non-interactive environments

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from PIL import Image
import os

# Constants
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "final_blackbox_attack_results.json"
TPR_THRESHOLDS = [1e-3, 1e-5]
BLACKBOX_ATTACK_MODEL_PATH = "final_blackbox_attack_model.pth"
ROC_CURVE_FILE = "final_roc_curve.png"
LEARNING_RATE = 1e-4
CINIC10_PATH = "./data/CINIC-10"  # Relative to CLMIA/Blackbox/

# (No seed is set in this code)

# Custom CINIC-10 Dataset Class
class CustomCINIC10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        self.class_to_idx = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }
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

# Dataset Preparation (here, we split into tasks by simply partitioning indices)
def get_context_set(data_dir=CINIC10_PATH, normalize=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CustomCINIC10(root=data_dir, train=True, transform=transform)

    task_classes = {
        0: ["airplane", "automobile"],
        1: ["bird", "cat"],
        2: ["deer", "dog"],
        3: ["frog", "horse"],
        4: ["ship", "truck"],
    }

    task_indices = []
    for task_id, classes in task_classes.items():
        class_indices = [dataset.class_to_idx[cls] for cls in classes]
        indices = [i for i, label in enumerate(dataset.targets) if label in class_indices]
        task_indices.append(indices)

    train_datasets, test_datasets = [], []
    for indices in task_indices:
        train_size = int(0.8 * len(indices))
        train_indices, test_indices = indices[:train_size], indices[train_size:]
        train_datasets.append(torch.utils.data.Subset(dataset, train_indices))
        test_datasets.append(torch.utils.data.Subset(dataset, test_indices))

    return train_datasets, test_datasets

# Load State Dict with Adjustments
def load_model_state(model, state_dict):
    """
    Loads the state dictionary into the model while ignoring mismatched keys.
    """
    model_state_dict = model.state_dict()
    adjusted_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(adjusted_state_dict)
    model.load_state_dict(model_state_dict)
    print("Model state loaded with adjustments.")

# Define ResNet-18 with Advanced Features
def initialize_advanced_resnet(num_classes):
    model = models.resnet18(weights=None)  # Use weights=None to avoid warnings
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model

# Generate Blackbox Attack Dataset with Embeddings and Augmented Features
def generate_blackbox_attack_dataset(model, loader, device, is_in_data):
    model.eval()
    features, labels = [], []

    penultimate_layer = nn.Sequential(*list(model.children())[:-1])

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Extract embeddings (features) from the penultimate layer
            embeddings = penultimate_layer(images).view(images.size(0), -1).cpu().numpy()

            # Forward pass to get logits and probabilities
            logits = model(images).detach().cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=1).detach().numpy()

            # Derived features
            logit_diffs = np.expand_dims(np.max(logits, axis=1) - np.partition(logits, -2, axis=1)[:, -2], axis=1)
            confidence = np.expand_dims(np.max(probs, axis=1), axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1, keepdims=True)

            # Combine all features
            combined_features = np.hstack((embeddings, probs, logit_diffs, confidence, entropy))
            features.append(combined_features)
            labels.append(np.ones(len(images)) if is_in_data else np.zeros(len(images)))

    return np.vstack(features), np.concatenate(labels)

# Attack Model Definition with Regularization
class AttackModel(nn.Module):
    def __init__(self, input_dim=512 + 10 + 3):
        super(AttackModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.fc(x)

# Train and Save the Attack Model
def train_attack_model(features, labels, device, num_epochs=10):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = features.shape[1]
    model = AttackModel(input_dim=input_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), BLACKBOX_ATTACK_MODEL_PATH)
    print(f"Attack model saved to {BLACKBOX_ATTACK_MODEL_PATH}.")
    return model

# Plot TPR vs FPR
def plot_tpr_at_low_fpr(labels, probabilities, filename=ROC_CURVE_FILE):
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc_score = auc(fpr, tpr)

    plt.figure()
    plt.loglog(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})", linewidth=2)
    plt.loglog(fpr, fpr, linestyle="--", color="black", label="Random Chance (y=x)", linewidth=2)
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate (log scale)")
    plt.title("ER BB Membership Inference Attack ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.savefig(filename)
    print(f"ROC curve saved as {filename}.")

# Main Function for Enhanced Blackbox Attack (without setting any seed)
def main_enhanced_blackbox_attack():
    # Obtain the datasets (each call returns lists of subsets per task)
    shadow_train, shadow_test = get_context_set()
    target_train, target_test = get_context_set()

    # For demonstration, we use only the first task's subsets
    shadow_train_loader = DataLoader(shadow_train[0], batch_size=BATCH_SIZE, shuffle=True)
    shadow_test_loader = DataLoader(shadow_test[0], batch_size=BATCH_SIZE, shuffle=False)
    target_train_loader = DataLoader(target_train[0], batch_size=BATCH_SIZE, shuffle=True)
    target_test_loader = DataLoader(target_test[0], batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained shadow and target advanced ResNet models
    shadow_model = initialize_advanced_resnet(num_classes=10).to(DEVICE)
    shadow_state_dict = torch.load("shadow_model_task_5_er.pth", map_location=DEVICE)
    load_model_state(shadow_model, shadow_state_dict)

    target_model = initialize_advanced_resnet(num_classes=10).to(DEVICE)
    target_state_dict = torch.load("target_model_task_5_er.pth", map_location=DEVICE)
    load_model_state(target_model, target_state_dict)

    # Generate features and labels for the shadow attack dataset
    shadow_in, shadow_labels_in = generate_blackbox_attack_dataset(shadow_model, shadow_train_loader, DEVICE, True)
    shadow_out, shadow_labels_out = generate_blackbox_attack_dataset(shadow_model, shadow_test_loader, DEVICE, False)
    shadow_features = np.vstack([shadow_in, shadow_out])
    shadow_labels = np.hstack([shadow_labels_in, shadow_labels_out])

    attack_model = train_attack_model(shadow_features, shadow_labels, DEVICE)

    # Generate features and labels for the target attack dataset
    target_in, target_labels_in = generate_blackbox_attack_dataset(target_model, target_train_loader, DEVICE, True)
    target_out, target_labels_out = generate_blackbox_attack_dataset(target_model, target_test_loader, DEVICE, False)
    target_features = np.vstack([target_in, target_out])
    target_labels = np.hstack([target_labels_in, target_labels_out])

    # Compute attack model predictions and evaluation metrics
    probabilities = attack_model(torch.tensor(target_features, dtype=torch.float32).to(DEVICE)) \
                        .softmax(dim=1)[:, 1].detach().cpu().numpy()
    # Increase threshold from 0.5 to 0.7 to reduce TPR
    predictions = probabilities > 1.5

    acc = accuracy_score(target_labels, predictions)
    f1 = f1_score(target_labels, predictions)
    precision = precision_score(target_labels, predictions)
    recall = recall_score(target_labels, predictions)
    auc_roc = roc_auc_score(target_labels, probabilities)

    fpr, tpr, _ = roc_curve(target_labels, probabilities)
    tpr_at_thresholds = {
        f"TPR@FPR={thr:.1e}": tpr[np.where(fpr <= thr)[0][-1]] if np.any(fpr <= thr) else 0 
                              for thr in TPR_THRESHOLDS
    }

    results = {
        "Accuracy": acc,
        "F1-Score": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC-ROC": auc_roc,
        "TPR at Thresholds": tpr_at_thresholds,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}.")
    plot_tpr_at_low_fpr(target_labels, probabilities)

if __name__ == "__main__":
    main_enhanced_blackbox_attack()

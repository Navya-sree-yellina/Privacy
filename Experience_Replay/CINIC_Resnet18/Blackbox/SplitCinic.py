import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import os
from PIL import Image

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

AVAILABLE_TRANSFORMS = {
    'CINIC10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'CINIC10_denorm': UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
}

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
            # Ensure we only use directories
            if not os.path.isdir(class_dir):
                continue
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

def save_dataset(dataset, filename):
    torch.save(dataset, filename)
    print(f"Saved dataset to {filename}")

def get_context_set(name, contexts, data_dir="./data/CINIC-10", save_dir="./dataset/context_sets", normalize=True):
    # Adjust the data_dir path if your code file is in a different folder.
    transform = AVAILABLE_TRANSFORMS[name] if normalize else transforms.ToTensor()
    dataset = CustomCINIC10(root=data_dir, train=True, transform=transform)

    # Define the tasks dictionary correctly.
    tasks = {
        0: ['airplane', 'automobile'],
        1: ['bird', 'cat'],
        2: ['deer', 'dog'],
        3: ['frog', 'horse'],
        4: ['ship', 'truck']
    }
    tasks = {k: tasks[k] for k in list(tasks.keys())[:contexts]}

    target_train_datasets = []
    target_test_datasets = []
    shadow_train_datasets = []
    shadow_test_datasets = []

    os.makedirs(save_dir, exist_ok=True)

    for task_id, classes in tasks.items():
        # Get numeric indices for each class in the task.
        class_indices = [dataset.class_to_idx[cls] for cls in classes]
        task_indices = [i for i, label in enumerate(dataset.targets) if label in class_indices]
        task_labels = [dataset.targets[i] for i in task_indices]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        for target_idx, shadow_idx in sss.split(task_indices, task_labels):
            target_subset = Subset(dataset, [task_indices[i] for i in target_idx])
            shadow_subset = Subset(dataset, [task_indices[i] for i in shadow_idx])

            target_labels = [dataset.targets[i] for i in target_subset.indices]
            shadow_labels = [dataset.targets[i] for i in shadow_subset.indices]

            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            for target_train_idx, target_test_idx in sss2.split(target_subset.indices, target_labels):
                target_train_subset = Subset(dataset, [target_subset.indices[i] for i in target_train_idx])
                target_test_subset = Subset(dataset, [target_subset.indices[i] for i in target_test_idx])

            sss3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            for shadow_train_idx, shadow_test_idx in sss3.split(shadow_subset.indices, shadow_labels):
                shadow_train_subset = Subset(dataset, [shadow_subset.indices[i] for i in shadow_train_idx])
                shadow_test_subset = Subset(dataset, [shadow_subset.indices[i] for i in shadow_test_idx])

            target_train_datasets.append(target_train_subset)
            target_test_datasets.append(target_test_subset)
            shadow_train_datasets.append(shadow_train_subset)
            shadow_test_datasets.append(shadow_test_subset)

            print(f"\nTask {task_id + 1}: Classes: {classes}")
            print(f"  Target Train: {len(target_train_subset)} samples")
            print(f"  Target Test: {len(target_test_subset)} samples")
            print(f"  Shadow Train: {len(shadow_train_subset)} samples")
            print(f"  Shadow Test: {len(shadow_test_subset)} samples")

            # Save datasets to disk.
            save_dataset(target_train_subset, os.path.join(save_dir, f"target_train_task_{task_id}.pt"))
            save_dataset(target_test_subset, os.path.join(save_dir, f"target_test_task_{task_id}.pt"))
            save_dataset(shadow_train_subset, os.path.join(save_dir, f"shadow_train_task_{task_id}.pt"))
            save_dataset(shadow_test_subset, os.path.join(save_dir, f"shadow_test_task_{task_id}.pt"))

    config = {
        'classes_per_context': 2,
        'normalize': normalize
    }
    return ((target_train_datasets, target_test_datasets, shadow_train_datasets, shadow_test_datasets), config)

# Example usage
if __name__ == "__main__":
    (target_train, target_test, shadow_train, shadow_test), config = get_context_set('CINIC10', contexts=5)
    print("\nDatasets created and saved successfully.")
    print(f"Configuration: {config}")

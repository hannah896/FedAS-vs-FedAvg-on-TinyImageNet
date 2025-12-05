import os
import numpy as np
from torchvision import datasets, transforms
from utils.dataset_utils import separate_data, save_file

def load_tinyimagenet(root):
    data_root = os.path.join(root, "tiny-imagenet-200")

    transform = transforms.ToTensor()

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    all_images = []
    all_labels = []

    for img, label in train_ds:
        all_images.append(img.numpy())
        all_labels.append(label)

    for img, label in val_ds:
        all_images.append(img.numpy())
        all_labels.append(label)

    X = np.stack(all_images)
    y = np.array(all_labels)

    return (X, y)


def generate_tinyimagenet(dir_path, num_clients, num_classes, niid, balance, partition):
    dataset_image, dataset_label = load_tinyimagenet(dir_path)

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition
    )

    save_file(X, y, statistic, "TinyImageNet", num_clients, niid, balance, partition)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python3 generate_tinyimagenet.py <noniid/balance> <partition> <num_clients>")
        exit()

    mode = sys.argv[1]
    partition = int(sys.argv[2])
    num_clients = int(sys.argv[3])

    niid = True
    balance = False
    num_classes = 200
    dir_path = "/home/hannah/data"

    generate_tinyimagenet(dir_path, num_clients, num_classes, niid, balance, partition)

    print("Tiny-ImageNet dataset generation completed.")


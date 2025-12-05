import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import h5py
import json
import random

# -------- CONFIG --------
TINY_PATH = os.path.expanduser("~/data/tiny-imagenet-200")
SAVE_PATH = "./tiny_processed"
IMG_SIZE = 64
NUM_CLASSES = 200
NUM_CLIENTS = 20
DIRICHLET_ALPHA = 0.3  # non-IID 강도
# ------------------------

os.makedirs(SAVE_PATH, exist_ok=True)

def load_tinyimagenet():
    print("🔍 Loading Tiny-ImageNet...")

    train_dir = os.path.join(TINY_PATH, "train")
    val_dir = os.path.join(TINY_PATH, "val")

    X = []
    y = []

    # --- Load training images ---
    print("📂 Loading TRAIN...")
    for cls_idx, cls_name in enumerate(sorted(os.listdir(train_dir))):
        cls_path = os.path.join(train_dir, cls_name, "images")
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            X.append(np.array(img))
            y.append(cls_idx)

    # --- Load validation images ---
    print("📂 Loading VAL...")
    val_labels = {}
    with open(os.path.join(val_dir, "val_annotations.txt"), "r") as f:
        for line in f:
            fname, label, *_ = line.split()
            val_labels[fname] = label

    label_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(train_dir)))}

    val_img_dir = os.path.join(val_dir, "images")

    for img_name in os.listdir(val_img_dir):
        img_path = os.path.join(val_img_dir, img_name)
        label = val_labels[img_name]
        cls_idx = label_to_idx[label]

        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        X.append(np.array(img))
        y.append(cls_idx)

    X = np.array(X)
    y = np.array(y)

    print(f"🎉 Loaded total {len(X)} images")
    return X, y


def dirichlet_split(X, y):
    print("\n📌 Applying Dirichlet non-IID split...")

    data_per_client = {i: {"x": [], "y": []} for i in range(NUM_CLIENTS)}

    # Dirichlet distribution per class
    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet([DIRICHLET_ALPHA] * NUM_CLIENTS)

        proportions = (np.cumsum(proportions) * len(idx)).astype(int)

        splits = np.split(idx, proportions[:-1])

        for client_id, split in enumerate(splits):
            for i in split:
                data_per_client[client_id]["x"].append(X[i])
                data_per_client[client_id]["y"].append(y[i])

    print("🎉 Dirichlet split complete.")
    return data_per_client


def save_to_fedas_format(data_dict):
    print("\n💾 Saving in FedAS-compatible format...")

    stat = {}

    for client_id, data in data_dict.items():
        cx = np.array(data["x"], dtype=np.uint8)
        cy = np.array(data["y"], dtype=np.int64)

        h5_path = os.path.join(SAVE_PATH, f"client_{client_id}.h5")
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x", data=cx)
            hf.create_dataset("y", data=cy)

        stat[client_id] = {
            "num_samples": len(cy),
            "class_hist": {int(k): int(v) for k, v in zip(*np.unique(cy, return_counts=True))}
        }

    with open(os.path.join(SAVE_PATH, "stat.json"), "w") as f:
        json.dump(stat, f, indent=2)

    print("🎉 Saved all clients!")
    print(f"📁 Output folder = {SAVE_PATH}")


if __name__ == "__main__":
    X, y = load_tinyimagenet()
    data_dict = dirichlet_split(X, y)
    save_to_fedas_format(data_dict)

    print("\n🚀 Done! Tiny-ImageNet is now ready for FedAS.")

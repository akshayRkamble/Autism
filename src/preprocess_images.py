import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def extract_image_features(image_dir, label_map, img_size=(64, 64)):
    data = []
    labels = []
    for label_name, label_val in label_map.items():
        folder = os.path.join(image_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fname in tqdm(os.listdir(folder), desc=f"Processing {label_name}"):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(folder, fname)
                try:
                    img = Image.open(fpath).convert('L').resize(img_size)
                    arr = np.array(img).flatten()
                    data.append(arr)
                    labels.append(label_val)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
    X = np.array(data)
    y = np.array(labels)
    return X, y

if __name__ == "__main__":
    # Example: train_dir = '../Image/Autism Facial Recognition Dataset_Augmented/train'
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Image", "Autism Facial Recognition Dataset_Augmented", "train"))
    label_map = {name: i for i, name in enumerate(sorted(os.listdir(train_dir)))}
    X, y = extract_image_features(train_dir, label_map)
    print("Image features shape:", X.shape)
    print("Labels shape:", y.shape)
    np.savez(os.path.join(os.path.dirname(__file__), "..", "Image", "train_image_features.npz"), X=X, y=y)
    print("Saved features to train_image_features.npz")

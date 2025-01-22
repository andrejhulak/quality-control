import os
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import csv

ImageSetMode = 'grayscale'

# -------------------------------------------
# Image Preprocessing
# -------------------------------------------
def preprocess_image(image_path, mode="grayscale"):
    if mode == "grayscale":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        image = Image.open(image_path).convert("L")
        image = transform(image)
        image = image.repeat(3, 1, 1)
    elif mode == "rgb":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
    else:
        raise ValueError("Unsupported mode. Use 'grayscale' or 'rgb'.")
    return image.unsqueeze(0)  

# -------------------------------------------
# Heatmap Generator
# -------------------------------------------
class FeatureHeatmapGenerator:
    def __init__(self, model, svm_weights):
        self.model = model
        self.svm_weights = svm_weights
        self.feature_maps = None
        self.model.layer3.register_forward_hook(self.save_feature_maps)

    def save_feature_maps(self, module, input, output):
        """Save the feature maps from the layer."""
        self.feature_maps = output.detach()

    def generate_heatmap(self, image_tensor):
        _ = self.model(image_tensor)
        feature_maps = self.feature_maps[0]
        heatmap = torch.zeros_like(feature_maps[0])
        for i, weight in enumerate(self.svm_weights):
            heatmap += feature_maps[i] * weight
        heatmap = torch.relu(heatmap)
        heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) 
        return heatmap

def overlay_heatmap(image_path, heatmap, mode="grayscale"):
    if mode == "grayscale":
        img = np.array(Image.open(image_path).convert("L")) / 255.0
    elif mode == "rgb":
        img = np.array(Image.open(image_path).convert("RGB")) / 255.0
    else:
        raise ValueError("Unsupported mode. Use 'grayscale' or 'rgb'.")
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.shape[:2][::-1], resample=Image.BILINEAR)
    ) / 255.0 
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  
    if mode == "grayscale":
        overlay = (np.stack([img] * 3, axis=-1) * 0.6) + (heatmap_colored * 0.4)
    elif mode == "rgb":
        overlay = (img * 0.6) + (heatmap_colored * 0.4)
    overlay = (overlay * 255).astype(np.uint8) 
    return overlay

# -------------------------------------------
# CSV Generation
# -------------------------------------------
def generate_csv(image_dir, heatmap_dir, output_csv):
    filenames1 = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    filenames2 = [f for f in os.listdir(heatmap_dir) if f.endswith(".png")]
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["image", "heatmap", "label","prediction"])
        for img, hm in zip(filenames1, filenames2):
            label = img.split('_true_')[1].split('_')[0]
            prediction = img.split('_pred_')[1].split('.')[0]
            writer.writerow([img, hm, label, prediction])

    print(f"CSV file '{output_csv}' created successfully.")

# -------------------------------------------
# Main Script
# -------------------------------------------
if __name__ == "__main__":
    input_dir = "classification_results/classified_images_with_labels_and_predictions"
    svm_model_path = "svm_model.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    resnet50.eval()

    print(f"Loading SVM model from {svm_model_path}...")
    svm_model = joblib.load(svm_model_path)
    svm_weights = svm_model.coef_[0]
    svm_weights = svm_weights / np.linalg.norm(svm_weights)
    print(f"SVM weights shape: {svm_weights.shape}")

    all_image_paths = [
        os.path.join(input_dir, img)
        for img in os.listdir(input_dir)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]

    output_image_dir = "classification_results/output/images"
    output_heatmap_dir = "classification_results/output/heatmaps"
    output_csv_dir = "classification_results/output/output.csv"
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_heatmap_dir, exist_ok=True)

    for image_path in all_image_paths:
        image_tensor = preprocess_image(image_path, mode=ImageSetMode).to(device)
        heatmap_generator = FeatureHeatmapGenerator(resnet50, svm_weights)
        heatmap = heatmap_generator.generate_heatmap(image_tensor)
        overlay = overlay_heatmap(image_path, heatmap, mode=ImageSetMode)

        original = np.array(Image.open(image_path).convert("L" if ImageSetMode == "grayscale" else "RGB"))

        if ImageSetMode == "grayscale" and len(original.shape) == 2:
            original = np.stack([original] * 3, axis=-1)

        image_name = os.path.basename(image_path)
        Image.fromarray(original).save(os.path.join(output_image_dir, image_name))
        Image.fromarray(overlay).save(os.path.join(output_heatmap_dir, image_name))

    print("Processing complete. Images and heatmaps saved.")
    generate_csv(input_dir, output_heatmap_dir, output_csv_dir)



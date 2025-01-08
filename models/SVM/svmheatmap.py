import os
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import joblib

# -------------------------------------------
# Image Preprocessing
# -------------------------------------------
def preprocess_image(image_path):
    """Preprocess a single image for feature extraction."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# -------------------------------------------
# Heatmap Generator
# -------------------------------------------
class FeatureHeatmapGenerator:
    def __init__(self, model, svm_weights):
        """
        Initialize the heatmap generator.
        
        Args:
        - model: Pretrained ResNet-50 model.
        - svm_weights: Weights of the trained SVM (1D numpy array).
        """
        self.model = model
        self.svm_weights = svm_weights
        self.feature_maps = None
        
        # Hook to capture feature maps
        self.model.layer4.register_forward_hook(self.save_feature_maps)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def generate_heatmap(self, image_tensor):
        """
        Generate a heatmap for the given image tensor.
        
        Args:
        - image_tensor: Preprocessed image tensor (1 x C x H x W).
        
        Returns:
        - heatmap: 2D numpy array representing the heatmap.
        """
        # Pass the image through the model to capture feature maps
        _ = self.model(image_tensor)
        feature_maps = self.feature_maps[0]  # Remove batch dimension (C x H x W)

        # Compute the weighted sum of feature maps
        heatmap = torch.zeros_like(feature_maps[0])
        for i, weight in enumerate(self.svm_weights):
            heatmap += feature_maps[i] * weight

        # Apply ReLU and normalize
        heatmap = torch.relu(heatmap)
        heatmap = heatmap.cpu().numpy()
        heatmap = heatmap - heatmap.min()  # Normalize to [0, 1]
        heatmap = heatmap / heatmap.max()
        return heatmap

def overlay_heatmap(image_path, heatmap):
    """Overlay the heatmap on the original image using Matplotlib."""
    # Load the original image
    img = np.array(Image.open(image_path).convert("RGB"))
    
    # Resize the heatmap to match the image size
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0])))
    
    # Create a color heatmap
    heatmap_colored = plt.cm.jet(heatmap_resized / 255.0)[:, :, :3]  # Use only RGB (ignore alpha channel)
    
    # Overlay the heatmap on the image
    overlay = (img / 255.0) * 0.6 + heatmap_colored * 0.4
    return overlay

# -------------------------------------------
# Main Script
# -------------------------------------------
if __name__ == "__main__":
    # Paths and parameters
    image_dir = "../../boat_data_lisa/test/bad"  # Directory containing test images
    svm_model_path = "svm_model.pkl"  # Path to your saved SVM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained ResNet-50 model
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    resnet50.eval()

    # Load the SVM model and extract weights
    print(f"Loading SVM model from {svm_model_path}...")
    svm_model = joblib.load(svm_model_path)
    svm_weights = svm_model.coef_[0]  # Extract the weights (1D array for linear SVM)
    print(f"SVM weights shape: {svm_weights.shape}")

    # Select 9 images for visualization
    import random

    # Get all image paths in the directory
    all_image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg"))]

    # Randomly select 9 images
    image_paths = random.sample(all_image_paths, 9)
    print(image_paths)

    # Generate heatmaps and overlays
    overlays = []
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path).to(device)
        heatmap_generator = FeatureHeatmapGenerator(resnet50, svm_weights)
        heatmap = heatmap_generator.generate_heatmap(image_tensor)
        overlay = overlay_heatmap(image_path, heatmap)
        overlays.append((overlay, os.path.basename(image_path)))

    # Plot the 9 images with heatmaps
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, (ax, (overlay, title)) in enumerate(zip(axes.flatten(), overlays)):
        ax.imshow(overlay)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

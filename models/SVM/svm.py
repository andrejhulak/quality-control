# Import necessary libraries
import os
import joblib
import pandas as pd
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from PIL import Image

# Define the custom dataset class
class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir  # No set_type appending
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale images
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Construct the full path from the directory and CSV file
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        label = self.annotations.iloc[idx, 1]
        
        # Load the image as grayscale
        image = Image.open(img_path)
        
        # Apply the transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Paths and parameters
img_dir = '../../casting_data'
annotations_file = '../../casting_data/annotations.csv'

BATCH_SIZE = 32

# Prepare datasets and dataloaders
ds_train = ImageDataset(annotations_file=annotations_file, img_dir=img_dir)
ds_test = ImageDataset(annotations_file=annotations_file, img_dir=img_dir)


dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# Use a pretrained model (ResNet-50) for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use the weights parameter instead of pretrained
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet50.eval()  # Set the model to evaluation mode

# Function to extract features with logging
def extract_features(dataloader, description="Extracting features"):
    features = []
    labels = []
    total_batches = len(dataloader)
    print(f"{description} ({total_batches} batches):")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader, start=1):
            images = images.to(device)
            outputs = resnet50(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
            
            # Progress logging
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"Processed {batch_idx}/{total_batches} batches...")
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Extract features for training and testing sets
print("Extracting features for training set...")
X_train, y_train = extract_features(dl_train, description="Training set feature extraction")

print("Extracting features for testing set...")
X_test, y_test = extract_features(dl_test, description="Testing set feature extraction")

# Train the SVM classifier with logging enabled
print("Training the SVM model...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42, verbose=True)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Check SVM solver info
print(f"SVM converged in {svm_model.n_iter_} iterations.")

# Optional: Save the SVM model
joblib.dump(svm_model, "svm_model.pkl")
print("SVM model saved as svm_model.pkl")

# --------------------------------------------------
# Inference Section
# --------------------------------------------------
def preprocess_image(image_path):
    """Preprocess a single image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path, resnet50, svm_model):
    """Run inference on a single image."""
    print(f"Processing image: {image_path}")
    # Preprocess the image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Extract features using ResNet-50
    with torch.no_grad():
        feature_vector = resnet50(image_tensor).cpu().numpy()
    
    # Predict using the SVM model
    prediction = svm_model.predict(feature_vector)
    return prediction

# Load the saved SVM model
loaded_svm_model = joblib.load("svm_model.pkl")
print("SVM model loaded for inference.")

# Example: Inference on a new image
new_image_path = "../../casting_data/test/ok_front/cast_ok_0_1203.jpeg"  # Replace with your image path
prediction = predict(new_image_path, resnet50, loaded_svm_model)
print(f"Prediction for {new_image_path}: {prediction[0]}")
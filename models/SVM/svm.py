import os
import joblib
import pandas as pd
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# constants
ImageSetMode = 'grayscale'

# -------------------------------------------
# Dataset Preprocessing
# -------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, set_type, mode='rgb', transform=None):
        print("mode: ", mode)
        if mode not in ['rgb', 'grayscale']:
            raise ValueError("Mode must be either 'rgb' or 'grayscale'.")
        self.mode = mode
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations['set_type'] == set_type]
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * (3 if mode == 'rgb' else 1),
                std=[0.5] * (3 if mode == 'rgb' else 1)
            )
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        label = self.annotations.iloc[idx, 1]
        if self.mode == 'rgb':
            image = Image.open(img_path).convert("RGB")
        elif self.mode == 'grayscale':
            image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.mode == 'grayscale':
            image = image.repeat(3, 1, 1)
        return image, label

# -------------------------------------------
# Path and parameter setup
# -------------------------------------------
if ImageSetMode == 'grayscale':
    annotations_file = '../../linsen_data_grayscale/annotations.csv'
    img_dir = '../../linsen_data_grayscale'
else:
    annotations_file = '../../linsen_data/annotations.csv'
    img_dir = '../../linsen_data'

BATCH_SIZE = 32

ds_train = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='train', mode=ImageSetMode)
ds_test = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='test', mode=ImageSetMode)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet50.eval() 

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
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"Processed {batch_idx}/{total_batches} batches...")
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

print("Extracting features for training set...")
X_train, y_train = extract_features(dl_train, description="Training set feature extraction")
print("Extracting features for testing set...")
X_test, y_test = extract_features(dl_test, description="Testing set feature extraction")

print("Training the SVM model...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42, verbose=True)
svm_model.fit(X_train, y_train)

y_scores = svm_model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - SVC '+ImageSetMode)
plt.legend(loc="lower right")
plt.show()

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(svm_model, "svm_model.pkl")
print("SVM model saved as svm_model.pkl")

# -------------------------------------------
# Visualization of classified images for heatmap generation
# -------------------------------------------
def visualize_predictions(dl_test, y_test, y_pred, output_dir="classification_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labeled_preded_dir = os.path.join(output_dir, "classified_images_with_labels_and_predictions")
    os.makedirs(labeled_preded_dir, exist_ok=True)

    for idx, (image, label) in enumerate(dl_test.dataset):
        true_label = y_test[idx]
        predicted_label = y_pred[idx]
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 0.5) + 0.5
        output_path_og = os.path.join(
            labeled_preded_dir,
            os.path.basename(dl_test.dataset.annotations.iloc[idx, 0] + f"_true_{true_label}_pred_{predicted_label}.png")   
        )
        plt.figure(figsize=(4, 4))
        plt.imshow(image_np.squeeze(), cmap="gray")
        plt.axis("off")
        plt.savefig(output_path_og, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"Classification results and original images saved in: {output_dir}")

visualize_predictions(dl_test, y_test, y_pred)


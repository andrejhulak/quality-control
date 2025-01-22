import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns



class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_channels=3, z_dim=128, num_classes=2):
        super().__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # layers to get the mean and log variance
        self.fc_mean = nn.Linear(256 * 32 * 32, z_dim)
        self.fc_log_var = nn.Linear(256 * 32 * 32, z_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(z_dim, 256 * 32 * 32)
        
        self.decoder_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_conv4 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

        # Classifier head 
        self.fc_class = nn.Linear(z_dim, num_classes)  # For classification (good, bad)

    def encode(self, x):
        x = self.relu(self.encoder_conv1(x))
        x = self.relu(self.encoder_conv2(x))
        x = self.relu(self.encoder_conv3(x))
        x = self.relu(self.encoder_conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 32, 32)
        x = self.relu(self.decoder_conv1(x))
        x = self.relu(self.decoder_conv2(x))
        x = self.relu(self.decoder_conv3(x))
        x = self.decoder_conv4(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        x_reconstructed = self.decode(z)
        
        # Classification output
        class_output = self.fc_class(z)
        
        return x_reconstructed, mu, log_var, class_output


def classifier_head_loss(class_output, labels):
    classification_loss = nn.CrossEntropyLoss()(class_output, labels)
    return classification_loss


class FilteredDataset(Dataset):
    def __init__(self, root_dir, transform, allowed_classes):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.allowed_indices = [
            i for i, (_, label) in enumerate(self.dataset.samples)
            if self.dataset.classes[label] in allowed_classes
        ]

    def __len__(self):
        return len(self.allowed_indices)

    def __getitem__(self, idx):
        return self.dataset[self.allowed_indices[idx]]


# divided by 255
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dataset_path = 'quality-control/linsen_data' 

train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoEncoder(input_channels=3, z_dim=128, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        _, _, _, class_output = model(data)
        loss = classifier_head_loss(class_output, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), 'vae_model.pth')


def test_vae_classification(model, test_dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    for data, labels in test_dataloader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            _, _, _, class_output = model(data) 
        _, predicted = torch.max(class_output, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    return accuracy, precision, recall, f1, y_true, y_pred



accuracy, precision, recall, f1, y_true, y_pred = test_vae_classification(model, test_dataloader, device)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
report = classification_report(y_true, y_pred, target_names=['Bad', 'Good'])
print("Classification Report:\n")
print(report)


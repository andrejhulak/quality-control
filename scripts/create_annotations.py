import os
import pandas as pd

# Define the root directory and subdirectories
root_dir = 'casting_data'  # Change this to the root folder path
train_dir = os.path.join(root_dir, 'train')
test_dir = os.path.join(root_dir, 'test')

# Define the labels
defective_label = 1
ok_label = 0

# List to hold all image paths and their labels
image_data = []

# Function to process images and append their path, labels, and set type to image_data
def process_folder(folder, label, set_type):
    for filename in os.listdir(folder):
        if filename.endswith('.jpeg'):
            # Full path to the image file
            full_path = os.path.join(folder, filename)
            # Append relative path, label, and set type (train/test) to the list
            relative_path = os.path.relpath(full_path, root_dir)
            image_data.append([relative_path, label, set_type])

# Process training images
process_folder(os.path.join(train_dir, 'def_front'), defective_label, 'train')
process_folder(os.path.join(train_dir, 'ok_front'), ok_label, 'train')

# Process test images
process_folder(os.path.join(test_dir, 'def_front'), defective_label, 'test')
process_folder(os.path.join(test_dir, 'ok_front'), ok_label, 'test')

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(image_data, columns=['file_path', 'label', 'set_type'])
df.to_csv(os.path.join(root_dir, 'annotations.csv'), index=False)

print('Annotations file created successfully!')

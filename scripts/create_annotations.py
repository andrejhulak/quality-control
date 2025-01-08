import os
import pandas as pd

root_dir = 'linsen_data'
train_dir = os.path.join(root_dir, 'train')
test_dir = os.path.join(root_dir, 'test')

defective_label = 1
ok_label = 0

image_data = []

def process_folder(folder, label, set_type):
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            full_path = os.path.join(folder, filename)
            relative_path = os.path.relpath(full_path, root_dir)
            image_data.append([relative_path, label, set_type])

process_folder(os.path.join(train_dir, 'not-good'), defective_label, 'train')
process_folder(os.path.join(train_dir, 'good'), ok_label, 'train')

process_folder(os.path.join(test_dir, 'not-good'), defective_label, 'test')
process_folder(os.path.join(test_dir, 'good'), ok_label, 'test')

df = pd.DataFrame(image_data, columns=['file_path', 'label', 'set_type'])
df.to_csv(os.path.join(root_dir, 'annotations.csv'), index=False)

print('Annotations file created successfully!')
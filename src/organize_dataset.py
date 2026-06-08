import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base, 'elpv-dataset', 'src', 'elpv_dataset', 'data')
labels_path = os.path.join(dataset_path, 'labels.csv')

# Read labels
df = pd.read_csv(labels_path, sep=r'\s+', header=None, names=['image', 'label', 'type'])
df['image'] = df['image'].str.strip()

# Assign classes
def assign_class(label):
    if label == 0.0:
        return 'clean'
    elif label == 1.0:
        return 'severe_defect'
    else:
        return 'mild_defect'

df['class'] = df['label'].apply(assign_class)

# Create folders
splits = ['train', 'val', 'test']
classes = ['clean', 'mild_defect', 'severe_defect']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(base, 'data', split, cls), exist_ok=True)

# Split and copy
for cls in classes:
    class_df = df[df['class'] == cls]
    train_df, temp_df = train_test_split(class_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for _, row in split_df.iterrows():
            src = os.path.join(dataset_path, row['image'])
            dst = os.path.join(base, 'data', split_name, cls, os.path.basename(row['image']))
            shutil.copy2(src, dst)

    print(f"{cls}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

print("\nDataset organized successfully!")
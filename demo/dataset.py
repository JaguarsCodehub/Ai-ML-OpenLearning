import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_extracted_data(source_dir, output_dir='dataset'):
    """
    Organize already extracted dataset into train/validation/test splits
    
    Args:
        source_dir: Path to the extracted dataset folder
        output_dir: Where to create the organized dataset
    """
    # Create necessary directories
    base_dir = Path(output_dir)
    for split in ['train', 'validation', 'test']:
        for category in ['healthy', 'diseased']:
            os.makedirs(base_dir / split / category, exist_ok=True)
    
    # Collect image paths
    source_path = Path(source_dir)
    healthy_images = []
    diseased_images = []
    
    # Walk through all directories
    print("Scanning directories...")
    for folder in source_path.glob('*'):
        if folder.is_dir():
            if 'healthy' in folder.name.lower():
                healthy_images.extend(list(folder.glob('*.jpg')))
            else:
                diseased_images.extend(list(folder.glob('*.jpg')))
    
    print(f"Found {len(healthy_images)} healthy images and {len(diseased_images)} diseased images")
    
    # Split datasets (70% train, 15% validation, 15% test)
    train_h, temp_h = train_test_split(healthy_images, test_size=0.3, random_state=42)
    valid_h, test_h = train_test_split(temp_h, test_size=0.5, random_state=42)
    
    train_d, temp_d = train_test_split(diseased_images, test_size=0.3, random_state=42)
    valid_d, test_d = train_test_split(temp_d, test_size=0.5, random_state=42)
    
    # Copy files to respective directories
    def copy_files(files, split_name, category):
        dest_dir = base_dir / split_name / category
        print(f"Copying {len(files)} files to {dest_dir}")
        for file in files:
            shutil.copy2(file, dest_dir)
    
    # Copy all files to their respective directories
    copy_files(train_h, 'train', 'healthy')
    copy_files(train_d, 'train', 'diseased')
    copy_files(valid_h, 'validation', 'healthy')
    copy_files(valid_d, 'validation', 'diseased')
    copy_files(test_h, 'test', 'healthy')
    copy_files(test_d, 'test', 'diseased')
    
    # Print statistics
    print("\nDataset Statistics:")
    for split in ['train', 'validation', 'test']:
        print(f"\n{split} set:")
        for category in ['healthy', 'diseased']:
            count = len(list((base_dir / split / category).glob('*.jpg')))
            print(f"  {category}: {count} images")

# Example usage
if __name__ == "__main__":
    organize_extracted_data('E:\jyotindra\Ai-ML-OpenLearning\demo\PlantVillage')
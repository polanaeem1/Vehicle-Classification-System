import os
import shutil
import random
import argparse
import sys
import numpy as np
import cv2
from collections import Counter
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import albumentations as A
except ImportError:
    print("Error: albumentations not found. Please run 'pip install albumentations'")
    sys.exit(1)

# --- Configuration ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
CLASSES = ['bus', 'compact', 'motorcycle', 'sedan', 'suv', 'truck', 'van']
PROJECT_NAME = 'car_classification_project'
MODEL_NAME = 'augmented_model'

def get_augmentation_pipeline():
    """
    Define the Albumentations pipeline for car classification.
    
    Transforms:
    - RandomBrightnessContrast: Simulates different lighting conditions.
    - MotionBlur: Simulates camera shake or moving vehicles.
    - GaussNoise: Simulates sensor noise/grain.
    - RandomShadow: Simulates shadows from trees/buildings (critical for outdoor cars).
    - Rotate: Small rotations to simulate slight camera tilt (avoid flipping cars upside down).
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomShadow(
            num_shadows_lower=1, 
            num_shadows_upper=3, 
            shadow_dimension=5, 
            shadow_roi=(0, 0.5, 1, 1), # Shadow mostly on bottom half (ground)
            p=0.4
        ),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        # Optional: CoarseDropout to simulate occlusions? Maybe too strong for classification.
    ])

def balance_and_augment_dataset(directory, target_count=None, dry_run=False):
    """
    Balances the dataset by oversampling minority classes using Albumentations.
    Instead of copying files, it generates NEW augmented versions.
    """
    print(f"\n--- Checking Class Balance in {directory} ---")
    
    counts = {}
    for cls in CLASSES:
        cls_path = os.path.join(directory, cls)
        os.makedirs(cls_path, exist_ok=True)
        counts[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Current distribution: {counts}")
    
    if sum(counts.values()) == 0:
        print("No data found! Skipping balancing.")
        return

    current_max = max(counts.values())
    if target_count is None:
        final_target = current_max
    else:
        final_target = max(current_max, target_count)
    
    if final_target == 0: return
    
    print(f"Target count per class: ~{final_target}")
    
    aug_pipeline = get_augmentation_pipeline()
    
    for cls in CLASSES:
        current_count = counts[cls]
        if current_count == 0:
             print(f"WARNING: Class {cls} has 0 images! Cannot augment.")
             continue
             
        if current_count < final_target:
            diff = final_target - current_count
            print(f"  Augmenting {cls}: Needs +{diff} images...")
            
            cls_path = os.path.join(directory, cls)
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Select random images to augment
            # We might need to loop if diff > len(files)
            images_to_augment = random.choices(files, k=diff)
            
            for i, fname in enumerate(images_to_augment):
                if dry_run:
                    continue
                    
                src_path = os.path.join(cls_path, fname)
                dst_filename = f"aug_{i}_{fname}"
                dst_path = os.path.join(cls_path, dst_filename)
                
                # Check if already exists to avoid re-work
                if os.path.exists(dst_path):
                    continue
                
                try:
                    # Load image with OpenCV (Albumentations expects numpy array)
                    image = cv2.imread(src_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply augmentation
                    augmented = aug_pipeline(image=image)['image']
                    
                    # Save back
                    augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dst_path, augmented_bgr)
                    
                except Exception as e:
                    print(f"    Error augmenting {fname}: {e}")
            
            if dry_run:
                print(f"    [DRY RUN] Would generate {diff} augmented images for {cls}")
            else:
                print(f"    Successfully added {diff} augmented images.")

def load_datasets_if_needed():
    """
    Checks if data exists. If not, downloads from HuggingFace/Sources using the logic from the original notebook.
    """
    if os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0:
        print("Data directory exists. Skipping download.")
        return

    print("Data directory not found or empty. Downloading datasets...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library needed. pip install datasets")
        sys.exit(1)

    # Setup directories
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            os.makedirs(os.path.join(split, cls), exist_ok=True)
            
    def get_split_dir():
        r = random.random()
        if r < 0.8: return TRAIN_DIR
        elif r < 0.9: return VAL_DIR
        else: return TEST_DIR

    # --- Helper Functions ---
    def map_stanford_label(label_str):
        name = label_str.lower()
        if 'sedan' in name: return 'sedan'
        if 'suv' in name: return 'suv'
        if 'truck' in name or 'pickup' in name: return 'truck'
        if 'van' in name or 'minivan' in name: return 'van'
        if 'hatchback' in name: return 'compact'
        if 'coupe' in name: return 'sedan'
        if 'wagon' in name: return 'sedan'
        return None

    def map_additional_label(label_str):
        name = label_str.lower()
        if 'bus' in name: return 'bus'
        if 'motorcycle' in name or 'bike' in name: return 'motorcycle'
        if 'truck' in name or 'pickup' in name: return 'truck'
        return None

    # 1. Stanford Cars
    try:
        print("Loading Stanford Cars...")
        ds_stanford = load_dataset("tanganke/stanford_cars", split="train+test")
        count = 0
        for i, item in enumerate(ds_stanford):
            label_idx = item['label']
            label_str = ds_stanford.features['label'].int2str(label_idx)
            target_cls = map_stanford_label(label_str)
            
            if target_cls:
                dest_dir = get_split_dir()
                save_path = os.path.join(dest_dir, target_cls, f"stanford_{i}.jpg")
                try:
                    item['image'].convert('RGB').save(save_path)
                    count += 1
                except:
                    pass
        print(f"Processed {count} images from Stanford Cars.")
    except Exception as e:
        print(f"Error loading Stanford Cars: {e}")

    # 2. Additional Data (DrBimmer/COCO)
    sources = [
        {"path": "DrBimmer/vehicle-classification", "split": "train"},
        {"path": "HuggingFaceM4/COCO", "split": "train"}
    ]
    
    try:
        from datasets import DownloadConfig
        # Reduce retries to fail faster if connection is bad
        dl_config = DownloadConfig(resume_download=True, max_retries=1)
    except ImportError:
        dl_config = None

    for source in sources:
        try:
            print(f"Trying {source['path']}...")
            # increased timeout via socket default if needed, but requests/aiohttp usually handle it.
            # dl_config helps limit retries.
            ds = load_dataset(source['path'], split=source['split'], streaming=True, download_config=dl_config)
            count = 0
            for i, item in enumerate(ds):
                if count >= 3000: break # Limit
                
                label_str = ""
                if 'label' in item:
                    label_str = ds.features['label'].int2str(item['label'])
                elif 'category' in item:
                    label_str = item['category']
                elif 'objects' in item:
                    if len(item['objects']['label']) > 0:
                        label_idx = item['objects']['label'][0]
                        label_str = ds.features['objects'].feature['label'].int2str(label_idx)
                
                target_cls = map_additional_label(label_str)
                if target_cls:
                    dest_dir = get_split_dir()
                    save_path = os.path.join(dest_dir, target_cls, f"{source['path'].split('/')[-1]}_{i}.jpg")
                    try:
                        img = item['image']
                        if not isinstance(img, Image.Image):
                            img = Image.open(img)
                        img.convert('RGB').save(save_path)
                        count += 1
                    except:
                        pass
            print(f"Added {count} images from {source['path']}.")
        except Exception as e:
            print(f"Skipping {source['path']}: {e}")

    # 3. Custom Local Datasets
    # Mapping: 
    # Datasets/Datasets/truck -> truck
    # Datasets/Datasets/car -> sedan (approx)
    # Car-Bike-Dataset/Bike -> motorcycle
    # Car-Bike-Dataset/Car -> sedan (approx)

    local_sources = [
        {"root": "Datasets/Datasets", "mapping": {"truck": "truck", "car": "sedan"}},
        {"root": "Car-Bike-Dataset", "mapping": {"Bike": "motorcycle", "Car": "sedan"}}
    ]

    print("Processing Local Datasets...")
    local_count = 0
    for source in local_sources:
        root_dir = source["root"]
        mapping = source["mapping"]
        
        if not os.path.exists(root_dir):
            print(f"Warning: {root_dir} not found. Skipping.")
            continue
            
        for src_label, target_cls in mapping.items():
            src_path = os.path.join(root_dir, src_label)
            if not os.path.exists(src_path):
                print(f"Warning: Subfolder {src_path} not found. Skipping.")
                continue
                
            print(f"Processing {src_path} -> {target_cls}...")
            # Handle potential subdirectories or flat files
            # The list_dir showed flat files in these dirs
            valid_files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            
            for fname in valid_files:
                fpath = os.path.join(src_path, fname)
                dest_dir = get_split_dir()
                # Create unique name to avoid collisions
                save_name = f"local_{src_label}_{fname}"
                save_path = os.path.join(dest_dir, target_cls, save_name)
                
                if os.path.exists(save_path): continue

                try:
                    # Convert to RGB and save
                    with Image.open(fpath) as img:
                        img.convert('RGB').save(save_path)
                    local_count += 1
                except Exception as e:
                    # print(f"Error processing {fname}: {e}")
                    pass 

    print(f"Processed {local_count} local images.")

def train_model(epochs=30):
    print("\n--- Starting YOLOv8 Classification Training (Augmented) ---")
    
    # Load foundational model
    model = YOLO('yolov8n-cls.pt')
    
    # Train
    # Note: YOLO's built-in 'augment=True' applies standard HSV/Flip/Scale.
    # Our pre-processing adds specialized noise/blur/weather effects that YOLO doesn't natively do as strongly.
    results = model.train(
        data=DATA_DIR, 
        epochs=epochs, 
        imgsz=224, 
        batch=32,
        augment=True, 
        project=PROJECT_NAME,
        name=MODEL_NAME
    )
    return model

def validate_model(model=None):
    print("\n--- Validating Model ---")
    
    # Load best.pt if available
    best_model_path = os.path.join(PROJECT_NAME, MODEL_NAME, "weights", "best.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best weights from {best_model_path} for validation...")
        model = YOLO(best_model_path)
    elif model is None:
        print("No model provided and best.pt not found. Cannot validate.")
        return

    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1}")
    
    # Confusion Matrix
    print("Generating Confusion Matrix on ALL test data...")
    y_true = []
    y_pred = []
    
    # Iterate through test set
    for cls in CLASSES:
        cls_path = os.path.join(TEST_DIR, cls)
        if not os.path.exists(cls_path): continue
        
        files = os.listdir(cls_path)
        # Testing on all available data in data/val
        for fname in files: 
            img_path = os.path.join(cls_path, fname)
            try:
                res = model(img_path, verbose=False)[0]
                pred_cls = res.names[res.probs.top1]
                y_true.append(cls)
                y_pred.append(pred_cls)
            except:
                pass
                
    if len(y_true) > 0:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    else:
        print("No test data found to evaluate.")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with Albumentations")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run data augmentation without saving/training")
    parser.add_argument("--skip-aug", action="store_true", help="Skip augmentation step (if already done)")
    parser.add_argument("--target-count", type=int, default=12500, help="Target number of images per class (default: 12500)")
    args = parser.parse_args()
    
    # 0. Check Data
    load_datasets_if_needed()
    
    # 1. Augment & Balance
    if not args.skip_aug:
        balance_and_augment_dataset(TRAIN_DIR, target_count=args.target_count, dry_run=args.dry_run)
    
    if args.dry_run:
        print("Dry run complete. Exiting.")
        return

    # 2. Train
    model = train_model(epochs=args.epochs)
    
    # 3. Validate
    validate_model(model)
    
    print(f"\nTraining Complete. Model saved to {PROJECT_NAME}/{MODEL_NAME}/weights/best.pt")

if __name__ == "__main__":
    main()

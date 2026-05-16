import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from ultralytics import YOLO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Global cache for dataloaders
_DATALOADERS_CACHE = None

def get_dataloaders(data_dir, imgsz, batch_size):
    global _DATALOADERS_CACHE
    if _DATALOADERS_CACHE is not None:
        return _DATALOADERS_CACHE

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=(x == 'train'), num_workers=4)
                  for x in ['train', 'val', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    _DATALOADERS_CACHE = (dataloaders, dataset_sizes, class_names)
    return _DATALOADERS_CACHE

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.savefig(f'cm_{model_name}.png')
    plt.close()

def build_pytorch_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_small(weights='DEFAULT')
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def train_model(model_name, data_dir, epochs, batch_size, imgsz, patience, device):
    print(f"\n{'='*50}\nTraining Model: {model_name}\n{'='*50}")
    start_time = time.time()
    
    if model_name == 'yolov8':
        # YOLOv8 Training
        model = YOLO('yolov8n-cls.pt')
        yolo_device = 0 if device.type != 'cpu' else 'cpu'
        
        # YOLOv8 auto-handles everything, including Early Stopping (patience)
        model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=4,
            patience=patience,
            optimizer='AdamW',
            lr0=0.001,
            project='benchmark_runs',
            name='yolov8_training',
            device=yolo_device
        )
        
        # Determine best weights
        best_yolo_path = os.path.join('benchmark_runs', 'yolov8_training', 'weights', 'best.pt')
        if os.path.exists(best_yolo_path):
            trained_model = YOLO(best_yolo_path)
            # Save a dedicated copy
            import shutil
            shutil.copy(best_yolo_path, 'best_yolov8.pt')
        else:
            trained_model = model
            
        train_time = time.time() - start_time
        return trained_model, train_time
        
    else:
        # PyTorch Models Training
        dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir, imgsz, batch_size)
        num_classes = len(class_names)
        
        model = build_pytorch_model(model_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            print()

        train_time = time.time() - start_time
        print(f'Training complete in {train_time // 60:.0f}m {train_time % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), f'best_{model_name}.pth')
        
        return model, train_time

def evaluate_model(model, model_name, data_dir, imgsz, batch_size, device):
    print(f"\nEvaluating {model_name} on test set...")
    
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir, imgsz, batch_size)
    test_loader = dataloaders['test']
    
    y_true = []
    y_pred = []
    
    top1_correct = 0
    top5_correct = 0
    total = 0

    if model_name == 'yolov8':
        # YOLOv8 Inference
        test_dir = os.path.join(data_dir, 'test')
        for class_idx, class_name in enumerate(class_names):
            cls_path = os.path.join(test_dir, class_name)
            if not os.path.exists(cls_path):
                continue
            
            for file in os.listdir(cls_path):
                img_path = os.path.join(cls_path, file)
                try:
                    results = model(img_path, verbose=False)
                    probs = results[0].probs
                    pred_class_idx = probs.top1
                    
                    # top5 comes as tensor or list depending on ultralytics version
                    top5_class_indices = probs.top5
                    if isinstance(top5_class_indices, torch.Tensor):
                        top5_class_indices = top5_class_indices.tolist()
                    
                    y_true.append(class_idx)
                    y_pred.append(pred_class_idx)
                    
                    if pred_class_idx == class_idx:
                        top1_correct += 1
                    if class_idx in top5_class_indices:
                        top5_correct += 1
                    total += 1
                except Exception as e:
                    pass
    else:
        # PyTorch Models Inference
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                
                _, preds = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                
                top1_correct += torch.sum(preds == labels.data).item()
                
                # Top-5 Accuracy Logic
                k = min(5, len(class_names))
                _, topk_preds = outputs.topk(k, 1, True, True)
                topk_preds = topk_preds.t()
                correct_topk = topk_preds.eq(labels.view(1, -1).expand_as(topk_preds))
                top5_correct += correct_topk.reshape(-1).float().sum(0).item()
                
                total += labels.size(0)

    top1_acc = top1_correct / total if total > 0 else 0
    top5_acc = top5_correct / total if total > 0 else 0
    
    print(f"\n{model_name} Test Top-1 Acc: {top1_acc:.4f}")
    print(f"{model_name} Test Top-5 Acc: {top5_acc:.4f}")
    
    # Generate classification report and Confusion Matrix
    if total > 0:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    else:
        print("No test data found. Cannot compute metrics.")
        
    return top1_acc, top5_acc

def main():
    # =============== Configuration ===============
    DATA_DIR = "data"          # Directory with train/val/test folders
    IMG_SIZE = 224             # Unified Image Size
    BATCH_SIZE = 128           # Batch Size
    EPOCHS = 30                # Max Epochs per Model
    PATIENCE = 10              # Early Stopping Patience
    # Models to benchmark
    MODELS_TO_BENCHMARK = ['resnet18', 'efficientnet_b0', 'mobilenet_v3', 'yolov8']
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # =============================================
    
    print("="*50)
    print("🚗 CAR CLASSIFICATION BENCHMARKING FRAMEWORK 🚗")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Early Stopping Patience: {PATIENCE}")
    
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] Dataset directory '{DATA_DIR}' not found. Please ensure train/val/test exist.")
        return
        
    results_records = []

    for model_name in MODELS_TO_BENCHMARK:
        trained_model, train_time = train_model(
            model_name=model_name,
            data_dir=DATA_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            imgsz=IMG_SIZE,
            patience=PATIENCE,
            device=DEVICE
        )
        
        top1, top5 = evaluate_model(
            model=trained_model,
            model_name=model_name,
            data_dir=DATA_DIR,
            imgsz=IMG_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
        
        results_records.append({
            'Model': model_name,
            'Top1': top1,
            'Top5': top5,
            'Training Time (s)': train_time
        })
        
        # Cleanup to save memory between models
        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary Table Output
    print(f"\n{'='*50}\nBENCHMARK RESULTS\n{'='*50}")
    df = pd.DataFrame(results_records)
    
    # Format the training time to mm:ss for display
    df['Training Time Formatted'] = df['Training Time (s)'].apply(lambda x: f"{int(x // 60)}m {int(x % 60)}s")
    df['Top1'] = df['Top1'].round(4)
    df['Top5'] = df['Top5'].round(4)
    
    display_df = df[['Model', 'Top1', 'Top5', 'Training Time Formatted']]
    print(display_df.to_string(index=False))
    
    csv_file = 'benchmark_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")
    
    print("\nBenchmark framework execution completed successfully! ✅")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

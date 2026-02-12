import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from segmentation_dataset import KaryotypeSegmentationDataset, collate_fn
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # Add more transforms (Flip, etc.) if needed via T.Compose or custom
    return T.Compose(transforms)

def validate_loss(model, loader, device):
    """
    Computes loss on validation set.
    NOTE: Mask R-CNN must be in loop/train mode to return loss dict, 
    but we use torch.no_grad to disable gradient tracking.
    """
    # model.train() is required for R-CNN to compute loss, even during validation
    # This might update BN stats unless frozen, but acceptable for monitoring
    model.train() 
    total_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            steps += 1
            
    return total_loss / steps if steps > 0 else 0.0

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # GPU Optimization
    # Enable cudnn benchmark for faster training on fixed input sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Dataset
    DATA_DIR = '/karyotyping/24_chromosomes_object'
    full_dataset = KaryotypeSegmentationDataset(DATA_DIR, get_transform(train=True))
    
    # 70/15/15 Split
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Configuration
    batch_size = 25 # Increased slightly for GPU (adjust if OOM)
    num_workers = 8 # Parallel data loading
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Model
    num_classes = 2 # 1 class (chromosome) + background
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    best_val_loss = float('inf')
    
    print("Starting training with Validation...")
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iter: {i}, Train Loss: {losses.item():.4f}")
        
        # Update LR
        lr_scheduler.step()
        
        # --- Validation Phase ---
        val_loss = validate_loss(model, val_loader, device)
        print(f"Epoch {epoch} Summary: Avg Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'mask_rcnn_karyotype_best.pth')
            print("  -> New Best Model Saved!")
            
    # Save Final
    torch.save(model.state_dict(), 'mask_rcnn_karyotype_final.pth')

    # --- Test Phase ---
    print("\nEvaluating on Test Set...")
    test_loss = validate_loss(model, test_loader, device)
    print(f"Test Set Loss: {test_loss:.4f}")
    print("Optimization Complete.")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from karyotype_classification import KaryotypeClassificationDataset, HybridKaryotypeClassifier, KaryotypeFeatureExtractor

def train_model():
    # Setup
    DATA_DIR = '/karyotyping/24_chromosomes_object'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Transforms
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_dataset = KaryotypeClassificationDataset(DATA_DIR, transform=data_transform)
    total_samples = len(full_dataset)
    print(f"Total Samples: {total_samples}")
    
    # 70/15/15 Split
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Dataloaders
    # Pin memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    
    # Model
    model = HybridKaryotypeClassifier(num_classes=24, num_features=5) 
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    num_epochs = 2
    best_acc = 0.0
    
    print("Starting classification training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, features, labels) in enumerate(train_loader):
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}")
        
        train_acc = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, features, labels in val_loader:
                images = images.to(device)
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(images, features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'karyotype_classifier_best.pth')
            print("  -> New Best Model Saved!")
            
    print("Training finished.")
    torch.save(model.state_dict(), 'karyotype_classifier_final.pth')

    # Test Phase
    print("\nEvaluating on Test Set...")
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, features, labels in test_loader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(images, features)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    print(f"Test Set Accuracy: {100 * correct_test / total_test:.2f}%")

if __name__ == "__main__":
    train_model()

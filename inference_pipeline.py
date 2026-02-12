import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from train_segmentation import get_model_instance_segmentation
from karyotype_classification import HybridKaryotypeClassifier, KaryotypeFeatureExtractor
from karyogram_module import process_chromosome, plot_karyogram # New Import
import argparse
from pathlib import Path

# Class mapping
CLASS_MAP = {
    0: 'A1', 1: 'A2', 2: 'A3',
    3: 'B4', 4: 'B5',
    5: 'C6', 6: 'C7', 7: 'C8', 8: 'C9', 9: 'C10', 10: 'C11', 11: 'C12',
    12: 'D13', 13: 'D14', 14: 'D15',
    15: 'E16', 16: 'E17', 17: 'E18',
    18: 'F19', 19: 'F20',
    20: 'G21', 21: 'G22',
    22: 'X', 23: 'Y'
}

def load_models(seg_path, cls_path, device):
    # 1. Segmentation Model
    print("Loading Segmentation Model...")
    seg_model = get_model_instance_segmentation(num_classes=2) # 1 class + bg
    seg_model.load_state_dict(torch.load(seg_path, map_location=device, weights_only=True))
    seg_model.to(device)
    seg_model.eval()
    
    # 2. Classification Model
    print("Loading Classification Model...")
    cls_model = HybridKaryotypeClassifier(num_classes=24, num_features=5)
    cls_model.load_state_dict(torch.load(cls_path, map_location=device, weights_only=True))
    cls_model.to(device)
    cls_model.eval()
    
    return seg_model, cls_model

def run_inference(image_path, seg_model, cls_model, device):
    # Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).to(device)
    
    # 1. Segmentation
    print("Running Segmentation...")
    with torch.no_grad():
        prediction = seg_model([img_tensor])
        
    # Filter predictions (confidence threshold)
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_masks = prediction[0]['masks'].cpu().numpy() # Extract masks (N, 1, H, W)
    
    threshold = 0.5
    keep_indices = pred_scores > threshold
    boxes = pred_boxes[keep_indices]
    scores = pred_scores[keep_indices]
    masks = pred_masks[keep_indices]
    
    print(f"Found {len(boxes)} chromosomes.")
    
    # Classification Setup
    feature_extractor = KaryotypeFeatureExtractor()
    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_h, img_w, _ = img_rgb.shape
    img_area = img_w * img_h
    
    results = [] # For visualization overlay
    karyogram_data = [] # For plotting karyogram
    
    # 2. Crop & Classify Loop
    print("Running Classification & Karyotyping...")
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Clamp coordinates
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(img_w, xmax); ymax = min(img_h, ymax)
        
        # Crop Image
        crop = img_rgb[ymin:ymax, xmin:xmax]
        if crop.size == 0: continue
        
        # Crop Mask
        # Mask is (1, H, W), taking [0] to get (H, W)
        mask_full = masks[i][0]
        # Crop mask to bbox
        mask_crop = mask_full[ymin:ymax, xmin:xmax]
        # Binarize (soft mask -> binary)
        mask_crop = (mask_crop > 0.5).astype(np.uint8)
        
        # Feature Extraction
        box_w = xmax - xmin
        box_h = ymax - ymin
        
        # Pass mask_crop to feature extractor
        features = feature_extractor.extract(crop, box_w, box_h, img_area, mask_crop)
        
        # Get CI from features (Indices: 0=AR, 1=RelArea, 2=MeanInt, 3=StdInt, 4=CI)
        ci_val = features[4].item()
        
        features = features.unsqueeze(0).to(device) # Batch dim
        
        # Transform for ResNet
        crop_tensor = cls_transform(crop).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = cls_model(crop_tensor, features)
            _, predicted_idx = torch.max(output, 1)
            class_idx = predicted_idx.item()
            class_name = CLASS_MAP[class_idx]
            
        # Store for Viz overlay
        results.append((box, class_name, mask_full))
        
        # Process for Karyogram (Rotate, etc)
        chrom_obj = process_chromosome(crop, mask_crop, class_idx, scores[i], ci=ci_val)
        karyogram_data.append(chrom_obj)
        
    return img_rgb, results, karyogram_data

def visualize(image, results, output_path):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()
    
    # Overlay masks
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    
    for box, class_name, mask_full in results:
        xmin, ymin, xmax, ymax = box
        
        # Draw Box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, class_name, color='red', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Accumulate Mask (Green channel)
        # mask_full is float 0..1, thresholded
        binary_mask = (mask_full > 0.5).astype(np.uint8)
        # Add to green channel of overlay
        mask_overlay[:, :, 1] = np.maximum(mask_overlay[:, :, 1], binary_mask * 100)

    # Blend original image with mask overlay
    # This is a bit rough, but works for viz
    # Alternatively, use contour drawing or alpha blending
    ax.imshow(mask_overlay, alpha=0.3)
        
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Annotated spread saved to {output_path}")

if __name__ == "__main__":
    import glob
    import os
    
    # Default paths
    SEG_MODEL = 'mask_rcnn_karyotype_best.pth'
    CLS_MODEL = 'karyotype_classifier_best.pth'
    
    # Pick a random test image
    TEST_IMGS = glob.glob('/karyotyping/24_chromosomes_object/JEPG/*.jpg')
    TEST_IMG = TEST_IMGS[0] if TEST_IMGS else None
    
    if not TEST_IMG:
        print("No images found to test.")
        exit()
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(SEG_MODEL) or not os.path.exists(CLS_MODEL):
        print("Models not found. Please train them first.")
    else:
        seg_model, cls_model = load_models(SEG_MODEL, CLS_MODEL, device)
        img, results, karyogram_data = run_inference(TEST_IMG, seg_model, cls_model, device)
        
        # 1. Save Annotated Spread
        visualize(img, results, "inference_result.png")
        
        # 2. Save Sorted Karyogram
        plot_karyogram(karyogram_data, "karyogram_result.png")

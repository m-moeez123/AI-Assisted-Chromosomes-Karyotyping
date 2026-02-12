import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import torch

def correct_orientation(image, mask):
    """
    Rotates the chromosome so its long axis is vertical.
    Uses PCA on the mask to find the principal axis.
    Returns rotated image, mask, and length (major axis).
    """
    # Find all points in the mask
    y, x = np.nonzero(mask)
    if len(y) == 0:
        return image, mask, 0 # No object
    
    # Form a data matrix
    pts = np.array([x, y]).T
    
    # PCA
    if len(pts) < 2: return image, mask, 0
    
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eig(cov)
    
    # Sort eigenvalues/vectors (largest first)
    sort_indices = np.argsort(evals)[::-1]
    evals = evals[sort_indices]
    evecs = evecs[:, sort_indices]
    
    # Calculate Length (approx 4 * sqrt(major_eigenvalue))
    # Eigenvalue represents variance. Length is roughly range of data along axis.
    length = 4 * np.sqrt(evals[0])
    
    # Principal axis angle
    angle = np.arctan2(evecs[1, 0], evecs[0, 0]) * 180 / np.pi
    
    # We want vertical alignment (90 degrees). 
    rotation_angle = 90 - angle 
    
    # Rotate image and mask
    image_rotated = rotate(image, -rotation_angle, reshape=True, cval=255) # White background
    mask_rotated = rotate(mask, -rotation_angle, reshape=True, order=0, cval=0) # Nearest neighbor for mask
    
    return image_rotated, mask_rotated, length

def process_chromosome(crop, mask, label, score, ci=0.5):
    """
    Wrapper to standardize one chromosome object.
    Includes Centromeric Index (CI) for advanced sorting.
    """
    # 1. Correct Orientation & Get Length
    img_rot, mask_rot, length = correct_orientation(crop, mask)
    
    # 2. Calculate Properties for Sorting
    area = np.sum(mask_rot > 0)
    
    return {
        'image': img_rot,
        'mask': mask_rot,
        'original_label': label, # Keep predicted label for reference
        'score': score,
        'area': area,
        'length': length,
        'ci': ci
    }

def heuristic_karyotype_assignment(chromosomes):
    """
    Sorts chromosomes using Length AND Centromeric Index (CI).
    Enforces Acrocentric geometry for Groups D (13-15) and G (21-22, Y).
    
    CI < 0.3 -> Acrocentric (p-arm is very small/absent)
    CI >= 0.3 -> Metacentric/Submetacentric
    """
    # Filter valid objects
    valid_chroms = [c for c in chromosomes if c['area'] > 100]
    
    # Split by Morphology (CI)
    # Note: KaryotypeFeatureExtractor._calculate_ci returns p/total.
    # Acrocentric (centromere at end) has p approx 0 -> CI approx 0.
    # Threshold 0.3 is reasonable.
    acrocentric = []
    others = []
    
    for c in valid_chroms:
        if c['ci'] < 0.3:
            acrocentric.append(c)
        else:
            others.append(c)
            
    # Sort both lists by Length (Descending)
    acrocentric.sort(key=lambda x: x['length'], reverse=True)
    others.sort(key=lambda x: x['length'], reverse=True)
    
    # Define Slots (Standard Human Karyotype)
    # Total 46 slots.
    # Group D (13,14,15): Slots 24, 25, 26, 27, 28, 29 (6 chromosomes) -> Indices 12, 12, 13, 13, 14, 14
    # Group G (21,22,Y): Slots 40-45 (5-6 chromosomes) -> Indices 20, 20, 21, 21, 23, (23)
    
    # We will try to fill the "Target Acrocentric Slots" with the `acrocentric` list first.
    # If not enough, borrow from `others` (misclassification).
    # If too many, push to `others` (misclassification).
    
    # Ideally, we reconstruct the full ordered list 1..46 by drawing from the sorted lists
    # based on the expected type of each position.
    
    final_order = [None] * 46
    
    # Indexes for input lists
    idx_acro = 0
    idx_other = 0
    
    # Expected types per chromosome pair (1-22, XY)
    # M=Meta/Submeta, A=Acrocentric
    # 1-12: M (Group A, B, C) -> 12 pairs = 24 slots
    # 13-15: A (Group D) -> 3 pairs = 6 slots
    # 16-20: M (Group E, F) -> 5 pairs = 10 slots
    # 21-22: A (Group G) -> 2 pairs = 4 slots
    # X: M (Group C size) -> 1 slot (or 2 if female)
    # Y: A (Group G size) -> 1 slot
    
    # We treat 1-22 pairs + X + Y.
    # Slots 0-23: M (Chr 1-12)
    # Slots 24-29: A (Chr 13-15)
    # Slots 30-39: M (Chr 16-20)
    # Slots 40-43: A (Chr 21-22)
    # Slot 44 (X): M
    # Slot 45 (Y): A
    
    # Let's simple-fill.
    
    sorted_slots = []
    
    # 1. Group A, B, C (Chr 1-12) - Expect M
    for _ in range(24): 
        if idx_other < len(others):
            sorted_slots.append(others[idx_other])
            idx_other += 1
        elif idx_acro < len(acrocentric):
            sorted_slots.append(acrocentric[idx_acro]) # Fallback
            idx_acro += 1
            
    # 2. Group D (Chr 13-15) - Expect A
    for _ in range(6):
        if idx_acro < len(acrocentric):
            sorted_slots.append(acrocentric[idx_acro])
            idx_acro += 1
        elif idx_other < len(others):
            sorted_slots.append(others[idx_other]) # Fallback
            idx_other += 1
            
    # 3. Group E, F (Chr 16-20) - Expect M
    for _ in range(10):
        if idx_other < len(others):
            sorted_slots.append(others[idx_other])
            idx_other += 1
        elif idx_acro < len(acrocentric):
            sorted_slots.append(acrocentric[idx_acro])
            idx_acro += 1
            
    # 4. Group G (Chr 21-22) - Expect A
    for _ in range(4):
        if idx_acro < len(acrocentric):
             sorted_slots.append(acrocentric[idx_acro])
             idx_acro += 1
        elif idx_other < len(others):
             sorted_slots.append(others[idx_other])
             idx_other += 1

    # 5. Sex Chromosomes (Assume XY for simplicity of logic)
    # X (M)
    if idx_other < len(others):
        sorted_slots.append(others[idx_other])
        idx_other += 1
    elif idx_acro < len(acrocentric):
        sorted_slots.append(acrocentric[idx_acro])
        idx_acro += 1
        
    # Y (A)
    if idx_acro < len(acrocentric):
        sorted_slots.append(acrocentric[idx_acro])
        idx_acro += 1
    elif idx_other < len(others):
        sorted_slots.append(others[idx_other])
        idx_other += 1
        
    # Add any remaining (if > 46 identified), usually debris
    while idx_other < len(others):
        sorted_slots.append(others[idx_other])
        idx_other += 1
    while idx_acro < len(acrocentric):
        sorted_slots.append(acrocentric[idx_acro])
        idx_acro += 1
    
    
    # Assign Labels based on Slot Index
    assigned_chroms = []
    
    # Standard Human Karyotype Map (Index -> Class)
    # 0-1 -> 0 (Chr1)
    # ...
    # 22-23 -> 11 (Chr12)
    # 24-25 -> 12 (Chr13)
    # ...
    
    for i in range(len(sorted_slots)):
        c = sorted_slots[i]
        
        # Determine Label from Rank i
        if i < 44:
            label = i // 2
        elif i == 44:
            label = 22 # X
        elif i == 45:
            label = 23 # Y
        else:
            label = 24 # Unknown/Extra
            
        c['label'] = label
        assigned_chroms.append(c)
        
    return assigned_chroms

def plot_karyogram(chromosomes, save_path="karyogram.png"):
    """
    Plots chromosomes in a standard Karyogram layout.
    Applies heuristic sorting to enforce structure.
    """
    
    # 1. Apply Heuristic Sorting & Assignment
    # This overwrites the 'label' with the sorted rank label
    sorted_chromosomes = heuristic_karyotype_assignment(chromosomes)
    
    # Group by Class
    karyotype = {i: [] for i in range(24)} # 0-23
    
    for chrom in sorted_chromosomes:
        lbl = chrom['label']
        if isinstance(lbl, torch.Tensor): lbl = lbl.item()
        if 0 <= lbl < 24:
            karyotype[lbl].append(chrom)
            
    # Sort within each class (Pairing)
    # Just to be sure, sort pair by length again
    for cls in karyotype:
        karyotype[cls].sort(key=lambda x: x['length'], reverse=True)

    # Plotting Setup
    cols = 8
    rows = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()
    
    # Styles
    fig.patch.set_facecolor('black')
    
    # Class Names Mapping (0->1 ... 21->22, 22->X, 23->Y)
    class_names = [str(i+1) for i in range(22)] + ['X', 'Y']
    
    for i in range(24): # 24 classes
        ax = axes[i]
        ax.set_facecolor('black')
        
        pair = karyotype[i]
        label_text = class_names[i]
        
        if not pair:
            ax.text(0.5, 0.5, f"{label_text}", 
                    ha='center', va='center', color='gray') # Removed (Missing) text to be cleaner
            ax.axis('off')
            continue
            
        # Create canvas for the pair
        h = max(c['image'].shape[0] for c in pair)
        total_w = sum(c['image'].shape[1] for c in pair) + (10 * (len(pair)-1))
        
        # White canvas
        canvas = np.ones((h, total_w, 3), dtype=np.uint8) * 255 
        
        current_x = 0
        for chrom in pair:
            img = chrom['image']
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
            ch, cw = img.shape[:2]
            y_offset = (h - ch) // 2
            
            canvas[y_offset:y_offset+ch, current_x:current_x+cw] = img
            current_x += cw + 10
            
        ax.imshow(canvas)
        ax.set_title(label_text, color='white', fontsize=12, pad=10)
        ax.axis('off')

    plt.suptitle("Automated AI Karyogram (Heuristic Sort)", fontsize=24, color='white', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='black', dpi=150)
    print(f"Karyogram saved to {save_path}")

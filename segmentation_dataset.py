import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

class KaryotypeSegmentationDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / 'annotations'
        self.images_dir = self.data_dir / 'JEPG'
        self.transforms = transforms
        
        self.xml_files = sorted(list(self.annotations_dir.glob('*.xml')))
        
    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_path = self.xml_files[idx]
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find('filename').text
        img_path = self.images_dir / filename
        
        # Load Image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocessing (CLAHE) - Essential for better Otsu Masks
        # We apply CLAHE here for mask generation, but return the original (or normalized) image
        # Actually, Mask R-CNN expects float tensors [0,1].
        
        # Parse Objects
        boxes = []
        labels = [] # We'll use 1 for chromosome, 0 is background
        
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Simple check for valid box
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1) # Class-agnostic: "Chromosome"

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Generate Masks on-the-fly (Pseudo-Masks using Otsu)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        masks = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            mask_full = np.zeros_like(gray, dtype=np.uint8)
            roi = gray[y1:y2, x1:x2]
            
            # Otsu
            if roi.size == 0:
                masks.append(mask_full)
                continue
                
            if np.mean(roi) > 127: # Light background assumption
                roi_proc = cv2.bitwise_not(roi)
            else:
                roi_proc = roi
            
            blur = cv2.GaussianBlur(roi_proc, (5, 5), 0)
            _, mask_roi = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            mask_full[y1:y2, x1:x2] = mask_roi
            masks.append(mask_full)
            
        masks = np.array(masks, dtype=np.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            # Note: Custom transforms handling would be needed for boxes/masks
            # For this MVP, we assume transforms is just ToTensor()
            image = self.transforms(image)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

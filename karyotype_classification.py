import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

class KaryotypeFeatureExtractor:
    """
    Extracts the 'Triad of Features' from chromosome crops/masks.
    """
    def extract(self, image, box_width, box_height, image_area, mask=None):
        # 1. Geometric: Aspect Ratio
        # (Handling division by zero)
        aspect_ratio = box_height / box_width if box_width > 0 else 0
        
        # 2. Scale: Relative Area
        box_area = box_width * box_height
        relative_area = box_area / image_area if image_area > 0 else 0
        
        # 3. Texture: Banding Density (Intensity Statistics)
        # Using the crop image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use mask to filter background if available
        if mask is not None:
            # mask is (H, W) uint8 0 or 1
            # Ensure resizing matches if mask comes from different source scale (unlikely here)
            if mask.shape != gray.shape:
                mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            pixels = gray[mask > 0]
            if pixels.size > 0:
                mean_intensity = np.mean(pixels) / 255.0
                std_intensity = np.std(pixels) / 255.0
            else:
                mean_intensity = 0.5
                std_intensity = 0.0
        else:
            mean_intensity = np.mean(gray) / 255.0
            std_intensity = np.std(gray) / 255.0
        
        # 4. Advanced Geometric: Centromere Index (CI)
        ci = self._calculate_ci(mask) if mask is not None else 0.5
        
        features = np.array([aspect_ratio, relative_area, mean_intensity, std_intensity, ci], dtype=np.float32)
        return torch.tensor(features)

    def _calculate_ci(self, mask):
        """
        Approximates Centromere Index (p / (p+q)) by finding the constriction.
        Assumes the chromosome is roughly vertical (aligned by PCA could be better).
        """
        if mask is None or np.sum(mask) == 0:
            return 0.5
            
        # 1. Calculate width profile along the major axis (assuming vertical for now)
        # Ideally we should rotate the mask to be vertical first.
        # Simple heuristic: if width > height, rotate 90
        coords = np.column_stack(np.where(mask > 0))
        # PCA for rotation
        if coords.shape[0] > 5: # Need points
            mean = np.mean(coords, axis=0)
            cov = np.cov(coords.T)
            start_vals, vectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(vectors[1, 1], vectors[1, 0]))
            
            # Rotate mask to vertical
            # (Skipping complex rotation for MVP speed, using bounding box aspect)
            pass

        h, w = mask.shape
        if w > h:
            # Rotate 90
            mask = np.rot90(mask)
            h, w = mask.shape
            
        # Profile: count pixels per row
        width_profile = np.sum(mask, axis=1)
        
        # Smooth profile
        from scipy.ndimage import gaussian_filter1d
        smooth_profile = gaussian_filter1d(width_profile, sigma=2)
        
        # Find local minima (constriction)
        # We process the central 80% to avoid tips
        start = int(0.1 * h)
        end = int(0.9 * h)
        if end <= start: return 0.5
        
        roi_profile = smooth_profile[start:end]
        if roi_profile.size == 0: return 0.5
        
        # Constriction is the minimum width in the central region
        min_idx = np.argmin(roi_profile)
        constriction_y = start + min_idx
        
        # p (short arm) and q (long arm)
        # p is the shorter length from constriction to end
        len1 = constriction_y
        len2 = h - constriction_y
        
        p = min(len1, len2)
        q = max(len1, len2)
        total = p + q
        
        ci = p / total if total > 0 else 0.5
        return ci

class KaryotypeClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / 'annotations'
        self.images_dir = self.data_dir / 'JEPG'
        self.transform = transform
        self.feature_extractor = KaryotypeFeatureExtractor()
        
        # Build index of valid chromosomes (skipping difficult ones if needed)
        self.samples = []
        self.class_to_idx = {
            'A1': 0, 'A2': 1, 'A3': 2,
            'B4': 3, 'B5': 4,
            'C6': 5, 'C7': 6, 'C8': 7, 'C9': 8, 'C10': 9, 'C11': 10, 'C12': 11,
            'D13': 12, 'D14': 13, 'D15': 14,
            'E16': 15, 'E17': 16, 'E18': 17,
            'F19': 18, 'F20': 19,
            'G21': 20, 'G22': 21,
            'X': 22, 'Y': 23
        }
        
        xml_files = sorted(list(self.annotations_dir.glob('*.xml')))
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename = root.find('filename').text
                
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                img_area = img_width * img_height
                
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in self.class_to_idx:
                        continue # Skip unknown classes like 'chromosomes' or typos
                        
                    bndbox = obj.find('bndbox')
                    xmin = max(0, int(bndbox.find('xmin').text))
                    ymin = max(0, int(bndbox.find('ymin').text))
                    xmax = min(img_width, int(bndbox.find('xmax').text))
                    ymax = min(img_height, int(bndbox.find('ymax').text))
                    
                    if xmax > xmin and ymax > ymin:
                        self.samples.append({
                            'filename': filename,
                            'bbox': (xmin, ymin, xmax, ymax),
                            'class_idx': self.class_to_idx[name],
                            'img_area': img_area
                        })
            except Exception as e:
                pass # Skip broken files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.images_dir / sample['filename']
        xmin, ymin, xmax, ymax = sample['bbox']
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop
        crop = image[ymin:ymax, xmin:xmax]
        
        # Generate Pseudo-mask for feature extraction training
        # (Since we don't have ground truth masks in XML, we approximate like in segmentation)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if np.mean(gray_crop) > 127: # Light bg
            gray_crop_proc = cv2.bitwise_not(gray_crop)
        else:
            gray_crop_proc = gray_crop
        blur = cv2.GaussianBlur(gray_crop_proc, (5, 5), 0)
        _, mask_crop = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract Manual Features
        box_w = xmax - xmin
        box_h = ymax - ymin
        features = self.feature_extractor.extract(crop, box_w, box_h, sample['img_area'], mask=mask_crop)
        
        # Transforms (Resize for ResNet)
        if self.transform:
            crop = self.transform(crop)
            
        return crop, features, sample['class_idx']

class HybridKaryotypeClassifier(nn.Module):
    def __init__(self, num_classes=24, num_features=5):
        super(HybridKaryotypeClassifier, self).__init__()
        
        # Backbone (ResNet18)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove head
        
        # Fusion Layer
        self.fusion = nn.Linear(num_ftrs + num_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x, features):
        x = self.backbone(x)
        # Concatenate
        combined = torch.cat((x, features), dim=1)
        
        out = self.fusion(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

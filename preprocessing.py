import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path

class KaryotypePreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / 'annotations'
        self.images_dir = self.data_dir / 'JEPG'
        self.output_dir = self.data_dir / 'processed'
        self.output_dir.mkdir(exist_ok=True)

    def load_sample(self, filename_stem):
        """Loads image and annotation for a given filename stem (no extension)."""
        xml_path = self.annotations_dir / f"{filename_stem}.xml"
        img_path = self.images_dir / f"{filename_stem}.jpg"

        if not xml_path.exists() or not img_path.exists():
            raise FileNotFoundError(f"Missing XML or JPG for {filename_stem}")

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            bbox = {
                'xmin': int(bndbox.find('xmin').text),
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text),
                'class': name
            }
            objects.append(bbox)

        # Load Image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, objects

    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        # CLAHE operates on the L channel of LAB color space or simpler grayscale
        # For RGB, we can convert to LAB, apply to L, and convert back
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        merged = cv2.merge((cl, a, b))
        final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return final

    def generate_pseudo_masks(self, image, objects):
        """
        Generates binary masks for each object using Otsu's thresholding 
        within the bounding box.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        instance_masks = []

        for obj in objects:
            xmin, ymin = max(0, obj['xmin']), max(0, obj['ymin'])
            xmax, ymax = min(width, obj['xmax']), min(height, obj['ymax'])
            
            # Extract ROI
            roi = gray[ymin:ymax, xmin:xmax]
            
            # Apply Otsu's thresholding
            # Inverting usually helps if background is light and chromosomes are dark
            # Check mean intensity to decide inversion
            if np.mean(roi) > 127: # Light background assumption
                roi = cv2.bitwise_not(roi)
                
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            _, mask_roi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Place back into full mask
            full_mask[ymin:ymax, xmin:xmax] = np.maximum(full_mask[ymin:ymax, xmin:xmax], mask_roi)
            
            instance_masks.append({
                'class': obj['class'],
                'mask': mask_roi,
                'bbox': [xmin, ymin, xmax, ymax]
            })

        return full_mask, instance_masks

    def visualize(self, original, processed, mask, filename_stem):
        """Visualizes changes."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(original)
        axes[0].set_title(f"Original: {filename_stem}")
        axes[0].axis('off')
        
        axes[1].imshow(processed)
        axes[1].set_title("Processed (CLAHE)")
        axes[1].axis('off')
        
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Pseudo-Mask (Otsu)")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{filename_stem}_preview.png"
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
        plt.close()

if __name__ == "__main__":
    # Test on a few samples
    DATASET_PATH = '/data/karyotyping/24_chromosomes_object'
    preprocessor = KaryotypePreprocessor(DATASET_PATH)
    
    # Get a few random files
    xml_files = glob.glob(os.path.join(DATASET_PATH, 'annotations', '*.xml'))
    sample_files = [Path(f).stem for f in xml_files[:3]]
    
    print(f"Processing samples: {sample_files}")
    
    for sample in sample_files:
        try:
            print(f"Processing {sample}...")
            img, objs = preprocessor.load_sample(sample)
            
            # 1. CLAHE
            img_clahe = preprocessor.apply_clahe(img)
            
            # 2. Masks (using original or CLAHE? Usually CLAHE helps thresholding)
            mask, _ = preprocessor.generate_pseudo_masks(img_clahe, objs)
            
            # 3. Visualize
            preprocessor.visualize(img, img_clahe, mask, sample)
            
        except Exception as e:
            print(f"Failed to process {sample}: {e}")

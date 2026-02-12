import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd

# Configuration
DATA_DIR = '/karyotyping/24_chromosomes_object'
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')
IMAGES_DIR = os.path.join(DATA_DIR, 'JEPG')

def perform_eda():
    print("Starting EDA...", flush=True)
    data = []
    xml_files = glob.glob(os.path.join(ANNOTATIONS_DIR, '*.xml'))
    
    # Limit to 500 files for speed if needed, but 5000 is manageable
    # xml_files = xml_files[:500] 

    print(f"Parsing {len(xml_files)} annotation files...", flush=True)

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = root.find('filename').text
            
            # Image dimensions
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image_area = width * height
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                box_width = xmax - xmin
                box_height = ymax - ymin
                box_area = box_width * box_height
                aspect_ratio = box_height / box_width if box_width > 0 else 0
                
                # Relative Size (Feature 2: Scale)
                relative_area = box_area / image_area if image_area > 0 else 0
                
                data.append({
                    'filename': filename,
                    'class': name,
                    'width': box_width, 'height': box_height,
                    'aspect_ratio': aspect_ratio,
                    'relative_area': relative_area
                })
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")

    df = pd.DataFrame(data)
    
    results = []
    results.append(f"Total Objects Extracted: {len(df)}")
    results.append("-" * 30)
    
    # 1. Class Distribution
    class_counts = df['class'].value_counts()
    results.append("\nClass Distribution (Top 10):")
    results.append(class_counts.head(10).to_string())
    results.append(f"\nTotal Classes: {len(class_counts)}")
    results.append(f"Min Count: {class_counts.min()} ({class_counts.idxmin()})")
    results.append(f"Max Count: {class_counts.max()} ({class_counts.idxmax()})")

    # 2. Geometric Analysis (Aspect Ratio) by Class
    results.append("\nAverage Aspect Ratio by Class (Top 5 Highest - Likely Acrocentric/Long):")
    ar_by_class = df.groupby('class')['aspect_ratio'].mean().sort_values(ascending=False)
    results.append(ar_by_class.head(5).to_string())
    
    results.append("\nAverage Aspect Ratio by Class (Top 5 Lowest - Likely Metacentric/Round):")
    results.append(ar_by_class.tail(5).to_string())

    # 3. Relative Size Analysis
    results.append("\nAverage Relative Area by Class (Largest 5):")
    size_by_class = df.groupby('class')['relative_area'].mean().sort_values(ascending=False)
    results.append(size_by_class.head(5).to_string())
    
    results.append("\nAverage Relative Area by Class (Smallest 5):")
    results.append(size_by_class.tail(5).to_string())

    # Output to file
    with open('eda_summary.txt', 'w') as f:
        f.write('\n'.join(results))
    
    print("\nEDA Completed. Summary saved to eda_summary.txt")
    print('\n'.join(results))

if __name__ == "__main__":
    perform_eda()

#!/bin/bash

# Ensure the library path is set

echo "=================================================="
echo "Starting Segmentation Training (Mask R-CNN)"
echo "Logs: segmentation.log"
echo "=================================================="
python3 train_segmentation.py > segmentation.log 2>&1

if [ $? -eq 0 ]; then
    echo "Segmentation Training Compliance. Starting Classification Training..."
else
    echo "Segmentation Training Failed! Check segmentation.log"
    exit 1
fi

echo "=================================================="
echo "Starting Classification Training (Hybrid ResNet)"
echo "Logs: classification.log"
echo "=================================================="
python3 train_classification.py > classification.log 2>&1

echo "=================================================="
echo "All Training Pipelines Completed."
echo "=================================================="

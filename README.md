# EasyOCR Integrated Project

This project contains scripts and a model for OCR (Optical Character Recognition) using EasyOCR integration.

## Training Results & Performance
The model achieved its optimal performance at Epoch 57 with the following metrics:
Validation Accuracy (Sequence): 81.29% (Full string recognition accuracy)
Character Accuracy: 93.05% (Individual character recognition accuracy)
Final Training Loss: 35.5929
Learning Rate: 0.000013

## Dataset
From https://www.kaggle.com/datasets/topkek69/vietnamese-license-plate-ocr.
A great dataset with about 12.000 files, divided in cropped and generated images.

## Files
- `train_easyocr_integrated.py`: Script for training the OCR model.
- `test_easyocr_integrated.py`: Script for testing the OCR model.
- `best_easyocr_integrated.pt`: Trained model weights.

## Usage
# Training
python train_easyocr_integrated.py --data ./dataset --epochs 50

# Testing
python test_easyocr_integrated.py --weights best_easyocr_integrated.pt --source ./test_images

## Requirements
- Python 3.x
- EasyOCR
- torch

## Installation
```bash
pip install easyocr torch torchvision opencv-python

## License
MIT License

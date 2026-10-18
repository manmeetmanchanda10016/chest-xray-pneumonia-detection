# Chest X-ray Pneumonia Detection

A deep learning project that classifies chest X-ray images as NORMAL or PNEUMONIA using ResNet18, PyTorch, and Streamlit.

## Features
- Transfer learning with ResNet18
- Chest X-ray image classification
- Interactive Streamlit dashboard
- Class probability scores

## Tech Stack
Python, PyTorch, Torchvision, Streamlit, Pillow

## Run Locally
Install the required packages:

`pip install -r requirements.txt`

Run the app:

`streamlit run app.py`

Place `chest_xray_resnet18.pth` in the same folder as `app.py`.

## Disclaimer
This is an educational project, not a clinically validated diagnostic tool. Do not use its predictions for medical decisions.

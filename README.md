# kerasCV_tflite

## Overview

This script demonstrates how to load a RetinaNet model, prepare it for inference, and convert it into TensorFlow Lite format for deployment on mobile or embedded devices. It involves setting up the environment, defining the model, generating predictions, and converting the model to TensorFlow Lite format.

## Code Breakdown
Environment Setup
``` python
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"
```

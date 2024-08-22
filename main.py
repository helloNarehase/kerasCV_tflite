import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_cv
import keras
import tensorflow as tf
import numpy as np

BATCH_SIZE = 12
model = keras_cv.models.RetinaNet(
    num_classes=2,
    bounding_box_format="xyxy",
    backbone=keras_cv.models.MobileNetV3LargeBackbone.from_preset(
        "mobilenet_v3_large_imagenet"
    )
)

model.load_weights("retinaNet2.keras")

@tf.function
def generate(image):
    pre = model(image)
    out = model.decode_predictions(pre, image)

    return out["boxes"], out["confidence"], out["classes"], out['num_detections']


concrete_func = generate.get_concrete_function(tf.TensorSpec(shape=[1, 640, 640, 3], dtype=tf.float32))


model.jit_compile = False
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# 모델 변환
tflite_model = converter.convert()
with open("retinaNet_N.tflite", "wb") as f:
    f.write(tflite_model)

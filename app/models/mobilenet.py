from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet')
model.save("mobilenet_v2_full.keras")
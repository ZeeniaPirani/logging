from PIL import Image
import pytesseract
import face_recognition
import cv2
import torch
import requests
from transformers import CLIPProcessor, CLIPModel


log_file = "logging.txt"

# with open(log_file, 'a') as f:
#     f.write(pytesseract.image_to_string(Image.open()))


#     image = face_recognition.load_image_file()
#     face_number = len(face_recognition.face_locations(image))
#     f.write(str(face_number))

#     net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
#     image = cv2.imread()

#     blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
#     net.setInput(blob)
#     detections = net.forward()

#     CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
#             "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
#             "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#     seen = set()
#     for i in range(detections.shape[2]):
#         if detections[0, 0, i, 2] > 0.2:
#             idx = int(detections[0, 0, i, 1])
#             seen.add(CLASSES[idx])

#     f.write(list(seen))


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("lol.jpg") 
labels = ["video game", "website", "movie", "desktop"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)  # shape: [1, 4]

for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
print("Prediction:", labels[probs.argmax()])

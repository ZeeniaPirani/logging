from PIL import Image
import easyocr
from face_recognition import load_image_file, face_locations
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel

# Finds all text written in image and logs it
def text_from_image(file, image_path):
    result = reader.readtext(image_path, detail=0)
    file.write(f"Text from image:\n{result}\n")

# Logs total number of faces in image
def face_recognition(file, image_path):
    # Runs face detection model, returns an array of all faces
    image = load_image_file(image_path)
    # Logs length of array (total faces in image)
    face_number = len(face_locations(image))
    file.write(f"\nTotal amount of faces:\n{face_number}\n")


def general_object_detection(file, image_path):
    # Loads pre-trained model and reads image
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
    image = cv2.imread(image_path)

    # Changes image to proper format
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # All classes that model can detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Adds object in set if model confidence score > 0.5
    seen = set()
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.5:
            idx = int(detections[0, 0, i, 1])
            seen.add(CLASSES[idx])

    # Logs list of seen objects
    file.write(f"\nObjects Detected:\n{list(seen)}\n\n")


def directed_object_detection(file, image_path):
    # Uses pretrained CLIP model to process image
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path) 
    # Possible labels for each image, model returns a confidence score for each label
    labels = ["video game","website", "movie", "desktop"]

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)  # shape: [1, 4]

    # Logs all confidence scores and model output
    for label, prob in zip(labels, probs[0]):
        file.write(f"{label}: {prob.item():.4f} ")
    file.write(f"\nFinal Prediction: {labels[probs.argmax()]}")


# Replace with logging file path and image path
log_file = "logging.txt"
path = ""

with open(log_file, 'a') as f:
    reader = easyocr.Reader(['en'])
    text_from_image(f, path)

    face_recognition(f, path)

    general_object_detection(f, path)

    directed_object_detection(f, path)
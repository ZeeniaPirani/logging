from PIL import Image
import easyocr
from face_recognition import load_image_file, face_locations
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import time
import json
import numpy as np
import argparse

# Finds all text written in image and logs it
def text_from_image(log_file, image_path):
    result = reader.readtext(image_path, detail=0)
    log_file.write(f"Text from image:\n{result}\n")

    # Creates new JSON object with value of image text, indents for easy readability
    data = {"image_text": result}
    text_json = json.dumps(data, indent=4)
    return text_json

# Logs total number of faces in image
# def face_recognition(file, image_path):
#     # Runs face detection model, returns an array of all faces
#     image = load_image_file(image_path)
#     # Logs length of array (total faces in image)
#     face_number = len(face_locations(image))
#     file.write(f"\nTotal amount of faces:\n{face_number}\n")

#     # Creates new JSON object with amount of faces in image
#     data = {"amount_faces": face_number}
#     face_json = json.dumps(data, indent=4)
#     return face_json


def general_object_detection(log_file, image_path):
    # Try-except to avoid errors with MobileNetSSD files
    try:
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
        log_file.write(f"\nObjects Detected:\n{list(seen)}\n\n")

        # JSON object with list of all objects seen
        data = {"general_objects_detected": list(seen)}
        general_od_json = json.dumps(data, indent=4)
        return general_od_json

    # Prints error message if issue with MobileNetSSD files
    except cv2.error:
        print("Error with MobileNetSSD files - download from https://github.com/ZeeniaPirani/logging and move to project file folder")


def directed_object_detection(log_file, image_path):
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
        log_file.write(f"{label}: {prob.item():.4f} ")
    log_file.write(f"\nFinal Prediction: {labels[probs.argmax()]}\n\n")

    # Returns JSON of most likely label
    data = {"direct_object_final": labels[probs.argmax()]}
    directed_od_json = json.dumps(data, indent=4)
    return directed_od_json


def color_histogram(log_file, image_path):
    image = cv2.imread(image_path)
    blue_color = cv2.calcHist([image], [0], None, [128], [0, 256]).flatten().tolist()
    green_color = cv2.calcHist([image], [1], None, [128], [0, 256]).flatten().tolist()
    red_color = cv2.calcHist([image], [2], None, [128], [0, 256]).flatten().tolist()

    histogram = {
        "blue":blue_color,
        "green":green_color,
        "red":red_color
    }

    color_histogram_json = json.dumps(histogram, indent=4)
    log_file.write(color_histogram_json)

    return color_histogram_json


def brightness_histogram(log_file, image_path):
    image = cv2.imread(image_path)
    image = image.astype(np.float32)

    B, G, R = cv2.split(image)

    luminance = 0.299*R + 0.587*G + 0.114*B
    hist = cv2.calcHist([luminance.astype(np.uint8)], [0], None, [128], [0, 256])

    data = {"luminance_histogram" : hist.flatten().tolist()}

    luminance_histogram_json = json.dumps(data, indent=4)
    log_file.write(luminance_histogram_json)

    return luminance_histogram_json


def calculate_pixel_difference(log_file, screenshot_1, screenshot_2):
    img1 = cv2.imread(screenshot_1)
    img2 = cv2.imread(screenshot_2)

    if img1.shape != img2.shape:
        print("Screenshots are different sizes")
        return
    
    difference = cv2.absdiff(img1, img2)

    if len(difference.shape) == 3:
        diff_mask = np.any(difference > 0, axis=2)
        num_different_pixels = np.sum(diff_mask)
    else:
        num_different_pixels = cv2.countNonZero(difference)
    
    data = {"pixel_difference":int(num_different_pixels)}
    pixel_difference_json = json.dumps(data, indent=4)
    log_file.write(pixel_difference_json)

    return pixel_difference_json

def calculate_rgb_difference(log_file, screenshot_1, screenshot_2):
    img1 = Image.open(screenshot_1).convert('RGB')
    img2 = Image.open(screenshot_2).convert('RGB')

    if img1.size != img2.size:
        print("Screenshots are different sizes")
        return

    arr1 = np.array(img1, dtype=np.int16)
    arr2 = np.array(img2, dtype=np.int16)

    diff_array = np.abs(arr1-arr2)
    total_rgb_difference = int(np.sum(diff_array))
   
    data = {"rgb_difference":total_rgb_difference}
    rgb_difference_json = json.dumps(data, indent=4)
    log_file.write(rgb_difference_json)

    return rgb_difference_json

parser = argparse.ArgumentParser(description="Analyze and compute the difference between two screenshots.")
parser.add_argument("image1", type=str, help="Path to first screenshot")
parser.add_argument("image2", type=str, help="Path to second screenshot")
parser.add_argument("output_file", type=str, help="Path to output file")

args = parser.parse_args()

imagePath1 = args.image1
imagePath2 = args.image2
output = args.output_file

with open(output, 'w') as logging_file:
    # Uses try-except when working with user input - catches errors if input isn't in project directory
    try:        
        reader = easyocr.Reader(['en'])
        
        # Creates variable to hold start time, new dictionary for final JSON object
        start_time = time.time()
        all_data = {}

        # Outputs current step in console, creates new key-value pair with detected text
        print("Starting text extraction...")
        all_data["text"] = text_from_image(logging_file, imagePath1)

        # Outputs time taken for text extraction to console
        text_extraction_time = time.time() 
        print(f"Text extraction time: {(text_extraction_time - start_time):.5f}\n")


        # Outputs current step, creates new key-value pair for amount of detected faces
        # print("Starting face recognition...")
        # all_data["face"] = face_recognition(f, path)

        # Prints total amount of time taken for face recognition
        face_recognition_time = time.time()
        print(f"Face recognition time: {(face_recognition_time - text_extraction_time):.5f}\n")


        # Prints current process, new pair in JSON with list of all objects detected
        print("Starting general object detection...")
        all_data["general"] = general_object_detection(logging_file, imagePath1)

        # Outputs time taken for general object detection
        general_od_time = time.time()
        print(f"General Object Detection Time: {(general_od_time - face_recognition_time):.5f}\n")


        # Creates new key-value pair for directed object detection
        print("Starting directed object detection...")
        all_data["directed"] = directed_object_detection(logging_file, imagePath1)

        # Outputs time taken
        directed_od_time = time.time()
        print(f"Directed Object Detection Time: {(directed_od_time - general_od_time):.5f}\n")


        # Creates new key-value pair for color histogram values
        print("Creating color histogram")
        all_data["color_histogram"] = color_histogram(logging_file, imagePath1)

        # Outputs time taken (not sure if needed)
        color_histogram_time = time.time()
        print(f"Color Histogram Time: {(color_histogram_time - directed_od_time):.5f}\n")
        

        # Creates new JSON entry for brightness histogram
        print("Creating luminance histogram")
        all_data["brightness_histogram"] = brightness_histogram(logging_file, imagePath1)
        
        brightness_histogram_time = time.time()
        print(f"Brightness Histogram Time: {(brightness_histogram_time - color_histogram_time):.5f}\n")


        print("Computing difference between two screenshots")
        all_data["pixel_difference"] = calculate_pixel_difference(logging_file, imagePath1, imagePath2)

        pixel_difference_time = time.time()
        print(f"Computing Difference Time: {(pixel_difference_time - brightness_histogram_time):.5f}\n")


        print("Computing RGB difference between two screenshots")
        all_data["rgb_difference"] = calculate_rgb_difference(logging_file, imagePath1, imagePath2)

        rgb_difference_time = time.time()
        print(f"RGB Difference Time: {(rgb_difference_time - pixel_difference_time):.5f}")

        # Final JSON object with all data
        complete_json = json.dumps(all_data, indent=4)

        # Prints total runtime for all four functions
        print(f"Total Runtime: {(pixel_difference_time - start_time):0.5f}\n")

    # If screenshot file not found, prints error message
    except FileNotFoundError:
        print("Image file not found - move file to project folder, enter the file extension as input as well")
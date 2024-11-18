

import cv2
import torch
import numpy as np
from sort.sort import *
import string
from tqdm import tqdm
import tensorflow as tf
import time
import cv2
import numpy as np
import joblib
import cv2
from PIL import Image
import numpy as np
# Assuming YOLOv8 can be loaded similarly to YOLOv5 models
from ultralytics import YOLO
import easyocr

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('/home/boss/Downloads/project/projectplate/files/license_plate_detector.pt')
mot_tracker = Sort()
plate = []
label_encoder = joblib.load('/home/boss/Downloads/project/projectplate/files/lable-en.joblib')
modelnlg = tf.keras.models.load_model('/home/boss/Downloads/project/projectplate/files/my_model.keras')
vehicles = [2, 3, 5, 7]

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
def parse_bbox(bbox_str):
    return [float(coord) for coord in bbox_str.strip('[]').split()]

# Function to crop and correct the license plate region from a frame
def crop_and_correct_license_plate(frame, bbox):
    """
    Crops and corrects the perspective of the license plate in the given frame.

    Parameters:
    frame (numpy.ndarray): The image frame containing the license plate.
    bbox (tuple): A tuple of four integers representing the bounding box (x1, y1, x2, y2).

    Returns:
    numpy.ndarray: The cropped and perspective-corrected license plate image.
    """
    try:
        # Extracting bounding box coordinates and validating them
        x1, y1, x2, y2 = map(int, bbox)
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x1 >= x2 or y1 >= y2:
            raise ValueError("Bounding box coordinates are out of frame boundaries or invalid.")
        
        width = x2 - x1
        height = y2 - y1

        # Defining the points for the bounding box
        pts1 = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        # Defining the points for the transformed bounding box
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Apply the perspective transformation to the license plate
        license_plate_crop = cv2.warpPerspective(frame, matrix, (width, height))
        current_time = int(time.time())
        car_filename = f"plate_{current_time}.jpg"
        cv2.imwrite(f"/home/boss/Downloads/project/projectplate/uploads/{car_filename}", license_plate_crop)
        

        return car_filename
    except Exception as e:

      print(f"Error: {e}")
      return None
    
def process_video(frame):
    frame = cv2.imread(frame)

    if frame is not None:
        
        
        

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
              x1, y1, x2, y2, score, class_id = detection
              if int(class_id) in vehicles:
                  detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        detections_.clear()

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            
            bbox = [x1, y1, x2, y2]
            license_plate_crop = crop_and_correct_license_plate(frame, bbox)
            return license_plate_crop
                
                
                
                
                

def format_plate(plate):
    letters = []
    numbers = []
    # Separate the letters and numbers
    for char in plate:
        if char.isdigit():
            numbers.append(char)
        else:
            letters.append(char)

    # Join letters and numbers in the desired format
    formatted_plate = " ".join(letters[::-1]) + " "+ "".join(numbers)
    return formatted_plate
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    

    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
        

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Resize image
    image = cv2.resize(image, (128, 128))
    
    # Ensure the correct shape
    image = cv2.medianBlur(image, 5)
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.dilate(blur, rect_kern, iterations=1)
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel_sharpening)
    # Normalize image
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    return image

def predict_image(image_path, model, label_encoder):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Predict the class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]
def newOldSegmentation(filename):
    
    # Read and preprocess the image
    gray = cv2.imread(filename, 0)
    gray = cv2.resize(gray, (200, 100))
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    # Find contours
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area
    Contourss = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    im2 = gray.copy()
    plate_num = ""
    imageResult = []
    listOfContours = []

    # Extract characters
    for cnt in Contourss:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = im2.shape
        if float(h) < 50:
            continue
        ratio = h / float(w)
        area = h * w
        if float(w) > 90:
            continue
        if area < 1000:
            continue
        if ratio > 7.5:
            continue
        if x < 12:
            continue
        if x > 565:
            continue
        if y < 75:
            continue
        listOfContours.append(x)

        # Determine if it's likely a number or a character
        if 2.4 <= ratio <= 5.3:  # assuming a rough heuristic that taller shapes are characters
                # Extract character
            char = gray[max(0, y - 50):min(gray.shape[0], y + h + 29), max(0, x - 5):min(gray.shape[1], x + w + 5)]
        elif (1.9 <= ratio >= 1.80):
            char = gray[max(0, y - 12):min(height, y + h + 40), max(0, x - 5):min(width, x + w + 5)]    
        elif (ratio < 1.0) :  # numbers are less tall
            char = gray[max(0, y - 20):min(height, y + h + 10), max(0, x - 5):min(width, x + w + 5)]
        else :
            char = gray[y:y+h, x:x+w]
        
        try:
            char = cv2.resize(char, (24, 44))
        except Exception:
            continue

        # Improved rectangle drawing around the detected character
        cv2.rectangle(im2, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)

        # Create a blank image and place the character in it
        charCopy = np.zeros((44, 24), dtype=np.uint8)
        charCopy[:,:] = char


        imageResult.append(charCopy)

    # Sort extracted characters
    indices = sorted(range(len(listOfContours)), key=lambda k: listOfContours[k])
    imageResultCopy = []
    for index in indices:
        if index < len(imageResult):
            imageResultCopy.append(imageResult[index])

    imageResult = np.array(imageResultCopy)
    imageResultCopy.clear()
    return imageResult

def  gettext(filename):
    
    plate.clear()
    segmented_chars = newOldSegmentation(filename)
    for idx, char_img in enumerate(segmented_chars):
        current_time = int(time.time())
        filename = f"priedct/char_{idx}{current_time}.png"
        cv2.imwrite(filename, char_img)
        predicted_label = predict_image(filename, modelnlg, label_encoder)
        plate.append(predicted_label)
    formatted_plate = format_plate(plate)
    
    return formatted_plate

def read_license_plate(img,lanagye):
    reader = easyocr.Reader([lanagye], gpu=False)
    img = cv2.imread(img)
    """Extracts text from a license plate image using EasyOCR."""
    # Apply pre-processing to improve image quality
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_thresh = cv2.medianBlur(img_thresh, 5)
    img_processed = cv2.bitwise_not(img_thresh)
    
    results = reader.readtext(img_processed)
    if results:
        best_result = max(results, key=lambda result: result[2])  # Find the result with the highest confidence
        text = best_result[1].replace(" ", "").lower() 
        return text  # Return text and confidence score
    return None, None  # Ensure always returning a tuple

from flask import Flask, render_template, request, jsonify
import os
import cv2
from flask import send_from_directory
import time
import torch
import json
import shutil

import model as task
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DATA_FILE = 'data/detections.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model


# Ensure upload and data directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

def load_detections():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    return []

def save_detections(detections):
    with open(DATA_FILE, 'w') as file:
        json.dump(detections, file, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    current_time = int(time.time())

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Detect license plate using YOLO
        
        car_filename = f"car_{current_time}.jpg"
        file_psath = os.path.join(UPLOAD_FOLDER, car_filename)
        
        shutil.copy(filepath, file_psath)
        platename = task.process_video(file_psath)
        if platename is  None:
            return jsonify({'error': 'No license plate detected'})
        return jsonify({'car_image': car_filename, 'plate_image': platename})
        
       
        
        # Assume the first detection is the license plate
        
        
        
        
    
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/recognize', methods=['POST'])
def recognize_plate():
    plate_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plate.jpg')
    
    if os.path.exists(plate_path):
        plate_img = cv2.imread(plate_path)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        
        car_image  = request.json.get('car_image')
        plate_image  = request.json.get('plate_image')
        platpripath = f"/home/boss/Downloads/project/projectplate/{plate_image}"
        print(platpripath)
        platenumber = ""
        if request.json.get('model')== "cnn": 
            platenumber =task.gettext(platpripath)
        elif request.json.get('model')== "easyocr":
            
            platenumber =task.read_license_plate(platpripath,request.json.get('language'))
        
        
        # Save the detection to JSON file
        detection = {
            'car_image': car_image,
            'plate_image': plate_image,
            'plate_number': platenumber
        }
        detections = load_detections()
        detections.append(detection)
        save_detections(detections)
        
        return jsonify({'text': platenumber})
    
    return jsonify({'error': 'Plate image not found'})

@app.route('/api/detections', methods=['GET'])
def get_detections():
    detections = load_detections()
    return jsonify(detections)

@app.route('/api/detections/<int:index>', methods=['PUT'])
def edit_detection(index):
    detections = load_detections()
    if 0 <= index < len(detections):
        detections[index]['plate_number'] = request.json.get('plate_number')
        save_detections(detections)
        return jsonify({'success': 'Detection updated'})
    return jsonify({'error': 'Detection not found'}), 404

@app.route('/api/detections/<int:index>', methods=['DELETE'])
def delete_detection(index):
    detections = load_detections()
    if 0 <= index < len(detections):
        detections.pop(index)
        save_detections(detections)
        return jsonify({'success': 'Detection deleted'})
    return jsonify({'error': 'Detection not found'}), 404

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port="7777")


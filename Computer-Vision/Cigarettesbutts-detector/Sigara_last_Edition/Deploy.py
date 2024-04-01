from flask import Flask, render_template, request, send_file
import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import os

app = Flask(__name__)

# Set up the YOLO model


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    
    file = request.files['file']

    # Save the uploaded file to a temporary location
    file_path = "C:\\Users\\Eyad\\Downloads\\Sigara_last_Edition\\img_to_process\\im.jpg"  # Change this path as per your requirements
    file.save(file_path)
    
    # Load and preprocess the image
    image = cv2.imread(file_path)
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filename = "C:\\Users\\Eyad\\Downloads\\Sigara_last_Edition\\img_processed\\im.jpg"
    cv2.imwrite(filename, image)
    file.save(filename)
    
    
    final_image = cv2.imread(filename)
    
    # Perform prediction with the model
    model = YOLO('best.pt')
    pred = model.predict(final_image, boxes=True, conf=0.05, iou = 0.85)
    boxes = pred[0].boxes
    depth = int(request.form['depth'])  # Assuming a depth input from a form

    # Calculate the final prediction
    final_ciggs_approx = depth * len(boxes)

    # Remove the temporary file
    # Make sure to handle file cleanup properly in a production environment
    os.remove(file_path)

    return f"Estimated number of cigarettes: {len(boxes)}  , {final_ciggs_approx}"

if __name__ == '__main__':
    app.run(debug=True)
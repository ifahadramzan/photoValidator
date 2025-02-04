import os
import io
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import vision
from google.cloud.vision_v1.types import Feature
from PIL import Image
from flask_cors import CORS

# Authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "photoid-validator.json"
client = vision.ImageAnnotatorClient()

# Flask App
app = Flask(__name__)
# Add CORS support
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["OPTIONS", "POST", "GET"],
        "allow_headers": ["Content-Type"]
    }
})

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance

def calculate_brightness(image_path):
    image = Image.open(image_path).convert("L") 
    stat = np.array(image)
    return np.mean(stat) 

def analyze_image(image_path):
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # PIL to check dimensions
    pil_image = Image.open(image_path)
    img_width, img_height = pil_image.size

    features = [
        Feature(type=Feature.Type.FACE_DETECTION),
        Feature(type=Feature.Type.IMAGE_PROPERTIES),
        Feature(type=Feature.Type.LABEL_DETECTION),
        Feature(type=Feature.Type.TEXT_DETECTION),
    ]

    response = client.annotate_image({"image": image, "features": features})
    result = {
        "status": "PASS",
        "passed": [],
        "failed": []
    }

    # 1. Check if it's a family photo (Multiple Faces)
    if len(response.face_annotations) > 1:
        result["status"] = "FAIL"
        result["failed"].append("1. Not a family photo")
    else:
        result["passed"].append("1. Not a family photo")

    # 2. Check Passport Type Photo (Aspect Ratio & Face Size)
    aspect_ratio = round(img_width / img_height, 2)

    if not (0.99 <= aspect_ratio <= 1.01 or 1.3 <= aspect_ratio <= 1.35):
        result["status"] = "FAIL"
        result["failed"].append("2. Passport type photo (Invalid aspect ratio)")
    else:
        result["passed"].append("2. Passport type photo (Valid aspect ratio)")

    # 3. Check Face Visibility (Face must be detected)
    if not response.face_annotations:
        result["status"] = "FAIL"
        result["failed"].append("3. Face must be clearly visible")
        return result
    else:
        result["passed"].append("3. Face must be clearly visible")

    # 4. Check Face Size (50-70% of the image height)
    face = response.face_annotations[0]
    x1, y1 = face.bounding_poly.vertices[0].x, face.bounding_poly.vertices[0].y
    x2, y2 = face.bounding_poly.vertices[2].x, face.bounding_poly.vertices[2].y
    face_height = y2 - y1

    face_ratio = (face_height / img_height) * 100
    if face_ratio < 50 or face_ratio > 70:
        result["status"] = "FAIL"
        result["failed"].append(f"2. Passport type photo (Face too small or too large, covers {face_ratio:.2f}% of image)")
    else:
        result["passed"].append(f"2. Passport type photo (Face size is {face_ratio:.2f}% of image)")

    # 5. Ensure the face is centered
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    image_center_x = img_width / 2
    image_center_y = img_height / 2

    x_offset = abs(face_center_x - image_center_x) / img_width * 100
    y_offset = abs(face_center_y - image_center_y) / img_height * 100

    if x_offset > 10 or y_offset > 15:
        result["status"] = "FAIL"
        result["failed"].append(f"2. Passport type photo (Face is not centered, offset: X={x_offset:.2f}%, Y={y_offset:.2f}%)")
    else:
        result["passed"].append(f"2. Passport type photo (Face is centered, offset: X={x_offset:.2f}%, Y={y_offset:.2f}%)")

    # 6. Check Background Color
    dominant_colors = response.image_properties_annotation.dominant_colors.colors
    light_background = any(
        c.color.red > 200 and c.color.green > 200 and c.color.blue > 200 for c in dominant_colors
    )

    if not light_background:
        result["status"] = "FAIL"
        result["failed"].append("6. Background is not white/light color")
    else:
        result["passed"].append("6. Background is white/light color")

    return result

@app.route('/validate-photo', methods=['POST'])
def validate_photo():
    #API Endpoint with Zoho Form
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = f"temp_{file.filename}"
    file.save(file_path)

    result = analyze_image(file_path)

    # Remove the temp file after processing
    os.remove(file_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

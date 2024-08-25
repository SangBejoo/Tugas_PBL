from flask import Flask, request, jsonify, render_template, render_template_string
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import joblib
import io
import base64
import numpy as np
from chat import get_response
from model import NeuralNet

app = Flask(__name__)

class_names = {
    0: 'red',
    1: 'white',
    2: 'purple',
    3: 'pink',
    4: 'blue',
    5: 'orange',
    6: 'black',
    7: 'silver',
    8: 'brown',
    9: 'green',
    10: 'yellow',
    11: 'grey'
}

num_labels = 12

# Define the model architecture for color classification
class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=6, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Load the pre-trained color classification model
color_model = ColorClassifier()
state_dict = torch.load("color_classifier.pth", map_location=torch.device('cpu'))
color_model.load_state_dict(state_dict)
color_model.eval()

# Load models and scaler
with open('color_combination_model.pkl', 'rb') as f:
    color_combination_model = joblib.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

# Load color data
colors_df = pd.read_csv('colors.csv')

def get_color_recommendations(input_rgb, n_recommendations=5):
    distances = np.sqrt((colors_df[['r', 'g', 'b']] - input_rgb) ** 2).sum(axis=1)
    nearest_indices = distances.nsmallest(n_recommendations).index
    recommended_colors = colors_df.loc[nearest_indices]
    return recommended_colors

def get_rgb_from_color_name(color_name):
    color_row = colors_df[colors_df['color_name'].str.lower() == color_name.lower()]
    if not color_row.empty:
        return color_row[['r', 'g', 'b']].values[0]
    else:
        return None

def predict_color_combination(color1_name, color2_name):
    color1_rgb = get_rgb_from_color_name(color1_name)
    color2_rgb = get_rgb_from_color_name(color2_name)
    if color1_rgb is not None and color2_rgb is not None:
        color1_rgb_scaled = np.array([color1_rgb]).reshape(1, -1)
        color2_rgb_scaled = np.array([color2_rgb]).reshape(1, -1)
        X_input = np.hstack((color1_rgb_scaled, color2_rgb_scaled))
        X_input_scaled = scaler.transform(X_input)
        prediction = color_combination_model.predict(X_input_scaled)
        return np.round(prediction[0]).astype(int)
    else:
        return None

def get_color_name_from_rgb(rgb):
    distances = np.sqrt((colors_df[['r', 'g', 'b']] - rgb) ** 2).sum(axis=1)
    nearest_color = colors_df.loc[distances.idxmin()]
    return nearest_color['color_name']

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    color_names = colors_df['color_name'].tolist()
    return render_template('index.html', color_names=color_names)

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/predict_color', methods=['POST'])
def predict_color():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img_original = img.copy()  # Make a copy for display purposes
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = color_model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        if predicted_class in class_names:
            predicted_color = class_names[predicted_class]
        else:
            predicted_color = 'unknown'

        img_byte_arr = io.BytesIO()
        img_original.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        img_str = 'data:image/jpeg;base64,' + base64.b64encode(img_byte_arr).decode('utf-8')

        return jsonify({'predicted_class': predicted_class, 'predicted_color': predicted_color, 'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = color_model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        if predicted_class in class_names:
            predicted_color = class_names[predicted_class]
        else:
            predicted_color = 'unknown'

        return jsonify({'predicted_color': predicted_color})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.post("/predict_chat")
def predict_chat():
    text = request.get_json().get("message")
    
    try:
        np_version = np.__version__
        print(f"NumPy version: {np_version}")
    except Exception as e:
        print(f"NumPy is not available: {e}")
        return jsonify({"answer": "Error: NumPy is not available."})
    
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.route('/recommend', methods=['POST'])
def recommend():
    color_name = request.form.get('color_name')
    input_rgb = get_rgb_from_color_name(color_name)
    if input_rgb is not None:
        recommended_colors = get_color_recommendations(input_rgb)
        return render_template_string(
            '{% if recommend_results %}<h3>Recommended Colors for {{ selected_color_name }}</h3><ul>{% for color in recommend_results %}<li>{{ color["color_name"] }} ({{ color["color_code"] }})</li>{% endfor %}</ul>{% endif %}',
            recommend_results=recommended_colors.to_dict(orient='records'),
            selected_color_name=color_name
        )
    else:
        return render_template_string(
            '<h3>Error</h3><p>Color not found. Please try again.</p>'
        ), 400

@app.route('/predict_combination', methods=['POST'])
def predict_combination():
    color1_name = request.form.get('color1_name')
    color2_name = request.form.get('color2_name')
    combination_rgb = predict_color_combination(color1_name, color2_name)
    if combination_rgb is not None:
        combination_color_name = get_color_name_from_rgb(combination_rgb)
        return render_template_string(
            '{% if combination_results %}<h3>Color Combination Result</h3><p><strong>First Color:</strong> {{ combination_results.color1 }}</p><p><strong>Second Color:</strong> {{ combination_results.color2 }}</p><p><strong>Combined Color RGB:</strong> ({{ combination_results.combination_rgb[0] }}, {{ combination_results.combination_rgb[1] }}, {{ combination_results.combination_rgb[2] }})</p><p><strong>Combined Color Hex:</strong> {{ combination_results.combination_color_hex }}</p><p><strong>Combined Color Name:</strong> {{ combination_results.combination_color_name }}</p>{% endif %}',
            combination_results={
                'color1': color1_name,
                'color2': color2_name,
                'combination_rgb': combination_rgb,
                'combination_color_name': combination_color_name,
                'combination_color_hex': '#{:02x}{:02x}{:02x}'.format(combination_rgb[0], combination_rgb[1], combination_rgb[2])
            }
        )
    else:
        return render_template_string(
            '<h3>Error</h3><p>One or both colors not found. Please try again.</p>'
        ), 400


if __name__ == "__main__":
    app.run(debug=True)

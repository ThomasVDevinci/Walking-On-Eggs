from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import json
import numpy as np
import pandas as pd
import folium
from collections import defaultdict
from PIL import Image

app = Flask(__name__)

# Charger le modèle YOLOv8
model = YOLO('../model/best.pt')

# Fonction pour charger les entrées depuis entries.json
def load_entries():
    try:
        with open('entries.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Fonction pour sauvegarder les entrées
def save_entries(entries):
    with open('entries.json', 'w') as f:
        json.dump(entries, f)

# Fonction pour ajouter une entrée en cumulant les classes détectées
def add_entry(entry):
    entries = load_entries()
    found = False
    
    for existing_entry in entries:
        if existing_entry['latitude'] == entry['latitude'] and existing_entry['longitude'] == entry['longitude']:
            existing_entry['object_classes'][entry['object_class']] = existing_entry['object_classes'].get(entry['object_class'], 0) + 1
            found = True
            break
    
    if not found:
        entry['object_classes'] = {entry['object_class']: 1}
        entries.append(entry)
    
    save_entries(entries)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    # Lire l'image et s'assurer qu'elle est au bon format
    img = Image.open(file.stream).convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # YOLO attend du BGR
    
    # Redimensionner l'image pour YOLO (par exemple 640x640)
    img_resized = cv2.resize(img, (640, 640))  # Redimensionner à 640x640 ou à la taille attendue par votre modèle
    
    # Effectuer la prédiction avec YOLO
    results = model(img_resized)  # Ne pas ajouter de dimension batch, YOLO s'en charge
    result = results[0]
    
    # Extraire les boîtes détectées
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 'No objects detected', 400
    
    xywh_cpu = boxes.xywh.cpu().numpy()
    conf_cpu = boxes.conf.cpu().numpy()
    cls_cpu = boxes.cls.cpu().numpy()
    
    class_names = result.names  # Dictionnaire {ID: Nom de classe}
    
    df = pd.DataFrame({
        'x': xywh_cpu[:, 0],
        'y': xywh_cpu[:, 1],
        'width': xywh_cpu[:, 2],
        'height': xywh_cpu[:, 3],
        'confidence': conf_cpu,
        'class': cls_cpu
    })
    
    df['name'] = df['class'].apply(lambda x: class_names.get(int(x), 'Unknown'))
    detections = df[df['confidence'] > 0.5]
    
    # Vérifier et convertir les coordonnées GPS
    try:
        lat = float(request.form.get('latitude', 0))
        lon = float(request.form.get('longitude', 0))
    except ValueError:
        return 'Invalid latitude or longitude', 400
    
    if lat == 0 or lon == 0:
        return 'Coordinates not provided or invalid', 400
    
    # Ajouter les objets détectés à entries.json
    for _, detection in detections.iterrows():
        entry = {
            'latitude': lat,
            'longitude': lon,
            'object_class': detection['name']
        }
        add_entry(entry)
    
    # Générer la carte Folium
    map_ = folium.Map(location=[lat, lon], zoom_start=12)
    entries = load_entries()
    grouped_entries = defaultdict(lambda: defaultdict(int))
    
    for entry in entries:
        grouped_entries[(entry['latitude'], entry['longitude'])].update(entry['object_classes'])
    
    for (lat, lon), object_classes in grouped_entries.items():
        label = ', '.join([f"{cls}: {count}" for cls, count in object_classes.items()])
        folium.Marker([lat, lon], popup=label, icon=folium.Icon(color='blue')).add_to(map_)
    
    map_html = 'static/map.html'
    map_.save(map_html)
    
    return render_template('result.html', map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)

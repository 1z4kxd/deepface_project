import os
import warnings
import logging
import partner_a_vision

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import time

last_ai_check_time = 0
cached_ai_results = []

def classify_emotion(face_roi):
    try:
        processed_gray = partner_a_vision.preprocess_face(face_roi)
        
        if processed_gray is None:
            raise ValueError("Preprocessing failed to return an image.")
            
        processed_bgr = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)

        results = DeepFace.analyze(processed_bgr, actions=['emotion'], enforce_detection=False, silent=True)
        
        emotions = results[0]['emotion'] 
        dominant = results[0]['dominant_emotion']
        return dominant, emotions
        
    except Exception as e:
        print(f"Classification Error: {e}")
        return "unknown", {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}

def draw_rich_visuals(frame, faces):
    """B2: Draws color-coded boxes, labels, and mini bar charts with caching."""
    global last_ai_check_time, cached_ai_results
    
    current_time = time.time()
    
    if current_time - last_ai_check_time > 1.0 or len(faces) != len(cached_ai_results):
        cached_ai_results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            dominant, emotions = classify_emotion(face_roi)
            cached_ai_results.append((dominant, emotions))
        last_ai_check_time = current_time
        
    for i, (x, y, w, h) in enumerate(faces):
        if i < len(cached_ai_results):
            dominant, emotions = cached_ai_results[i]
        else:
            dominant, emotions = "neutral", {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 100}
            
        color = (0, 255, 0) if dominant == 'happy' else (0, 0, 255) if dominant == 'angry' else (255, 0, 0)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        conf = emotions.get(dominant, 0)
        label = f"{dominant.upper()}: {conf:.1f}%"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        chart_x, chart_y = x + w + 10, y
        for j, (emo, score) in enumerate(emotions.items()):
            bar_width = int((score / 100) * 50) 
            cv2.putText(frame, emo[:3], (chart_x, chart_y + (j*15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.rectangle(frame, (chart_x + 25, chart_y + (j*15) - 8), (chart_x + 25 + bar_width, chart_y + (j*15) + 2), color, -1)
            
    return frame

def run_batch_analysis(dataset_path="dataset", detection_func=None):
    true_labels = []
    pred_labels = []
    results_data = []

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found. Please create it and add images.")
        return

    for true_emotion in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, true_emotion)
        if not os.path.isdir(folder_path): continue
            
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
                
            faces = detection_func(img) if detection_func else []
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = img[y:y+h, x:x+w]
            else:
                face_roi = img
                
            dominant, _ = classify_emotion(face_roi)
                
            true_labels.append(true_emotion)
            pred_labels.append(dominant)
            results_data.append({"Image": img_name, "True": true_emotion, "Predicted": dominant})
            
            print(f"Processed {img_name} -> Predicted: {dominant}")
            
    if not results_data:
        print("No images were processed. Ensure your dataset folder has valid images.")
        return

    df = pd.DataFrame(results_data)
    df.to_csv("results.csv", index=False)
    print("Saved results.csv")
    
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n=============================")
    print(f"Overall Accuracy: {acc:.2f}")
    print(f"=============================\n")
    
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels)
    
    plt.title("Emotion Classification Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

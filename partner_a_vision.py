import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_image, target_size=(224, 224)):
    if face_image is None or face_image.size == 0:
        return None
        
    resized = cv2.resize(face_image, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_processed_face = clahe.apply(gray)
    
    return final_processed_face

def detect_faces(image, scale_factor=1.1, min_neighbors=7, min_size=(50, 50)):
    if image is None:
        return []
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors, 
        minSize=min_size
    )
    
    if len(faces) == 0:
        return []
        
    return faces

def run_webcam_loop(process_frame_callback=None):
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    skip_n_frames = 3
    fps = 0
    start_time = time.time()
    last_faces = [] 
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        
        if frame_count % skip_n_frames == 0:
            last_faces = detect_faces(frame)
            
        if process_frame_callback:
            frame = process_frame_callback(frame, last_faces)
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Facial Emotion Recognition - Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{frame_count}.jpg", frame)
            print("Screenshot saved!")

    cap.release()
    cv2.destroyAllWindows()
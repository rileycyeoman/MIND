import os
import cv2
import mediapipe as mp
import numpy as np

def get_landmarks(image):
    # Initialize MediaPipe Face Mesh solution
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    # Convert the image color space from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform face mesh detection
    results = face_mesh.process(rgb_image)
    
    # Check if any landmarks are detected
    if not results.multi_face_landmarks:
        print("No face landmarks detected.")
        return []
    
    # Get the first detected face's landmarks
    face_landmarks = results.multi_face_landmarks[0].landmark
    
    # Extract the landmarks' (x, y, z) coordinates
    landmarks = []
    for landmark in face_landmarks:
        landmarks.append((landmark.x, landmark.y, landmark.z))
    
    return landmarks

def visualize_landmarks_on_white_background(image, landmarks):
    height, width, _ = image.shape
    # Create a white background image
    white_background = np.ones((height, width, 3), dtype=np.uint8) * 255
    for landmark in landmarks:
        x = int(landmark[0] * width)
        y = int(landmark[1] * height)
        cv2.circle(white_background, (x, y), 1, (0, 0, 0), -1)  # Draw a small black circle at each landmark
    
    return white_background

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Load the image
        image = cv2.imread(input_path)
        if image is None:
            continue
        
        # Get landmarks
        landmarks = get_landmarks(image)
        
        # Visualize landmarks on a white background
        if landmarks:
            output_image = visualize_landmarks_on_white_background(image, landmarks)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, output_image)
            print(f"Processed and saved: {output_path}")

input_folder = 'models_vit/data/train'
output_folder = 'models_vit/data/train_landmarks'
process_folder(input_folder, output_folder)
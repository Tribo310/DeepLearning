import cv2
import torch
import numpy as np
from torchvision import transforms

# ============================================
# 1. FACE DETECTOR (OpenCV DNN)
# ============================================
def load_face_detector():
    net = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    )
    return net

def detect_faces(net, image, confidence_threshold=0.5):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Padding нэмэх
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            faces.append((x1, y1, x2, y2, float(confidence)))
    return faces

# ============================================
# 2. LANDMARK MODEL (таны ResNet9)
# ============================================
def load_landmark_model(checkpoint_path, device):
    model = YourResNet9()  # таны model class
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # таны model input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

def predict_landmarks(model, face_crop, device):
    """Face crop → landmark coordinates буцаана"""
    h, w = face_crop.shape[:2]
    
    input_tensor = transform(face_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)  # shape: [1, 10] (5 landmarks * 2)
    
    landmarks = output.cpu().numpy().reshape(-1, 2)
    
    # Normalize хийсэн координатыг буцааж scale хийх
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h
    
    return landmarks

# ============================================
# 3. MAIN PIPELINE
# ============================================
LANDMARK_NAMES = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

def process_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model-уудыг ачаалах
    face_net = load_face_detector()
    landmark_model = load_landmark_model(
        'checkpoints/resnet9_landmark_epoch3_loss0.0000.pt', 
        device
    )
    
    image = cv2.imread(image_path)
    display = image.copy()
    
    # Step 1: Face detection
    faces = detect_faces(face_net, image)
    print(f"Илрүүлсэн нүүр: {len(faces)}")
    
    for (x1, y1, x2, y2, conf) in faces:
        # Face box зурах
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(display, f"{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Step 2: Face crop → landmark prediction
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
            
        landmarks = predict_landmarks(landmark_model, face_crop, device)
        
        # Landmark-уудыг эргүүлэн original image coordinate болгох
        for idx, (lx, ly) in enumerate(landmarks):
            abs_x = int(x1 + lx)
            abs_y = int(y1 + ly)
            cv2.circle(display, (abs_x, abs_y), 4, (0, 255, 0), -1)
            cv2.putText(display, LANDMARK_NAMES[idx], (abs_x+5, abs_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imshow('Result', display)
    cv2.waitKey(0)

if __name__ == '__main__':
    process_image('tony_stark.jpg')
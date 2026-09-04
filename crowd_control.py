import cv2
import numpy as np
import time
import os

def opening_animation():
    cv2.namedWindow("AI Crowd Control", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AI Crowd Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start = time.time()
    particles = []

    while True:
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)

        elapsed = time.time() - start

        cv2.putText(frame, "HI DUMBS", (450, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

        cx, cy = 640, 420
        size_w = 40
        size_h = 60
        
        angle_deg = (elapsed * 120.0) % 360.0
        angle_rad = np.radians(angle_deg)
        
        pts = np.array([
            [-size_w, -size_h],
            [size_w, -size_h],
            [0, 0],
            [-size_w, size_h],
            [size_w, size_h],
            [0, 0]
        ], dtype=np.float32)
        
        pts_inner = pts * 0.75
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        def get_rotated(p):
            rot = np.zeros_like(p)
            rot[:, 0] = p[:, 0] * cos_a - p[:, 1] * sin_a + cx
            rot[:, 1] = p[:, 0] * sin_a + p[:, 1] * cos_a + cy
            return rot.astype(np.int32)
            
        rot_pts = get_rotated(pts)
        rot_pts_inner = get_rotated(pts_inner)

        y1 = (rot_pts[0][1] + rot_pts[1][1]) / 2.0
        y2 = (rot_pts[3][1] + rot_pts[4][1]) / 2.0
        
        if y1 < y2:
            top_bulb = rot_pts[0:3]
            bottom_bulb = rot_pts[3:6]
        else:
            top_bulb = rot_pts[3:6]
            bottom_bulb = rot_pts[0:3]
            
        progress = min(elapsed / 3.0, 1.0)
        fill_top = 1.0 - progress
        fill_bottom = progress

        sand_mask = np.zeros((720, 1280), dtype=np.uint8)
        
        if fill_top > 0:
            mask_t = np.zeros((720, 1280), dtype=np.uint8)
            cv2.fillPoly(mask_t, [top_bulb], 255)
            min_y_top = int(np.min(top_bulb[:, 1]))
            h_top = int((cy - min_y_top) * np.sqrt(fill_top))
            sand_rect_t = np.zeros((720, 1280), dtype=np.uint8)
            cv2.rectangle(sand_rect_t, (0, cy - h_top), (1280, cy), 255, -1)
            cv2.bitwise_or(sand_mask, cv2.bitwise_and(mask_t, sand_rect_t), sand_mask)
            
        if fill_bottom > 0:
            mask_b = np.zeros((720, 1280), dtype=np.uint8)
            cv2.fillPoly(mask_b, [bottom_bulb], 255)
            max_y_bot = int(np.max(bottom_bulb[:, 1]))
            h_bot = int((max_y_bot - cy) * np.sqrt(fill_bottom))
            sand_rect_b = np.zeros((720, 1280), dtype=np.uint8)
            cv2.rectangle(sand_rect_b, (0, max_y_bot - h_bot), (1280, max_y_bot), 255, -1)
            cv2.bitwise_or(sand_mask, cv2.bitwise_and(mask_b, sand_rect_b), sand_mask)
            
        frame[sand_mask == 255] = (34, 180, 238)
        
        cv2.polylines(frame, [rot_pts[0:3]], True, (120, 120, 120), 3, cv2.LINE_AA)
        cv2.polylines(frame, [rot_pts[3:6]], True, (120, 120, 120), 3, cv2.LINE_AA)
        cv2.polylines(frame, [rot_pts_inner[0:3]], True, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.polylines(frame, [rot_pts_inner[3:6]], True, (200, 200, 200), 1, cv2.LINE_AA)
        
        for i in range(len(rot_pts)):
            cv2.line(frame, tuple(rot_pts[i]), tuple(rot_pts_inner[i]), (180, 180, 180), 1, cv2.LINE_AA)

        if progress < 1.0:
            if np.random.rand() > 0.1:
                particles.append([cx + np.random.uniform(-1.5, 1.5), cy, np.random.uniform(-0.5, 0.5), np.random.uniform(5, 8)])
                
        new_particles = []
        for p in particles:
            p[0] += p[2]
            p[1] += p[3]
            
            max_y_bot = int(np.max(bottom_bulb[:, 1]))
            h_bot = int((max_y_bot - cy) * np.sqrt(fill_bottom))
            surface_y = max_y_bot - h_bot
            
            if p[1] < surface_y:
                new_particles.append(p)
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (34, 180, 238), -1, cv2.LINE_AA)
                
        particles = new_particles

        cv2.putText(frame, "Initializing...", (450, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("AI Crowd Control", frame)

        if elapsed > 3:
            break

        if cv2.waitKey(1) == 27:
            break


opening_animation()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def train_suspect_recognizer(suspects_dir, cascade):
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print(
            "Suspect recognition disabled: install opencv-contrib-python "
            "to enable cv2.face.LBPHFaceRecognizer_create()."
        )
        return None, False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces_data = []
    labels = []
    
    if not os.path.exists(suspects_dir):
        return None, False

    valid_images = [f for f in os.listdir(suspects_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not valid_images:
        return None, False
        
    for idx, filename in enumerate(valid_images):
        filepath = os.path.join(suspects_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        detected_faces = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in detected_faces:
            faces_data.append(img[y:y+h, x:x+w])
            labels.append(1)
            
    if not faces_data:
        return None, False
        
    recognizer.train(faces_data, np.array(labels))
    return recognizer, True

suspect_recognizer, has_suspects = train_suspect_recognizer("suspects", face_cascade)

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

boxes = []
faces = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    frame_count += 1

    if frame_count % 3 == 0:
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    crowd_count = max(len(boxes), len(faces))

    cv2.putText(frame, f"Crowd: {crowd_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    high_crowd = crowd_count > 10

    if high_crowd:
        cv2.putText(frame, "HIGH CROWD DENSITY", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    for (x, y, w, h) in faces:
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        is_threat = False
        if has_suspects:
            face_roi = gray[y:y+h, x:x+w]
            try:
                label, confidence = suspect_recognizer.predict(face_roi)
                if confidence < 80:
                    is_threat = True
            except:
                pass
                
        if is_threat or (high_crowd and w > 100):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "POSSIBLE THREAT", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("AI Crowd Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

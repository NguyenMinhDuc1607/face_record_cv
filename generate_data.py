import cv2
import os
import numpy as np
from os.path import join

cascade_path = 'cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
new_dimension = (200, 200)
data_path = 'trainingData'
os.makedirs(data_path, exist_ok=True)

# Tải mô hình LBPH để so sánh khuôn mặt
model = cv2.face.LBPHFaceRecognizer_create()
model_path = 'models/model.yml'
if os.path.exists(model_path):
    model.read(model_path)

# Đọc file person_map.txt để lấy thông tin ID và tên
person_map = {}
if os.path.exists('models/person_map.txt'):
    with open('models/person_map.txt', 'r') as f:
        for line in f:
            person_id, name = line.strip().split(':')
            person_map[int(person_id)] = name

def get_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]  # Lấy mặt đầu tiên
    return (x, y, w, h), gray[y:y + h, x:x + w]

# Nhập tên người dùng
name = input("Nhập tên của người: ").strip()
if not name:
    print("[ERROR] Tên không được để trống.")
    exit()

# Tạo ID mới (dựa trên số thư mục hiện có)
person_id = len([d for d in os.listdir(data_path) if os.path.isdir(join(data_path, d))]) + 1
person_path = join(data_path, f'person_{person_id}')
os.makedirs(person_path, exist_ok=True)

cap = cv2.VideoCapture(0)
num_faces = 0
max_faces = 200  # Số lượng ảnh tối đa cho mỗi người

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Không thể truy cập webcam.")
        break
    (box, face) = get_face(frame)
    if face is not None:
        face_resized = cv2.resize(face, new_dimension)
        # Kiểm tra xem khuôn mặt đã tồn tại trong mô hình chưa
        if os.path.exists(model_path):
            label, confidence = model.predict(face_resized)
            if confidence < 40:  # Ngưỡng độ tin cậy
                print(f"[INFO] Khuôn mặt đã tồn tại (ID: {label}, Tên: {person_map.get(label, 'Unknown')}). Bỏ qua.")
                break
        # Lưu ảnh nếu khuôn mặt chưa tồn tại
        cv2.imwrite(join(person_path, f'face_{num_faces}.jpg'), face_resized)
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {num_faces} ({name})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        num_faces += 1
    cv2.imshow('Capturing Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or num_faces >= max_faces:
        break

cap.release()
cv2.destroyAllWindows()

# Lưu thông tin ID và tên vào person_map.txt
if num_faces > 0:
    with open('models/person_map.txt', 'a') as f:
        f.write(f"{person_id}:{name}\n")
    print(f"[INFO] Đã chụp {num_faces} ảnh khuôn mặt cho {name} (ID: {person_id}).")
else:
    print(f"[INFO] Không lưu dữ liệu cho {name} do không phát hiện khuôn mặt mới.")
    os.rmdir(person_path)  # Xóa thư mục nếu không có ảnh
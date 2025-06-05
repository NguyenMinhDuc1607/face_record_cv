import cv2
import os

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read('models/model.yml')

# Đọc file person_map.txt để lấy thông tin ID và tên
person_map = {}
if os.path.exists('models/person_map.txt'):
    with open('models/person_map.txt', 'r') as f:
        for line in f:
            person_id, name = line.strip().split(':')
            person_map[int(person_id)] = name

cap = cv2.VideoCapture(0)
confidence_threshold = 40  # Ngưỡng độ tin cậy (có thể điều chỉnh)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Không thể truy cập webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200))
        label, confidence = model.predict(roi_gray)

        conf_percent = int(100 * (1 - confidence / 400))
        if confidence < confidence_threshold and label in person_map:
            result_text = f"ID: {label} ({person_map[label]})"
            color = (0, 255, 0)  # Xanh cho người được nhận diện
        else:
            result_text = "Not Detected!"
            color = (0, 0, 255)  # Đỏ cho người lạ

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {conf_percent}%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
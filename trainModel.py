import cv2
import numpy as np
import os
from os import listdir
from os.path import isdir, isfile, join
import time

data_path = 'trainingData/'
person_dirs = [d for d in listdir(data_path) if isdir(join(data_path, d))]

if not person_dirs:
    print("[ERROR] Không tìm thấy thư mục dữ liệu trong trainingData.")
    exit()

faces = []
labels = []

for person_dir in person_dirs:
    person_id = int(person_dir.split('_')[1])  # Lấy ID từ tên thư mục (person_X)
    person_path = join(data_path, person_dir)
    files = [f for f in listdir(person_path) if isfile(join(person_path, f))]
    for file in files:
        img_path = join(person_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Không thể đọc ảnh {img_path}.")
            continue
        faces.append(np.asarray(img, dtype=np.uint8))
        labels.append(person_id)

faces = np.asarray(faces)
labels = np.asarray(labels, dtype=np.int32)

if len(faces) == 0:
    print("[ERROR] Không có dữ liệu khuôn mặt hợp lệ để huấn luyện.")
    exit()

model = cv2.face.LBPHFaceRecognizer_create()

# Đo thời gian huấn luyện
# start_time = time.time_ns()
#
# for _ in range(10):
#     model.train(faces, labels)
#
# end_time = time.time_ns()
# training_time = (end_time - start_time) / (10 * 1000)
# print(f"[INFO] Thời gian huấn luyện: {training_time:.2f} microseconds")

model.train(faces, labels)

os.makedirs('models', exist_ok=True)
model.save('models/model.yml')
print("[INFO] Huấn luyện hoàn tất và mô hình đã được lưu.")
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from ultralytics import YOLO

# Initialize FaceNet and MTCNN
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
detector = MTCNN()
model = pickle.load(open("svm_model_160x160_video.pkl", 'rb'))

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # You can use 'yolov8n.pt' for a smaller model

# Khởi tạo VideoWriter
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Đọc video
video_path = r"data/video/The_Gallagher.mp4"
cap = cv.VideoCapture(video_path)

# Giảm độ phân giải xuống 480x360
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Sử dụng YOLO để phát hiện người
    results = yolo_model(frame)

    # Xử lý kết quả từ YOLO
    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # Assuming class 0 is 'person'
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                face_img = frame[y:y+h, x:x+w]
                rgb_img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)

                # Detect faces using MTCNN
                faces = detector.detect_faces(rgb_img)
                for face in faces:
                    fx, fy, fw, fh = face['box']
                    img = rgb_img[fy:fy+fh, fx:fx+fw]
                    img = cv.resize(img, (160, 160))  # 1x160x160x3
                    img = np.expand_dims(img, axis=0)
                    ypred = facenet.embeddings(img)

                    # Assuming model.predict_proba gives probabilities
                    probabilities = model.predict_proba(ypred)
                    max_prob = np.max(probabilities)
                    face_name = model.predict(ypred)

                    if max_prob >= 0.4:  # Nếu độ chính xác >= 40%
                        final_name = encoder.inverse_transform(face_name)[0]
                        display_text = f"{final_name} ({max_prob * 100:.2f}%)"
                    else:  # Nếu không nhận diện được, hiển thị "unknown"
                        display_text = "unknown"

                    # Vẽ hộp và hiển thị tên + độ chính xác (hoặc unknown)
                    cv.rectangle(frame, (x + fx, y + fy), (x + fx + fw, y + fy + fh), (255, 0, 255), 2)
                    cv.putText(frame, display_text, (x + fx, y + fy - 10), cv.FONT_HERSHEY_SIMPLEX,
                               1, (0, 0, 255), 2, cv.LINE_AA)  # Màu đỏ cho chữ

    out.write(frame)
    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()

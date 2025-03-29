# Load model SVM
model_path = "svm_model_160x160.pkl"
svm_model = pickle.load(open(model_path, 'rb'))

# Load embeddings & encoder
data_path = "faces_embeddings_done_4classes.npz"
faces_embeddings = np.load(data_path)
Y = faces_embeddings['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)

# Khởi tạo MTCNN và FaceNet
detector = MTCNN()
facenet = FaceNet()

# Load ảnh kiểm tra
image_path = "Fiona Gallagher Icon.jpg"  # Thay bằng đường dẫn ảnh của bạn
t_im = cv.imread(image_path)
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)

# Phát hiện khuôn mặt
faces = detector.detect_faces(t_im)

if len(faces) == 0:
    print("Không tìm thấy khuôn mặt nào trong ảnh.")
else:
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)  # Đảm bảo tọa độ không âm

    # Cắt và resize khuôn mặt
    face_img = t_im[y:y+h, x:x+w]
    face_img = cv.resize(face_img, (160, 160))
    face_img = np.expand_dims(face_img, axis=0)

    # Trích xuất đặc trưng
    test_embedding = facenet.embeddings(face_img)

    # Dự đoán bằng SVM
    ypreds = svm_model.predict(test_embedding)
    predicted_name = encoder.inverse_transform(ypreds)[0]

    # Tính độ chính xác
    proba = svm_model.predict_proba(test_embedding)
    confidence = max(proba[0])

    print(f"Dự đoán: {predicted_name} (Độ chính xác: {confidence:.2f})")
    if confidence < 0.5:
        print("=> Kết quả không chắc chắn, có thể là Unknown.")
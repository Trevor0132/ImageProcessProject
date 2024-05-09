import threading
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage.feature import hog
from skimage.filters import gabor
from skimage.feature import local_binary_pattern

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, 32, 1, method="uniform")
    # 降维
    pca = PCA(n_components=64)
    lbp_pca = pca.fit_transform(lbp)
    return lbp_pca.flatten()

# 提取Gabor特征
def extract_gabor_features(image):
    gabor_features = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        gabor_filter_real, gabor_filter_imag = gabor(image, frequency=0.6, theta=theta)
        
        img_mod=np.sqrt(gabor_filter_real.astype(float)**2+gabor_filter_imag.astype(float)**2)
        #图像缩放（下采样）
        newimg = cv2.resize(img_mod,(0,0),fx=1/4,fy=1/4,interpolation=cv2.INTER_AREA)
        tempfea = newimg.flatten()  #矩阵展平
        gabor_features.append(tempfea)
        
        # gabor_features.append(gabor_filter_real)
        # gabor_features.append(gabor_filter_imag)
    return np.array(gabor_features).flatten()

# 提取HOG特征
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(16, 16), transform_sqrt=True, block_norm="L2-Hys", visualize=True)
    return features

def faceDetect(face_cascade, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_save = None  # 设置默认值
    
    # 在检测到的人脸周围绘制矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        face_save = cv2.resize(face_roi, (256, 256))
        
    return image, face_save

def process_image(input_image, __model):
    if input_image is None :
        return 'Face Detect Fail!'
    
    if __model == "knn":    
        model_path = "./model/hog/knn_model_1.pkl"
    elif __model == "svm":
        model_path = "./model/hog/svm_model.pkl"
    elif __model == "decision tree":
        model_path = "./model/hog/decision_tree_model.pkl"
    elif __model == "logistic regression":
        model_path = "./model/hog/Logistic_regression_classification.pkl"
    elif __model == "random forest":
        model_path = "./model/hog/Random_forest.pkl"
    elif __model == "naiva bayes":
        model_path = "./model/hog/naive_bayes_model.pkl"
    else:
        return "can't find model"
    print(model_path)
    
    model = joblib.load(model_path)
    # gabor_features = extract_gabor_features(input_image)
    hog_features = extract_hog_features(input_image)
    # lbp_features = extract_lbp_features(input_image)
    # combined_features = np.concatenate((gabor_features, hog_features))
    predict = model.predict(hog_features.reshape(1, -1))  # 可以在这里选择分类器的类别
    if predict == 0:
        print('angry')
        return 'angry'
    elif predict == 1:
        print('disgust')
        return 'disgust'
    elif predict == 2:
        print('fear')
        return 'fear'
    elif predict == 3:
        print('happy')
        return 'happy'
    elif predict == 4:
        print('neutral')
        return 'neutral'
    elif predict == 5:
        print('sad')
        return 'sad'
    elif predict == 6:
        print('surprise')
        return 'surprise'

class ImageDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression App")

        self.camera = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.image = None
        
        self.frame_count = 0
        self.process_every_n_frames = 10  # 每10帧检测一次
        
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        self.result_label = tk.Label(root)
        self.result_label.config(text="result: ")
        self.result_label.pack()

        self.detect_button = tk.Button(root, text="Detect Image", command=self.detect_image)
        self.detect_button.pack()

        self.options = ["knn", "svm", "decision tree", "logistic regression", "random forest", "naiva bayes"]

        self.model = self.options[0]

        self.dropdown = tk.StringVar()
        self.dropdown.set(self.options[0])

        self.dropdown_menu = tk.OptionMenu(root, self.dropdown, *self.options, command=self.updateModel)
        self.dropdown_menu.pack()
        
        self.camera_button = tk.Button(root, text="Play Camera", command=self.toggle_camera)
        self.camera_button.pack()

        self.lock = threading.Lock()  # 创建线程锁

        self.is_playing = False
        
    def detect_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path, 0)
            img = self.image.copy()
            # Perform detection using OpenCV
            image_with_desc = process_image(img, self.model)
            self.result_label.config(text="result: " + image_with_desc)
            self.show_image(img)

    def updateModel(self, val):
        self.model = str(val)
        print(self.model)
        if self.image is not None:
            img = self.image.copy()
            # Perform detection using OpenCV
            image_with_desc = process_image(img, self.model)
            self.result_label.config(text="result: " + image_with_desc)
            self.show_image(img)

    def show_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def toggle_camera(self):
        if self.is_playing:
            self.stop_camera()
        else:
            self.play_camera()

    def play_camera(self):
        if not self.is_playing:
            self.is_playing = True
            self.camera_thread = threading.Thread(target=self._play_camera)
            self.camera_thread.start()

    def _play_camera(self):
        while self.is_playing:
            ret, frame = self.camera.read()
            if ret:
                frame, face = faceDetect(self.face_cascade, frame)
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames == 0:
                    t = threading.Thread(target=self.process_image, args=(face,))
                    t.start()
                self.show_image(frame)
            else:
                messagebox.showerror("Error", "Failed to capture frame")
                break

    def process_image(self, input_image):
        with self.lock:
            image_with_desc = process_image(input_image, self.model)
            self.result_label.config(text="result: " + image_with_desc)

    def stop_camera(self):
        self.is_playing = False

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDetectionApp(root)
    app.run()
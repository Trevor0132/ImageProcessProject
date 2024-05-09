import cv2
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import os
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from sklearn.tree import DecisionTreeClassifier

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

def read_data(path):
    X = []
    y = []
    label2id = {}  # 定义标签字典
    for i, label in enumerate(os.listdir(path)):
        label2id[label] = i
        for img_file in os.listdir(os.path.join(path, label)):
            image = cv2.imread(os.path.join(path, label, img_file), cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
            # image = image / 255.0
            X.append(image)
            y.append(i)
    return X, y, label2id

X, y, label2id = read_data("./data/data/train")

X_combined = []
y_combined = []

i = 0
for image, label in zip(X, y):
    print(i)
    gabor_features = extract_gabor_features(image)
    # # print(gabor_features.shape)
    hog_features = extract_hog_features(image)
    # # print(hog_features.shape)
    lbp_features = extract_lbp_features(image)
    # # print(lbp_features.shape)
    combined_features = np.concatenate((gabor_features, hog_features, lbp_features))
    X_combined.append(combined_features)
    y_combined.append(label)
    i = i + 1 

X_combined = np.array(X_combined)
y_combined = np.array(y_combined)

# X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# 随机森林
rf = RandomForestClassifier(n_estimators=180)
rf.fit(X_combined, y_combined)
print("Random_forest train ok!")
joblib.dump(rf, "Random_forest.pkl")

# svm
svm = sklearn.svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')
svm.fit(X_combined, y_combined)
print("svm_model train ok!")
joblib.dump(svm, 'svm_model.pkl')

# knn
knn = KNeighborsClassifier(n_neighbors=1)  
knn.fit(X_combined,y_combined)
print("knn_model_1 train ok!")
joblib.dump(knn, 'knn_model_1.pkl')

#决策树算法 
tree_D = DecisionTreeClassifier()
tree_D.fit(X_combined,y_combined)
print("decision_tree_model train ok!")
joblib.dump(tree_D,'decision_tree_model.pkl')

#朴素贝叶斯分类 
mlt=GaussianNB()
mlt.fit(X_combined,y_combined)
print("naive_bayes_model train ok!")
joblib.dump(mlt, 'naive_bayes_model.pkl')

# #逻辑回归分类 
logistic = LogisticRegression()
logistic.fit(X_combined,y_combined)
print("Logistic_regression_classification train ok!")
joblib.dump(logistic,'Logistic_regression_classification.pkl')

X, y, label2id = read_data("./data/data/test")

X_test = []
y_test = []

i = 0
for image, label in zip(X, y):
    print(i)
    gabor_features = extract_gabor_features(image)
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    combined_features = np.concatenate((gabor_features, hog_features, lbp_features))
    X_test.append(combined_features)
    y_test.append(label)
    i = i + 1

X_test = np.array(X_test)
y_test = np.array(y_test)

# rf = joblib.load("./model/gabor/Random_forest.pkl")
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("RandomForestClassifier Accuracy: {:.2f}".format(accuracy))

# svm = joblib.load("./model/gabor/svm_model.pkl")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("svm Accuracy: {:.2f}".format(accuracy))

# knn = joblib.load("./model/gabor/knn_model_1.pkl")
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("knn Accuracy: {:.2f}".format(accuracy))

# tree_D = joblib.load("./model/gabor/decision_tree_model.pkl")
y_pred = tree_D.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("DecisionTreeClassifier Accuracy: {:.2f}".format(accuracy))

# mlt = joblib.load("./model/gabor/naive_bayes_model.pkl")
y_pred = mlt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("GaussianNB Accuracy: {:.2f}".format(accuracy))

# logistic = joblib.load("./model/gabor/Logistic_regression_classification.pkl")
y_pred = logistic.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LogisticRegression Accuracy: {:.2f}".format(accuracy))
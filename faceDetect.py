import cv2

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

i = 0

while True:
    # 读取视频流
    ret, frame = cap.read()
    
    # 将视频帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 在检测到的人脸周围绘制矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_save = cv2.resize(face_gray, (256, 256))

    # 显示视频流
    cv2.imshow('Face Detection', frame)

    key = cv2.waitKey(1)
    
    # 按下q键退出循环
    if key & 0xFF == ord('q'):
        break
    # 按下b键保存人脸图片
    elif key & 0xFF == ord('b'):
        cv2.imwrite('./MyDataSet/' + str(i) + 'face.tiff', face_save)
        i = i + 1

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
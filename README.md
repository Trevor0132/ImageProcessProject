<!--
 * @FilePath: \ImageProcessProject\README.md
 * @Brief: 
 * @Author: Trevor(wuchenfeng0132@qq.com)
 * @Date: 2024-05-09 22:16:51
-->
# 数字图像处理课程设计  
## 项目一 硬币识别
### 1、项目介绍
本项目实现了中华人民共和国1元硬币、5角硬币、1角硬币的数量检测、面额检测，具有UI界面。
### 2、方案介绍
采用霍夫变换检测圆，并通过颜色区别5角和其他硬币，再将剩下的硬币通过半径大小区别1角和1元硬币。
### 3、运行项目
项目基于python开发，UI界面使用tkinter库开发，python自带无需安装，安装opencv-python库即可，直接运行。
## 项目二 人脸表情识别
### 1、项目介绍
本项目实现人脸表情检测，实现了相机跟踪人脸并检测angry、disgust、fear、happy、neutral、sad、surprise这些表情，具有UI界面。
### 2、方案介绍
采用HOG特征和随机森林模型进行检测，数据集使用ck+和jaffe数据集的混合并参和了一百多张作者自己人脸表情的数据，经测试集测试，准确率达到90%以上。
### 3、运行项目
项目基于python开发，UI界面使用tkinter库开发，python自带无需安装，安装opencv-python、sklearn库。python版本3.7。
## 特别鸣谢
本项目为课程设计题目，实际开发周期为一周，略显粗糙，感谢共同参与开发的好友@wuyi-del。
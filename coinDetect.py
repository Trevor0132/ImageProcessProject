import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

def adjust_gamma(image, gamma=1.0):
    brighter_image = np.array(np.power((image / 255), gamma) * 255, dtype=np.uint8)
    return brighter_image

def process_image_and_display_results(input_image, gamma=1.0, minRadius=30, maxRadius=50):
    
    if minRadius > maxRadius:
        return input_image
    
    # Initialize variables for counting
    total_sum = 0
    total_count = 0

    # Convert the input image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # Apply gamma correction to adjust the brightness
    gray = adjust_gamma(gray, gamma)
    # cv2.imshow("gamma", gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("blurred", blurred)
    # Apply adaptive thresholding to create a binary image
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 3)
    # cv2.imshow("binary_image", binary_image)

    # Erosion to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    # cv2.imshow("eroded_image", eroded_image)

    # Dilation to fill gaps in the contours
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    # cv2.imshow("dilated_image", dilated_image)
    
    # Apply Canny edge detection
    edges = cv2.Canny(dilated_image, 10, 100)
    # cv2.imshow("edges", edges)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=int(minRadius), maxRadius=int(maxRadius))

    if circles is not None:
        # Iterate over detected circles
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            center = (i[0], i[1])
            radius = i[2]

            # Draw the circle on the original image
            cv2.circle(input_image, center, radius, (0, 255, 0), 2)

            # Extract the region of interest (ROI) within the circle
            x, y = center
            top_left = (max(x - radius, 0), max(y - radius, 0))
            bottom_right = (min(x + radius, input_image.shape[1]), min(y + radius, input_image.shape[0]))

            roi = input_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            if roi is None or roi.size == 0:
                return input_image
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds of the yellow color in HSV format
            lower_yellow = np.array([10, 100, 100])
            upper_yellow = np.array([50, 255, 255])

            # Create a mask to extract the yellow color range in the ROI
            mask = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)
            # cv2.imshow(str(total_count)+"mask", mask)
            # Check if any pixel in the ROI falls within the yellow color range
            is_yellow = np.any(mask)
            if is_yellow:
                total_sum += 0.5
            else:
                if radius > (maxRadius + minRadius) / 2:
                    total_sum += 1
                else:
                    total_sum += 0.1

            total_count += 1

        # Display total_sum and total_count on the output image
        cv2.putText(input_image, "Total Sum: " + str(total_sum) + " Total Count: " + str(total_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return input_image

class ImageDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coins Detection App")
        
        self.image = None

        self.camera = cv2.VideoCapture(1)
    
        self.gamma = 1
        self.minRadius = 20
        self.maxRadius = 60

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.gamma_label = tk.Label(root, text="gamma:")
        self.gamma_label.pack()

        self.gamma_slider = tk.Scale(root, from_=0, to=15, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_gamma)
        self.gamma_slider.set(self.gamma)
        self.gamma_slider.pack()

        self.minRadius_label = tk.Label(root, text="minRadius:")
        self.minRadius_label.pack()

        self.minRadius_slider = tk.Scale(root, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, command=self.update_minRadius)
        self.minRadius_slider.set(self.minRadius)
        self.minRadius_slider.pack()
        
        self.maxRadius_label = tk.Label(root, text="maxRadius:")
        self.maxRadius_label.pack()

        self.maxRadius_slider = tk.Scale(root, from_=0, to=200, resolution=1, orient=tk.HORIZONTAL, command=self.update_maxRadius)
        self.maxRadius_slider.set(self.maxRadius)
        self.maxRadius_slider.pack()

        self.detect_button = tk.Button(root, text="Detect Image", command=self.detect_image)
        self.detect_button.pack()

        self.camera_button = tk.Button(root, text="Play Camera", command=self.toggle_camera)
        self.camera_button.pack()

        self.is_playing = False

    def detect_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            img = self.image.copy()
            # Perform detection using OpenCV
            image_with_desc = process_image_and_display_results(img, self.gamma, self.minRadius, self.maxRadius)
            # description = "Detected Image: " + file_path
            # image_with_desc = cv2.putText(image_rgb, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.show_image(image_with_desc)

    def update_gamma(self, val):
        self.gamma = float(val)
        if self.image is not None:
            img = self.image.copy()
            # Perform detection using OpenCV
            image_with_desc = process_image_and_display_results(img, self.gamma, self.minRadius, self.maxRadius)
            # description = "Detected Image: " + file_path
            # image_with_desc = cv2.putText(image_rgb, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.show_image(image_with_desc)

    def update_minRadius(self, val):
        self.minRadius = float(val)
        if self.image is not None:
            img = self.image.copy()
            # Perform detection using OpenCV
            image_with_desc = process_image_and_display_results(img, self.gamma, self.minRadius, self.maxRadius)
            # description = "Detected Image: " + file_path
            # image_with_desc = cv2.putText(image_rgb, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.show_image(image_with_desc)
            
    def update_maxRadius(self, val):
        self.maxRadius = float(val)
        if self.image is not None:
            img = self.image.copy()
            # Perform detection using OpenCV
            image_with_desc = process_image_and_display_results(img, self.gamma, self.minRadius, self.maxRadius)
            # description = "Detected Image: " + file_path
            # image_with_desc = cv2.putText(image_rgb, description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.show_image(image_with_desc)

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
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = process_image_and_display_results(frame, self.gamma, self.minRadius, self.maxRadius)
                self.show_image(frame)
            else:
                messagebox.showerror("Error", "Failed to capture frame")
                break

    def stop_camera(self):
        self.is_playing = False

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDetectionApp(root)
    app.run()

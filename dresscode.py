import cv2
import numpy as np
import os
import threading
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import datetime

# === Setup paths ===
base_dir = os.path.dirname(__file__)
shirt_path = os.path.join(base_dir, 'college shirt.jpg')
logo_path = os.path.join(base_dir, 'college logo.jpg')

# === Load images ===
shirt_img = cv2.imread(shirt_path)
logo_template = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
if shirt_img is None or logo_template is None:
    raise FileNotFoundError("Reference images not found!")

# === Extract dominant dark blue color using KMeans ===
def get_dominant_hsv_color(image, k=2):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    return kmeans.cluster_centers_[0]

dominant_hsv = get_dominant_hsv_color(shirt_img)
h, s, v = dominant_hsv
lower_blue = np.array([max(int(h - 15), 0), 50, 30], dtype=np.uint8)
upper_blue = np.array([min(int(h + 15), 179), 255, 255], dtype=np.uint8)

# === ORB logo detector ===
orb = cv2.ORB_create()
kp_logo, des_logo = orb.detectAndCompute(logo_template, None)

# === Load YOLOv8 model ===
model = YOLO("yolov8n.pt")

def is_dark_blue(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(mask) / (roi.size / 3)
    return blue_ratio > 0.15

def has_logo(roi):
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(gray, None)
        if des_frame is None:
            return False
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_logo, des_frame)
        return len(matches) > 5
    except:
        return False

# === GUI Setup ===
root = Tk()
root.title("MVJ Dress Code Detection")
root.geometry("900x700")

frame_label = Label(root)
frame_label.pack()

alert_label = Label(root, text="", font=("Arial", 14), fg="red")
alert_label.pack()

def update_alert(text, color="red"):
    alert_label.config(text=text, fg=color)

cap = cv2.VideoCapture(0)

def capture_frame():
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        messagebox.showinfo("Capture", f"Image saved as {filename}")

capture_btn = Button(root, text="Capture Image", command=capture_frame)
capture_btn.pack()

def video_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)

        alert_triggered = False
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            upper_half = roi[:(y2 - y1) // 2, :]
            blue = is_dark_blue(upper_half)
            logo = has_logo(upper_half)

            compliant = blue and logo
            color = (0, 255, 0) if compliant else (0, 0, 255)

            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            radius = max((x2 - x1), (y2 - y1)) // 3
            cv2.circle(frame, center, radius, color, 3)

            if not compliant:
                alert_triggered = True

        update_alert("Non-compliant person detected!" if alert_triggered else "All persons compliant", "red" if alert_triggered else "green")

        # Convert and show in GUI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)

        root.update_idletasks()
        root.update()

threading.Thread(target=video_loop, daemon=True).start()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
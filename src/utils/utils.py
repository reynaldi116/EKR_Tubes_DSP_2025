# utils.py
import cv2
import numpy as np

# Muat Haar Cascade untuk deteksi wajah (pastikan file xml ada di path yang benar)
# Anda bisa download dari repository OpenCV di GitHub:
# https://github.com/opencv/opencv/tree/master/data/haarcascades
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Tidak dapat memuat file cascade wajah.")
except Exception as e:
    print(f"Error saat memuat cascade wajah: {e}")
    face_cascade = None

def detect_face_roi(frame):
    """
    Mendeteksi wajah dalam frame dan mengembalikan ROI.
    Args:
        frame (np.array): Frame input.
    Returns:
        tuple: (x, y, w, h) koordinat ROI wajah, atau None jika tidak terdeteksi.
    """
    if face_cascade is None:
        print("Cascade wajah tidak dimuat, menggunakan seluruh frame sebagai ROI.")
        return 0, 0, frame.shape[1], frame.shape[0] # Gunakan seluruh frame jika cascade gagal

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return faces[0]  # Ambil wajah pertama yang terdeteksi
    return None

def get_roi_pixels(frame, roi):
    """
    Mengekstrak piksel dari ROI.
    Args:
        frame (np.array): Frame input.
        roi (tuple): (x, y, w, h) koordinat ROI.
    Returns:
        np.array: Piksel dari ROI.
    """
    if roi is None:
        return frame # Kembalikan seluruh frame jika ROI tidak ada
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]
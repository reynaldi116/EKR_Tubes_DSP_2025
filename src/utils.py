# utils.py
import cv2
import mediapipe as mp
import numpy as np

class FaceDetectorMP:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Inisialisasi detektor wajah menggunakan MediaPipe Face Detection.

        Args:
            min_detection_confidence (float): Keyakinan minimum untuk deteksi.
            model_selection (int): 0 untuk model jarak pendek (hingga 2m), 
                                   1 untuk model jangkauan penuh (hingga 5m).
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection 
        )

    def detect_face_bounding_box(self, frame_rgb):
        """
        Mendeteksi wajah utama dalam frame RGB dan mengembalikan bounding box-nya.

        Args:
            frame_rgb (np.array): Frame input dalam format RGB.

        Returns:
            tuple: (x, y, w, h) untuk bounding box wajah, atau None jika tidak terdeteksi.
        """
        results = self.face_detection.process(frame_rgb)
        if results.detections:
            # Ambil deteksi pertama (biasanya yang paling menonjol atau paling pasti)
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame_rgb.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Pastikan koordinat valid
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            if w > 0 and h > 0:
                return x, y, w, h
        return None

    def close(self):
        """Melepaskan resource model deteksi wajah."""
        if self.face_detection:
            self.face_detection.close()

def get_roi_pixels(frame, roi_coords_tuple):
    """
    Mengekstrak piksel dari ROI (Region of Interest) pada frame.

    Args:
        frame (np.array): Frame input.
        roi_coords_tuple (tuple): (x, y, w, h) koordinat ROI.

    Returns:
        np.array: Piksel dari ROI, atau array kosong jika ROI tidak valid.
    """
    if roi_coords_tuple is None:
        return np.array([]) # Kembalikan array kosong jika tidak ada ROI

    x, y, w, h = roi_coords_tuple
    
    # Cek dasar validitas ROI
    if w <= 0 or h <= 0:
        return np.array([])

    frame_h, frame_w = frame.shape[:2]

    # Pastikan ROI berada dalam batas frame
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(x_start + w, frame_w)
    y_end = min(y_start + h, frame_h)

    if x_end <= x_start or y_end <= y_start: # Jika ROI menjadi tidak valid setelah clamping
        return np.array([])
        
    return frame[y_start:y_end, x_start:x_end]

# Fungsi detect_face_roi yang lama (Haar Cascade) bisa dihapus atau dikomentari jika tidak digunakan lagi.
# try:
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     if face_cascade.empty():
#         # print("Peringatan: Tidak dapat memuat file cascade wajah Haar.")
#         face_cascade = None
# except Exception as e:
#     # print(f"Error saat memuat cascade wajah Haar: {e}")
#     face_cascade = None

# def detect_face_roi_haar(frame):
#     # ... (implementasi lama dengan Haar Cascade) ...
#     pass
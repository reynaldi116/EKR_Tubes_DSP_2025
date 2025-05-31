# utils.py
import cv2
import mediapipe as mp
import numpy as np

class FaceDetectorMP:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Inisialisasi detektor wajah MediaPipe Face Detection.
        
        Args:
            min_detection_confidence (float): Threshold confidence minimum untuk deteksi wajah.
            model_selection (int): Pilih model deteksi wajah, 
                                   0 untuk jarak pendek (~2m), 1 untuk jangkauan penuh (~5m).
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection 
        )

    def detect_face_bounding_box(self, frame_rgb):
        """
        Mendeteksi wajah utama dalam frame RGB dan mengembalikan bounding box.

        Args:
            frame_rgb (np.array): Frame gambar dalam format RGB.

        Returns:
            tuple: (x, y, w, h) koordinat bounding box wajah, atau None jika tidak ada wajah terdeteksi.
        """
        results = self.face_detection.process(frame_rgb)
        if results.detections:
            # Ambil deteksi wajah pertama (biasanya yang paling pasti)
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame_rgb.shape

            # Konversi koordinat relatif ke pixel absolut
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Pastikan koordinat tidak keluar dari frame
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            # Validasi ukuran bounding box sebelum mengembalikan
            if w > 0 and h > 0:
                return x, y, w, h
        return None

    def close(self):
        """Melepaskan resource model deteksi wajah MediaPipe."""
        if self.face_detection:
            self.face_detection.close()

def get_roi_pixels(frame, roi_coords_tuple):
    """
    Ekstrak piksel ROI dari frame berdasarkan koordinat bounding box.

    Args:
        frame (np.array): Frame gambar input.
        roi_coords_tuple (tuple): Koordinat ROI dalam format (x, y, w, h).

    Returns:
        np.array: Piksel pada ROI, atau array kosong jika ROI tidak valid.
    """
    if roi_coords_tuple is None:
        return np.array([])  # Return kosong jika ROI tidak ada

    x, y, w, h = roi_coords_tuple
    
    # Validasi ukuran ROI minimal
    if w <= 0 or h <= 0:
        return np.array([])

    frame_h, frame_w = frame.shape[:2]

    # Pastikan ROI tidak keluar dari batas frame
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(x_start + w, frame_w)
    y_end = min(y_start + h, frame_h)

    # Jika ROI invalid setelah pembatasan
    if x_end <= x_start or y_end <= y_start:
        return np.array([])
        
    return frame[y_start:y_end, x_start:x_end]

# Catatan:
# Kode deteksi wajah lama dengan Haar Cascade dicomment karena sudah diganti MediaPipe yang lebih akurat dan cepat.
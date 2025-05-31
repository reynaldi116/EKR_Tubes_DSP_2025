# pose_respiration_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import collections  # Untuk struktur data deque yang efisien, digunakan untuk smoothing

class PoseRespirationTracker:
    def __init__(self, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5, 
                 model_complexity=1,
                 raw_signal_multiplier=250.0,  # Faktor skala sinyal pernapasan mentah
                 internal_smoothing_window=1):  # Ukuran window smoothing internal (1=tanpa smoothing)
        """
        Inisialisasi pelacak pernapasan berbasis MediaPipe Pose.
        Parameter model dan smoothing dapat disesuaikan.

        Args:
            min_detection_confidence (float): Threshold keyakinan deteksi pose.
            min_tracking_confidence (float): Threshold keyakinan pelacakan pose.
            model_complexity (int): Kompleksitas model pose (0=cepat, 2=akurat, 1=kompromi).
            raw_signal_multiplier (float): Skala pengali untuk sinyal dy mentah.
            internal_smoothing_window (int): Window untuk rata-rata bergerak smoothing internal.
        """
        # Setup MediaPipe Pose dengan parameter yang diberikan
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        self.mp_drawing = mp.solutions.drawing_utils  # Utilitas untuk menggambar landmark pada frame
        
        self.prev_shoulder_y_mid = None  # Posisi vertikal tengah bahu frame sebelumnya
        self.raw_signal_multiplier = float(raw_signal_multiplier)  # Pastikan multiplier bertipe float
        
        # Setup deque untuk smoothing sinyal dy internal, hanya aktif jika window > 1
        self.internal_smoothing_window = max(1, int(internal_smoothing_window))
        if self.internal_smoothing_window > 1:
            self.dy_history = collections.deque(maxlen=self.internal_smoothing_window)
        else:
            self.dy_history = None  # Nonaktifkan smoothing jika window=1

    def get_respiration_signal_and_draw_landmarks(self, frame_rgb, frame_to_draw_on=None):
        """
        Proses frame RGB untuk ekstrak sinyal pernapasan dari gerakan bahu,
        dan gambar landmark pose jika frame tujuan disediakan.

        Args:
            frame_rgb (np.array): Frame input dalam format RGB.
            frame_to_draw_on (np.array, optional): Frame BGR untuk menggambar landmark.
                                                    Jika None, tidak menggambar.

        Returns:
            tuple:
              raw_respiration_signal (float): Nilai sinyal pernapasan mentah (dy * multiplier).
              pose_detected_flag (bool): True jika pose berhasil dideteksi, False jika tidak.
        """
        results = self.pose.process(frame_rgb)  # Jalankan deteksi pose MediaPipe
        raw_signal = 0.0
        pose_detected = False
        dy = 0.0  # Perubahan posisi vertikal bahu antar frame

        if results.pose_landmarks:  # Jika pose berhasil dideteksi
            pose_detected = True
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Ambil landmark bahu kiri dan kanan
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # Cek visibilitas landmark (agar hanya pakai yang cukup jelas)
                if left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3:
                    # Hitung posisi tengah vertikal bahu
                    current_shoulder_y_mid = (left_shoulder.y + right_shoulder.y) / 2.0
                    
                    # Jika sudah ada posisi sebelumnya, hitung perubahan posisi (dy)
                    if self.prev_shoulder_y_mid is not None:
                        dy = self.prev_shoulder_y_mid - current_shoulder_y_mid
                    
                    # Update posisi tengah bahu untuk frame berikutnya
                    self.prev_shoulder_y_mid = current_shoulder_y_mid
                else:
                    # Jika bahu kurang terlihat, reset posisi sebelumnya dan bersihkan smoothing
                    self.prev_shoulder_y_mid = None
                    if self.dy_history:
                        self.dy_history.clear()

            except (IndexError, AttributeError):
                # Jika landmark tidak ditemukan atau error lain, reset semua
                self.prev_shoulder_y_mid = None
                if self.dy_history:
                    self.dy_history.clear()
                pose_detected = False

            # Gambar landmark pada frame tujuan jika disediakan dan pose valid
            if frame_to_draw_on is not None and results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_to_draw_on,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                )
        else:
            # Jika pose tidak terdeteksi, reset posisi dan smoothing
            self.prev_shoulder_y_mid = None
            if self.dy_history:
                self.dy_history.clear()

        # Terapkan smoothing internal (jika aktif)
        if self.dy_history is not None:
            # Tambahkan dy ke history hanya jika dy valid atau jika history kosong (awal)
            if self.prev_shoulder_y_mid is not None or not self.dy_history:
                self.dy_history.append(dy)

            if self.dy_history:
                averaged_dy = np.mean(list(self.dy_history))
                raw_signal = averaged_dy * self.raw_signal_multiplier
            else:
                raw_signal = dy * self.raw_signal_multiplier
        else:
            # Tanpa smoothing, langsung kalikan dy
            raw_signal = dy * self.raw_signal_multiplier
            
        return raw_signal, pose_detected

    def close(self):
        """Melepaskan resource model MediaPipe Pose saat aplikasi selesai."""
        if self.pose:
            self.pose.close()

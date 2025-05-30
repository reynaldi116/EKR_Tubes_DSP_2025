# pose_respiration_tracker.py
import cv2
import mediapipe as mp
import numpy as np

class PoseRespirationTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        """
        Inisialisasi pelacak pernapasan menggunakan MediaPipe Pose.

        Args:
            min_detection_confidence (float): Keyakinan minimum untuk deteksi pose.
            min_tracking_confidence (float): Keyakinan minimum untuk pelacakan pose.
            model_complexity (int): Kompleksitas model pose (0, 1, atau 2). 
                                    0 lebih cepat, 2 lebih akurat. 1 adalah kompromi baik.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        self.mp_drawing = mp.solutions.drawing_utils # Untuk menggambar landmark (opsional)
        
        self.prev_shoulder_y_mid = None # Menyimpan posisi y tengah bahu sebelumnya

    def get_respiration_signal_and_draw_landmarks(self, frame_rgb, frame_to_draw_on=None):
        """
        Memproses frame RGB untuk mendapatkan sinyal pernapasan dari gerakan bahu 
        dan (opsional) menggambar landmark pose.

        Args:
            frame_rgb (np.array): Frame input dalam format RGB.
            frame_to_draw_on (np.array, opsional): Frame (BGR) tempat landmark akan digambar.
                                                   Jika None, tidak ada penggambaran.

        Returns:
            tuple: (raw_respiration_signal, pose_detected_flag)
                   raw_respiration_signal (float): Nilai sinyal pernapasan mentah.
                   pose_detected_flag (bool): True jika pose terdeteksi, False jika tidak.
        """
        results = self.pose.process(frame_rgb)
        raw_signal = 0.0
        pose_detected = False

        if results.pose_landmarks:
            pose_detected = True
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Dapatkan landmark bahu kiri dan kanan
                # Landmark ini dinormalisasi (0.0 - 1.0) relatif terhadap dimensi gambar
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # Periksa visibilitas (opsional, tapi baik untuk robustnes)
                if left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3: # Ambang batas visibilitas
                    # Ambil rata-rata koordinat y dari kedua bahu
                    current_shoulder_y_mid = (left_shoulder.y + right_shoulder.y) / 2.0
                    
                    if self.prev_shoulder_y_mid is not None:
                        # Sinyal adalah perubahan posisi y.
                        # Koordinat y MediaPipe: 0 di atas, 1 di bawah.
                        # Jika bahu naik (menghirup), y menurun. Maka dy = prev_y - current_y akan positif.
                        dy = self.prev_shoulder_y_mid - current_shoulder_y_mid
                        # Kita bisa mengalikan dengan skalar untuk memperbesar amplitudo jika perlu
                        raw_signal = dy * 1000 # Skala bisa disesuaikan
                    
                    self.prev_shoulder_y_mid = current_shoulder_y_mid
                else: # Jika bahu tidak cukup terlihat
                    self.prev_shoulder_y_mid = None # Reset
            except (IndexError, AttributeError):
                # Jika landmark tidak ditemukan atau error lain
                self.prev_shoulder_y_mid = None # Reset
                pose_detected = False # Anggap pose tidak valid jika landmark penting hilang

            # Gambar landmark pada frame_to_draw_on jika disediakan
            if frame_to_draw_on is not None:
                self.mp_drawing.draw_landmarks(
                    frame_to_draw_on,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                )
        else: # Tidak ada pose terdeteksi
            self.prev_shoulder_y_mid = None # Reset

        return raw_signal, pose_detected

    def close(self):
        """Melepaskan resource model pose."""
        if self.pose:
            self.pose.close()
# pose_respiration_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import collections # Untuk opsi smoothing internal

class PoseRespirationTracker:
    def __init__(self, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5, 
                 model_complexity=1,
                 raw_signal_multiplier=250.0, # Nilai default yang lebih konservatif
                 internal_smoothing_window=1): # Ukuran window untuk smoothing internal (1 = tanpa smoothing)
        """
        Inisialisasi pelacak pernapasan menggunakan MediaPipe Pose.

        Args:
            min_detection_confidence (float): Keyakinan minimum untuk deteksi pose.
            min_tracking_confidence (float): Keyakinan minimum untuk pelacakan pose.
            model_complexity (int): Kompleksitas model pose (0, 1, atau 2). 
                                    0 lebih cepat, 2 lebih akurat. 1 adalah kompromi baik.
            raw_signal_multiplier (float): Faktor pengali untuk sinyal mentah dy.
            internal_smoothing_window (int): Ukuran window untuk rata-rata bergerak pada dy. 
                                             Setel ke 1 untuk menonaktifkan smoothing internal.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        self.mp_drawing = mp.solutions.drawing_utils # Untuk menggambar landmark (opsional)
        
        self.prev_shoulder_y_mid = None # Menyimpan posisi y tengah bahu sebelumnya
        self.raw_signal_multiplier = float(raw_signal_multiplier) # Pastikan float
        
        # Untuk smoothing internal (opsional)
        self.internal_smoothing_window = max(1, int(internal_smoothing_window)) # Minimal 1
        if self.internal_smoothing_window > 1:
            self.dy_history = collections.deque(maxlen=self.internal_smoothing_window)
        else:
            self.dy_history = None


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
        dy = 0.0 # Inisialisasi dy

        if results.pose_landmarks:
            pose_detected = True
            landmarks = results.pose_landmarks.landmark
            
            try:
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                if left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3: # Ambang batas visibilitas
                    current_shoulder_y_mid = (left_shoulder.y + right_shoulder.y) / 2.0
                    
                    if self.prev_shoulder_y_mid is not None:
                        dy = self.prev_shoulder_y_mid - current_shoulder_y_mid # Hitung dy
                    
                    # Update posisi bahu sebelumnya setelah semua perhitungan dy selesai untuk frame ini
                    self.prev_shoulder_y_mid = current_shoulder_y_mid
                else: # Jika bahu tidak cukup terlihat
                    self.prev_shoulder_y_mid = None # Reset, dy akan tetap 0 atau dari frame sebelumnya jika smoothing
                    if self.dy_history: # Jika smoothing aktif, bersihkan history dy agar tidak ada diskontinuitas besar
                        self.dy_history.clear()

            except (IndexError, AttributeError):
                # Jika landmark tidak ditemukan atau error lain
                self.prev_shoulder_y_mid = None # Reset
                if self.dy_history:
                    self.dy_history.clear()
                pose_detected = False # Anggap pose tidak valid jika landmark penting hilang

            # Gambar landmark jika perlu
            if frame_to_draw_on is not None and results.pose_landmarks: # Pastikan results.pose_landmarks ada
                self.mp_drawing.draw_landmarks(
                    frame_to_draw_on,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                )
        else: # Tidak ada pose terdeteksi sama sekali
            self.prev_shoulder_y_mid = None # Reset
            if self.dy_history: # Jika smoothing aktif, bersihkan history dy
                self.dy_history.clear()

        # Terapkan smoothing internal jika diaktifkan
        if self.dy_history is not None:
            # Hanya tambahkan dy jika prev_shoulder_y_mid BUKAN None PADA frame ini 
            # (artinya dy valid dihitung) ATAU jika history masih kosong (awal)
            # Ini mencegah dy=0.0 dari 'landmark hilang' mendominasi rata-rata secara tidak benar
            if self.prev_shoulder_y_mid is not None or not self.dy_history : 
                 self.dy_history.append(dy)

            if self.dy_history: # Pastikan tidak kosong
                averaged_dy = np.mean(list(self.dy_history))
                raw_signal = averaged_dy * self.raw_signal_multiplier
            else: # Fallback jika history kosong (seharusnya tidak terjadi jika append benar)
                raw_signal = dy * self.raw_signal_multiplier

        else: # Tanpa smoothing internal
            raw_signal = dy * self.raw_signal_multiplier
            
        return raw_signal, pose_detected

    def close(self):
        """Melepaskan resource model pose."""
        if self.pose:
            self.pose.close()
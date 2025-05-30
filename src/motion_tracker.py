# motion_tracker.py
import cv2
import numpy as np

class RespirationMotionTracker:
    def __init__(self, max_features=30, quality_level=0.2, min_distance=5, block_size=5):
        """
        Inisialisasi pelacak gerakan untuk estimasi sinyal pernapasan.

        Args:
            max_features (int): Jumlah maksimum fitur yang akan dilacak.
            quality_level (float): Tingkat kualitas minimal untuk deteksi fitur (0-1).
            min_distance (int): Jarak minimal antar fitur yang terdeteksi.
            block_size (int): Ukuran blok untuk perhitungan matriks turunan sudut.
        """
        self.prev_gray_roi = None
        self.prev_points_roi = None  # Titik fitur dalam koordinat ROI

        # Parameter untuk cv2.goodFeaturesToTrack
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size
        )

        # Parameter untuk cv2.calcOpticalFlowPyrLK (Lucas-Kanade optical flow)
        self.lk_params = dict(
            winSize=(15, 15),  # Ukuran window pencarian di setiap level piramida
            maxLevel=2,        # Jumlah maksimum level piramida
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.retrack_threshold = max_features // 3 # Jumlah minimum fitur sebelum redeteksi

    def get_motion_signal(self, frame_bgr, roi_coords_dict):
        """
        Menghitung sinyal gerakan (perpindahan vertikal rata-rata) dari ROI.

        Args:
            frame_bgr (np.array): Frame video input dalam format BGR.
            roi_coords_dict (dict): Dictionary {'x': x, 'y': y, 'w': w, 'h': h} 
                                    untuk koordinat ROI.

        Returns:
            float: Nilai sinyal gerakan. Mengembalikan 0.0 jika tidak ada cukup fitur
                   atau gerakan yang valid terdeteksi.
        """
        x, y, w, h = roi_coords_dict['x'], roi_coords_dict['y'], roi_coords_dict['w'], roi_coords_dict['h']

        # Pastikan ROI berada dalam batas frame
        frame_h_orig, frame_w_orig = frame_bgr.shape[:2]
        x = max(0, min(x, frame_w_orig - 1))
        y = max(0, min(y, frame_h_orig - 1))
        w = max(1, min(w, frame_w_orig - x)) # Pastikan lebar minimal 1
        h = max(1, min(h, frame_h_orig - y)) # Pastikan tinggi minimal 1

        if w <= 0 or h <= 0: # Jika ROI tidak valid
            self.prev_gray_roi = None
            self.prev_points_roi = None
            return 0.0

        roi_frame = frame_bgr[y:y+h, x:x+w]
        current_gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        motion_signal = 0.0

        # Jika tidak ada titik sebelumnya atau jumlah titik terlalu sedikit, deteksi ulang
        if self.prev_points_roi is None or len(self.prev_points_roi) < self.retrack_threshold:
            self.prev_points_roi = cv2.goodFeaturesToTrack(current_gray_roi, mask=None, **self.feature_params)
            self.prev_gray_roi = current_gray_roi.copy()
            # Jika tidak ada fitur terdeteksi, kembalikan 0
            if self.prev_points_roi is None:
                return 0.0
            # Belum ada gerakan yang bisa dihitung pada frame pertama deteksi fitur
            return 0.0

        # Jika ada frame dan titik sebelumnya, hitung optical flow
        if self.prev_gray_roi is not None and self.prev_points_roi is not None:
            # Hitung optical flow
            # p1 adalah posisi baru dari fitur, st adalah status (1 jika terlacak)
            p1_roi, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray_roi, current_gray_roi, self.prev_points_roi, None, **self.lk_params
            )

            if p1_roi is not None and st is not None:
                good_new_roi = p1_roi[st == 1]    # Titik baru yang berhasil dilacak
                good_old_roi = self.prev_points_roi[st == 1] # Titik lama yang sesuai

                # Perlu setidaknya beberapa titik untuk rata-rata yang stabil
                if len(good_new_roi) >= self.retrack_threshold // 2 and len(good_new_roi) > 3:
                    # Hitung rata-rata perpindahan vertikal (sumbu y)
                    # dy positif berarti gerakan ke bawah, negatif berarti ke atas (tergantung orientasi gambar)
                    dy = good_new_roi[:, 1] - good_old_roi[:, 1]
                    motion_signal = np.mean(dy)
                    
                    # Update titik untuk frame berikutnya
                    self.prev_points_roi = good_new_roi.reshape(-1, 1, 2)
                else:
                    # Tidak cukup titik yang terlacak, paksa redeteksi pada frame berikutnya
                    self.prev_points_roi = None 
            else:
                # Optical flow gagal, paksa redeteksi
                self.prev_points_roi = None
        
        # Simpan frame abu-abu dan titik untuk iterasi berikutnya
        self.prev_gray_roi = current_gray_roi.copy()

        return motion_signal if not np.isnan(motion_signal) else 0.0
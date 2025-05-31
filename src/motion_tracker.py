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
        self.prev_gray_roi = None  # Frame abu-abu sebelumnya pada ROI untuk optical flow
        self.prev_points_roi = None  # Titik fitur sebelumnya dalam ROI (koordinat pixel)

        # Parameter untuk deteksi fitur sudut terbaik (Good Features to Track)
        self.feature_params = dict(
            maxCorners=max_features,         # Maksimal jumlah fitur sudut yang diambil
            qualityLevel=quality_level,      # Kualitas minimal fitur, 0 sampai 1
            minDistance=min_distance,        # Jarak minimal antar fitur agar tersebar merata
            blockSize=block_size             # Ukuran blok yang dipakai untuk menghitung matriks turunan
        )

        # Parameter untuk optical flow Lucas-Kanade (estimasi pergerakan titik fitur)
        self.lk_params = dict(
            winSize=(15, 15),  # Ukuran jendela pencarian di tiap level piramida
            maxLevel=2,        # Maksimal level piramida untuk optical flow
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  
            # Kriteria penghentian iterasi: maksimal 10 iterasi atau perubahan error < 0.03
        )
        
        # Threshold jumlah fitur minimum sebelum kita redeteksi fitur baru
        self.retrack_threshold = max_features // 3  

    def get_motion_signal(self, frame_bgr, roi_coords_dict):
        """
        Hitung sinyal gerakan vertikal rata-rata dari ROI pada frame input.

        Args:
            frame_bgr (np.array): Frame video input dalam format BGR.
            roi_coords_dict (dict): {'x': x, 'y': y, 'w': w, 'h': h} koordinat ROI.

        Returns:
            float: Sinyal gerakan vertikal rata-rata (perpindahan y) dari fitur yang dilacak.
                   Mengembalikan 0.0 jika fitur tidak cukup atau gerakan tidak valid.
        """
        # Ambil koordinat ROI dari dictionary
        x, y, w, h = roi_coords_dict['x'], roi_coords_dict['y'], roi_coords_dict['w'], roi_coords_dict['h']

        # Pastikan ROI tidak keluar dari batas frame asli
        frame_h_orig, frame_w_orig = frame_bgr.shape[:2]
        x = max(0, min(x, frame_w_orig - 1))
        y = max(0, min(y, frame_h_orig - 1))
        w = max(1, min(w, frame_w_orig - x))  # Lebar minimal 1 pixel
        h = max(1, min(h, frame_h_orig - y))  # Tinggi minimal 1 pixel

        # Jika ROI tidak valid, reset data sebelumnya dan kembalikan sinyal 0
        if w <= 0 or h <= 0:
            self.prev_gray_roi = None
            self.prev_points_roi = None
            return 0.0

        # Crop ROI dari frame dan konversi ke grayscale untuk optical flow
        roi_frame = frame_bgr[y:y+h, x:x+w]
        current_gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        motion_signal = 0.0

        # Jika tidak ada fitur sebelumnya atau jumlah fitur kurang dari threshold,
        # lakukan deteksi ulang fitur (goodFeaturesToTrack)
        if self.prev_points_roi is None or len(self.prev_points_roi) < self.retrack_threshold:
            self.prev_points_roi = cv2.goodFeaturesToTrack(current_gray_roi, mask=None, **self.feature_params)
            self.prev_gray_roi = current_gray_roi.copy()
            # Jika fitur tidak ditemukan, return 0 (tidak ada sinyal gerak)
            if self.prev_points_roi is None:
                return 0.0
            # Frame pertama deteksi fitur belum bisa dihitung gerakannya
            return 0.0

        # Jika ada frame dan titik fitur sebelumnya, hitung optical flow Lucas-Kanade
        if self.prev_gray_roi is not None and self.prev_points_roi is not None:
            p1_roi, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray_roi, current_gray_roi, self.prev_points_roi, None, **self.lk_params
            )

            # Cek apakah optical flow berhasil dilacak
            if p1_roi is not None and st is not None:
                # Ambil fitur yang berhasil dilacak (status=1)
                good_new_roi = p1_roi[st == 1]
                good_old_roi = self.prev_points_roi[st == 1]

                # Jika cukup fitur terlacak, hitung perpindahan vertikal rata-rata
                if len(good_new_roi) >= self.retrack_threshold // 2 and len(good_new_roi) > 3:
                    # dy = perpindahan vertikal (y) antara frame sekarang dan sebelumnya
                    dy = good_new_roi[:, 1] - good_old_roi[:, 1]
                    motion_signal = np.mean(dy)
                    
                    # Update titik fitur untuk iterasi berikutnya
                    self.prev_points_roi = good_new_roi.reshape(-1, 1, 2)
                else:
                    # Jika fitur kurang, reset supaya fitur dideteksi ulang di frame berikutnya
                    self.prev_points_roi = None
            else:
                # Optical flow gagal, reset fitur supaya dideteksi ulang
                self.prev_points_roi = None
        
        # Simpan frame grayscale ROI untuk iterasi berikutnya
        self.prev_gray_roi = current_gray_roi.copy()

        # Kembalikan sinyal gerakan vertikal rata-rata
        # Pastikan bukan NaN, jika NaN kembalikan 0.0
        return motion_signal if not np.isnan(motion_signal) else 0.0

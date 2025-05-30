# signal_processing.py
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft

# --- Parameter Filter ---
# rPPG (Detak Jantung: umumnya 0.75 - 4 Hz atau 45 - 240 BPM)
RPPG_LOWCUT = 0.75
RPPG_HIGHCUT = 4.0

# Respirasi (Pernapasan: umumnya 0.1 - 0.5 Hz atau 6 - 30 BPM)
RESP_LOWCUT = 0.1
RESP_HIGHCUT = 0.5

# Ukuran buffer untuk sinyal mentah sebelum filtering dan analisis FFT
SIGNAL_BUFFER_SIZE = 256 # Sesuaikan berdasarkan FPS dan kebutuhan analisis frekuensi

class SignalProcessor:
    def __init__(self, fs, buffer_size=SIGNAL_BUFFER_SIZE):
        """
        Inisialisasi pemroses sinyal.
        Args:
            fs (float): Frekuensi sampling (FPS kamera).
            buffer_size (int): Ukuran buffer sinyal.
        """
        self.fs = fs
        self.buffer_size = buffer_size
        self.rppg_raw_signal = []
        self.resp_raw_signal = [] # Bisa dari gerakan atau modulasi rPPG

    def _butter_bandpass_filter(self, data, lowcut, highcut, order=5):
        """
        Menerapkan filter bandpass Butterworth.
        Args:
            data (list or np.array): Sinyal input.
            lowcut (float): Frekuensi cutoff bawah.
            highcut (float): Frekuensi cutoff atas.
            order (int): Orde filter.
        Returns:
            np.array: Sinyal terfilter.
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        # Pastikan low dan high valid
        if low <= 0: low = 0.001
        if high >= 1: high = 0.999
        if low >= high: # Jika frekuensi tidak valid, kembalikan data asli
            print(f"Peringatan: Rentang frekuensi filter tidak valid ({lowcut}-{highcut} Hz). Mengembalikan data asli.")
            return np.array(data)
        try:
            b, a = butter(order, [low, high], btype='band')
            y = filtfilt(b, a, data)
            return y
        except ValueError as e:
            print(f"Error saat filtering: {e}. Mengembalikan data asli.")
            return np.array(data)


    def process_rppg(self, roi_pixels_green_channel_mean):
        """
        Memproses sinyal rPPG mentah, filter, dan estimasi BPM.
        Args:
            roi_pixels_green_channel_mean (float): Nilai rata-rata channel hijau dari ROI.
        Returns:
            tuple: (sinyal_rppg_terfilter, bpm_estimasi)
        """
        self.rppg_raw_signal.append(roi_pixels_green_channel_mean)
        if len(self.rppg_raw_signal) > self.buffer_size:
            self.rppg_raw_signal.pop(0) # Jaga ukuran buffer

        if len(self.rppg_raw_signal) < self.buffer_size: # Butuh cukup data untuk filter & FFT
            return np.array([]), 0.0

        # Detrending sederhana (opsional, bisa kompleks)
        signal_detrended = np.array(self.rppg_raw_signal) - np.mean(self.rppg_raw_signal)

        # Filtering
        filtered_rppg = self._butter_bandpass_filter(signal_detrended, RPPG_LOWCUT, RPPG_HIGHCUT)

        # Estimasi BPM menggunakan FFT (Fast Fourier Transform)
        # Kita memerlukan sinyal yang cukup panjang untuk resolusi frekuensi yang baik
        if len(filtered_rppg) < self.buffer_size // 2 : # Minimal setengah buffer untuk FFT
             return filtered_rppg, 0.0

        N = len(filtered_rppg)
        yf = fft(filtered_rppg)
        xf = np.linspace(0.0, 1.0/(2.0*(1/self.fs)), N//2) # Frekuensi

        # Cari frekuensi dominan dalam rentang rPPG
        freq_indices = np.where((xf >= RPPG_LOWCUT) & (xf <= RPPG_HIGHCUT))[0]
        if len(freq_indices) == 0:
            return filtered_rppg, 0.0

        dominant_freq_idx = freq_indices[np.argmax(np.abs(yf[freq_indices]))]
        dominant_freq = xf[dominant_freq_idx]
        bpm = dominant_freq * 60

        return filtered_rppg, bpm

    def process_respiration(self, roi_movement_metric): # Atau bisa dari analisis sinyal rPPG
        """
        Memproses sinyal respirasi mentah, filter, dan estimasi RPM (Respiration Per Minute).
        Args:
            roi_movement_metric (float): Metrik gerakan dari ROI (misalnya, perubahan posisi centroid ROI dada).
                                         Atau bisa juga menggunakan modulasi dari sinyal rPPG.
        Returns:
            tuple: (sinyal_respirasi_terfilter, rpm_estimasi)
        """
        self.resp_raw_signal.append(roi_movement_metric)
        if len(self.resp_raw_signal) > self.buffer_size:
            self.resp_raw_signal.pop(0)

        if len(self.resp_raw_signal) < self.buffer_size:
            return np.array([]), 0.0

        signal_detrended = np.array(self.resp_raw_signal) - np.mean(self.resp_raw_signal)
        filtered_resp = self._butter_bandpass_filter(signal_detrended, RESP_LOWCUT, RESP_HIGHCUT, order=3) # Orde lebih rendah mungkin cukup

        if len(filtered_resp) < self.buffer_size // 2:
             return filtered_resp, 0.0

        N = len(filtered_resp)
        yf = fft(filtered_resp)
        xf = np.linspace(0.0, 1.0/(2.0*(1/self.fs)), N//2)

        freq_indices = np.where((xf >= RESP_LOWCUT) & (xf <= RESP_HIGHCUT))[0]
        if len(freq_indices) == 0:
            return filtered_resp, 0.0

        dominant_freq_idx = freq_indices[np.argmax(np.abs(yf[freq_indices]))]
        dominant_freq = xf[dominant_freq_idx]
        rpm = dominant_freq * 60

        return filtered_resp, rpm

    def get_raw_rppg_signal_for_plot(self):
        return np.array(self.rppg_raw_signal)

    def get_raw_resp_signal_for_plot(self):
        return np.array(self.resp_raw_signal)
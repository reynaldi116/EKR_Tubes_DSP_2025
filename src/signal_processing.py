# signal_processing.py
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d # <<<--- IMPORT BARU untuk moving average

# --- Parameter Filter ---
# rPPG (Detak Jantung: umumnya 0.75 - 4 Hz atau 45 - 240 BPM)
RPPG_LOWCUT = 0.75
RPPG_HIGHCUT = 4.0
RPPG_FILTER_ORDER = 5

# Respirasi (Pernapasan: umumnya 0.1 - 0.5 Hz atau 6 - 30 BPM)
RESP_LOWCUT = 0.1
RESP_HIGHCUT = 0.8 # Sedikit dinaikkan batas atasnya, bisa sampai 0.8Hz (48 RPM) untuk aktivitas ringan
RESP_FILTER_ORDER = 2 # Bisa juga 5 jika sinyal sangat noisy

# Ukuran buffer untuk sinyal mentah sebelum filtering dan analisis FFT
# Dinaikkan untuk stabilitas yang lebih baik (misal, untuk ~10-15 detik data pada 30 FPS)
# Default sebelumnya 256 (~8.5s @ 30fps)
SIGNAL_BUFFER_SIZE = 384 # Sekarang ~12.8 detik @ 30fps. Anda bisa coba 450 atau 512 juga.

class SignalProcessor:
    def __init__(self, fs, buffer_size=SIGNAL_BUFFER_SIZE):
        """
        Inisialisasi pemroses sinyal.
        Args:
            fs (float): Frekuensi sampling (FPS kamera).
            buffer_size (int): Ukuran buffer sinyal.
        """
        if fs <= 0:
            print(f"Peringatan: Frekuensi sampling (fs) tidak valid: {fs}. Menggunakan fs=30.0 sebagai default.")
            fs = 30.0 # Fallback jika FPS tidak valid
        self.fs = fs
        self.buffer_size = buffer_size
        self.rppg_raw_signal = []
        self.resp_raw_signal = []

    def _butter_bandpass_filter(self, data, lowcut, highcut, order):
        """
        Menerapkan filter bandpass Butterworth.
        Args:
            data (list or np.array): Sinyal input.
            lowcut (float): Frekuensi cutoff bawah.
            highcut (float): Frekuensi cutoff atas.
            order (int): Orde filter.
        Returns:
            np.array: Sinyal terfilter, atau array kosong jika error.
        """
        if len(data) < order * 3: # Data harus cukup panjang untuk filter
            # print(f"Peringatan filter: data tidak cukup panjang ({len(data)}) untuk orde filter ({order}).")
            return np.array([]) # Kembalikan array kosong jika data tidak cukup

        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Validasi batas frekuensi
        if not (0 < low < 1 and 0 < high < 1 and low < high):
            print(f"Peringatan filter: Batas frekuensi tidak valid setelah normalisasi (low: {low}, high: {high}).")
            return np.array(data) # Kembalikan data asli jika batas tidak valid

        try:
            b, a = butter(order, [low, high], btype='band')
            y = filtfilt(b, a, data) # filtfilt untuk zero-phase filtering
            return y
        except ValueError as e:
            print(f"Error saat filtering ({lowcut}-{highcut} Hz, order {order}): {e}. Mengembalikan data asli.")
            return np.array(data) # Kembalikan data asli jika ada error lain

    def _detrend_with_moving_average(self, signal_segment, window_seconds):
        """
        Menghilangkan tren dari segmen sinyal menggunakan moving average.
        Args:
            signal_segment (np.array): Segmen sinyal input.
            window_seconds (float): Ukuran window moving average dalam detik.
        Returns:
            np.array: Sinyal yang sudah di-detrend.
        """
        if len(signal_segment) == 0:
            return np.array([])

        window_samples = int(self.fs * window_seconds)
        if window_samples < 3: # Window MA minimal 3 sampel
            window_samples = 3
        
        if len(signal_segment) >= window_samples:
            # uniform_filter1d lebih efisien dan menangani tepi dengan baik (mode 'reflect')
            moving_avg = uniform_filter1d(signal_segment, size=window_samples, mode='reflect')
            detrended_signal = signal_segment - moving_avg
        else:
            # Fallback ke simple mean removal jika data tidak cukup untuk window MA
            detrended_signal = signal_segment - np.mean(signal_segment)
        return detrended_signal

    def process_rppg(self, roi_pixels_green_channel_mean):
        """
        Memproses sinyal rPPG mentah, filter, dan estimasi BPM.
        """
        self.rppg_raw_signal.append(roi_pixels_green_channel_mean)
        if len(self.rppg_raw_signal) > self.buffer_size:
            self.rppg_raw_signal.pop(0)

        if len(self.rppg_raw_signal) < self.buffer_size:
            return np.array([]), 0.0 # Butuh buffer penuh untuk analisis stabil

        current_rppg_segment = np.array(self.rppg_raw_signal)

        # Detrending menggunakan Moving Average (window ~1.5-2 detik untuk rPPG)
        # Ini membantu menghilangkan variasi pencahayaan lambat atau gerakan kepala kecil
        detrended_rppg = self._detrend_with_moving_average(current_rppg_segment, window_seconds=2.0)

        # Filtering
        filtered_rppg = self._butter_bandpass_filter(detrended_rppg, RPPG_LOWCUT, RPPG_HIGHCUT, RPPG_FILTER_ORDER)
        if len(filtered_rppg) == 0: # Jika filtering gagal atau data tidak cukup
            return current_rppg_segment, 0.0 # Kembalikan sinyal mentah (atau detrended) agar plot tidak kosong

        # Estimasi BPM menggunakan FFT
        N = len(filtered_rppg)
        if N < self.fs: # Butuh setidaknya 1 detik data untuk FFT yang berarti
            return filtered_rppg, 0.0
        
        yf = fft(filtered_rppg)
        # Frekuensi dihitung hingga Nyquist (fs/2)
        # (1.0 / self.fs) adalah periode sampling (T)
        # N * (1.0 / self.fs) adalah total durasi sinyal
        # Frekuensi step = 1 / (Total Durasi Sinyal) = self.fs / N
        xf = np.fft.fftfreq(N, 1.0/self.fs)[:N//2] # Menggunakan np.fft.fftfreq untuk kemudahan

        # Cari frekuensi dominan dalam rentang rPPG
        # Pastikan xf positif karena fftfreq menghasilkan negatif juga
        valid_freq_indices = np.where((xf >= RPPG_LOWCUT) & (xf <= RPPG_HIGHCUT))[0]
        
        if len(valid_freq_indices) == 0:
            return filtered_rppg, 0.0 # Tidak ada komponen frekuensi di rentang yang diinginkan

        # Ambil magnitudo spektrum
        abs_yf = np.abs(yf[valid_freq_indices])
        if len(abs_yf) == 0: # Seharusnya tidak terjadi jika valid_freq_indices tidak kosong
            return filtered_rppg, 0.0

        dominant_freq_index_in_subset = np.argmax(abs_yf)
        dominant_freq = xf[valid_freq_indices[dominant_freq_index_in_subset]]
        
        bpm = dominant_freq * 60
        # Pembulatan sederhana untuk BPM agar tidak terlalu fluktuatif digit desimalnya
        bpm = round(bpm, 1) 

        return filtered_rppg, bpm

    def process_respiration(self, raw_motion_signal_value):
        """
        Memproses sinyal respirasi mentah (dari gerakan), filter, dan estimasi RPM.
        """
        self.resp_raw_signal.append(raw_motion_signal_value)
        if len(self.resp_raw_signal) > self.buffer_size:
            self.resp_raw_signal.pop(0)

        if len(self.resp_raw_signal) < self.buffer_size:
            return np.array([]), 0.0

        current_resp_segment = np.array(self.resp_raw_signal)

        # Detrending menggunakan Moving Average (window lebih panjang, misal ~8-10 detik untuk respirasi)
        # Ini sangat penting untuk sinyal gerakan yang bisa memiliki drift karena postur
        detrended_resp = self._detrend_with_moving_average(current_resp_segment, window_seconds=10.0)
        
        # Filtering
        filtered_resp = self._butter_bandpass_filter(detrended_resp, RESP_LOWCUT, RESP_HIGHCUT, RESP_FILTER_ORDER)
        if len(filtered_resp) == 0:
            return current_resp_segment, 0.0

        # Estimasi RPM menggunakan FFT
        N = len(filtered_resp)
        if N < self.fs * 2: # Butuh data yang cukup panjang untuk frekuensi rendah respirasi (misal 2x periode terendah)
            return filtered_resp, 0.0
            
        yf = fft(filtered_resp)
        xf = np.fft.fftfreq(N, 1.0/self.fs)[:N//2]

        valid_freq_indices = np.where((xf >= RESP_LOWCUT) & (xf <= RESP_HIGHCUT))[0]
        
        if len(valid_freq_indices) == 0:
            return filtered_resp, 0.0

        abs_yf = np.abs(yf[valid_freq_indices])
        if len(abs_yf) == 0:
            return filtered_resp, 0.0
            
        dominant_freq_index_in_subset = np.argmax(abs_yf)
        dominant_freq = xf[valid_freq_indices[dominant_freq_index_in_subset]]
        
        rpm = dominant_freq * 60
        rpm = round(rpm, 1)

        return filtered_resp, rpm

    def get_raw_rppg_signal_for_plot(self):
        return np.array(self.rppg_raw_signal)

    def get_raw_resp_signal_for_plot(self):
        return np.array(self.resp_raw_signal)
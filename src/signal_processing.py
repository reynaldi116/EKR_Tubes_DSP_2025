# signal_processing.py
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d  # Untuk moving average detrending yang efisien

# --- Parameter filter untuk detak jantung (rPPG) ---
# Rentang frekuensi normal detak jantung ~0.75 - 4 Hz (45 - 240 BPM)
RPPG_LOWCUT = 0.75
RPPG_HIGHCUT = 4.0
RPPG_FILTER_ORDER = 5

# --- Parameter filter untuk pernapasan (respirasi) ---
# Rentang frekuensi pernapasan ~0.1 - 0.8 Hz (6 - 48 RPM)
RESP_LOWCUT = 0.1
RESP_HIGHCUT = 0.8  # Batas atas dinaikkan untuk aktivitas ringan
RESP_FILTER_ORDER = 2  # Orde filter yang lebih rendah untuk noise rendah

# Ukuran buffer untuk simpan data sinyal sebelum filtering dan FFT
SIGNAL_BUFFER_SIZE = 384  # ~12.8 detik data @ 30 FPS, agar analisis stabil

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
            fs = 30.0  # Fallback jika FPS tidak valid
        self.fs = fs
        self.buffer_size = buffer_size
        self.rppg_raw_signal = []  # Buffer sinyal rPPG mentah (channel hijau)
        self.resp_raw_signal = []  # Buffer sinyal pernapasan mentah (gerakan)

    def _butter_bandpass_filter(self, data, lowcut, highcut, order):
        """
        Terapkan filter bandpass Butterworth pada data sinyal.

        Args:
            data (list/np.array): Data sinyal input.
            lowcut (float): Frekuensi cutoff bawah.
            highcut (float): Frekuensi cutoff atas.
            order (int): Orde filter Butterworth.

        Returns:
            np.array: Data sinyal terfilter, atau array kosong jika data tidak cukup.
        """
        if len(data) < order * 3:  # Cek data cukup panjang untuk filtering
            return np.array([])  # Return kosong jika tidak cukup

        nyq = 0.5 * self.fs  # Frekuensi Nyquist
        low = lowcut / nyq
        high = highcut / nyq
        
        # Validasi batas cutoff normalisasi
        if not (0 < low < 1 and 0 < high < 1 and low < high):
            print(f"Peringatan filter: Batas frekuensi tidak valid (low: {low}, high: {high}).")
            return np.array(data)  # Return data asli jika cutoff tidak valid

        try:
            b, a = butter(order, [low, high], btype='band')
            y = filtfilt(b, a, data)  # Zero-phase filtering
            return y
        except ValueError as e:
            print(f"Error saat filtering ({lowcut}-{highcut} Hz): {e}. Return data asli.")
            return np.array(data)

    def _detrend_with_moving_average(self, signal_segment, window_seconds):
        """
        Hapus tren lambat sinyal dengan moving average.

        Args:
            signal_segment (np.array): Segmen sinyal input.
            window_seconds (float): Ukuran window moving average dalam detik.

        Returns:
            np.array: Sinyal yang sudah dihilangkan tren lambatnya (detrended).
        """
        if len(signal_segment) == 0:
            return np.array([])

        window_samples = int(self.fs * window_seconds)
        if window_samples < 3:  # Minimal 3 sampel window
            window_samples = 3
        
        if len(signal_segment) >= window_samples:
            # Moving average efisien dengan uniform_filter1d dan mode refleksi tepi
            moving_avg = uniform_filter1d(signal_segment, size=window_samples, mode='reflect')
            detrended_signal = signal_segment - moving_avg
        else:
            # Jika data pendek, buang rata-rata sederhana
            detrended_signal = signal_segment - np.mean(signal_segment)
        return detrended_signal

    def process_rppg(self, roi_pixels_green_channel_mean):
        """
        Proses sinyal rPPG dari channel hijau ROI:
        simpan, detrend, filter, FFT untuk estimasi BPM.

        Args:
            roi_pixels_green_channel_mean (float): Rata-rata intensitas hijau ROI frame terbaru.

        Returns:
            tuple: (filtered_rppg (np.array), estimated_bpm (float))
        """
        self.rppg_raw_signal.append(roi_pixels_green_channel_mean)
        if len(self.rppg_raw_signal) > self.buffer_size:
            self.rppg_raw_signal.pop(0)

        if len(self.rppg_raw_signal) < self.buffer_size:
            return np.array([]), 0.0  # Buffer belum penuh

        current_rppg_segment = np.array(self.rppg_raw_signal)

        # Detrend sinyal dengan moving average ~2 detik window
        detrended_rppg = self._detrend_with_moving_average(current_rppg_segment, window_seconds=2.0)

        # Filter bandpass untuk rentang detak jantung
        filtered_rppg = self._butter_bandpass_filter(detrended_rppg, RPPG_LOWCUT, RPPG_HIGHCUT, RPPG_FILTER_ORDER)
        if len(filtered_rppg) == 0:
            return current_rppg_segment, 0.0  # Jika gagal filter, return sinyal mentah

        # FFT untuk estimasi frekuensi dominan => BPM
        N = len(filtered_rppg)
        if N < self.fs:  # Minimal 1 detik data untuk FFT
            return filtered_rppg, 0.0
        
        yf = fft(filtered_rppg)
        xf = np.fft.fftfreq(N, 1.0/self.fs)[:N//2]  # Frekuensi positif

        # Cari indeks frekuensi dalam rentang rPPG
        valid_freq_indices = np.where((xf >= RPPG_LOWCUT) & (xf <= RPPG_HIGHCUT))[0]
        
        if len(valid_freq_indices) == 0:
            return filtered_rppg, 0.0

        abs_yf = np.abs(yf[valid_freq_indices])
        if len(abs_yf) == 0:
            return filtered_rppg, 0.0

        dominant_freq_index_in_subset = np.argmax(abs_yf)
        dominant_freq = xf[valid_freq_indices[dominant_freq_index_in_subset]]

        bpm = dominant_freq * 60  # Konversi Hz ke BPM
        bpm = round(bpm, 1)

        return filtered_rppg, bpm

    def process_respiration(self, raw_motion_signal_value):
        """
        Proses sinyal pernapasan (gerakan):
        simpan, detrend dengan window lebih panjang, filter, FFT untuk RPM.

        Args:
            raw_motion_signal_value (float): Sinyal mentah pernapasan frame terbaru.

        Returns:
            tuple: (filtered_resp (np.array), estimated_rpm (float))
        """
        self.resp_raw_signal.append(raw_motion_signal_value)
        if len(self.resp_raw_signal) > self.buffer_size:
            self.resp_raw_signal.pop(0)

        if len(self.resp_raw_signal) < self.buffer_size:
            return np.array([]), 0.0

        current_resp_segment = np.array(self.resp_raw_signal)

        # Detrend dengan moving average ~10 detik window (drift postur dihilangkan)
        detrended_resp = self._detrend_with_moving_average(current_resp_segment, window_seconds=10.0)
        
        # Filter bandpass respirasi
        filtered_resp = self._butter_bandpass_filter(detrended_resp, RESP_LOWCUT, RESP_HIGHCUT, RESP_FILTER_ORDER)
        if len(filtered_resp) == 0:
            return current_resp_segment, 0.0

        # FFT untuk frekuensi dominan respirasi (RPM)
        N = len(filtered_resp)
        if N < self.fs * 2:  # Butuh cukup panjang untuk frekuensi rendah (2 periode)
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
        
        rpm = dominant_freq * 60  # Konversi Hz ke RPM
        rpm = round(rpm, 1)

        return filtered_resp, rpm

    def get_raw_rppg_signal_for_plot(self):
        # Return buffer sinyal rPPG mentah untuk plotting
        return np.array(self.rppg_raw_signal)

    def get_raw_resp_signal_for_plot(self):
        # Return buffer sinyal respirasi mentah untuk plotting
        return np.array(self.resp_raw_signal)

# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np

class RealtimePlotter:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        
        self.fig, self.axs = plt.subplots(3, 1, figsize=(6, 7)) 
        self.fig.suptitle("Sinyal Fisiologis & Mentah RGB")

        # Plot rPPG
        self.axs[0].set_title("Sinyal rPPG Terfilter")
        self.axs[0].set_xlabel("Sampel")
        self.axs[0].set_ylabel("Amplitudo")
        # Simpan referensi ke data yang diplot jika diperlukan untuk penyimpanan terpisah
        self.current_rppg_data = np.zeros(self.buffer_size)
        self.line_rppg, = self.axs[0].plot(self.current_rppg_data, color='purple', label='rPPG Terfilter')
        self.axs[0].grid(True)
        self.axs[0].legend(loc='upper right')

        # Plot Respirasi (Terfilter dan Mentah)
        self.axs[1].set_title("Sinyal Respirasi")
        self.axs[1].set_xlabel("Sampel")
        self.axs[1].set_ylabel("Amplitudo")
        self.current_resp_filtered_data = np.zeros(self.buffer_size)
        self.line_resp_filtered, = self.axs[1].plot(self.current_resp_filtered_data, color='orange', label='Respirasi Terfilter')
        
        self.resp_raw_plot_buffer = np.zeros(self.buffer_size) # Sudah ada dari sebelumnya
        self.line_resp_raw, = self.axs[1].plot(self.resp_raw_plot_buffer, color='cyan', label='Respirasi Mentah', linestyle=':')
        self.axs[1].grid(True)
        self.axs[1].legend(loc='upper right')

        # Plot Sinyal RGB Mentah
        self.axs[2].set_title("Sinyal RGB Mentah dari ROI Wajah")
        self.axs[2].set_xlabel("Sampel")
        self.axs[2].set_ylabel("Intensitas Rata-rata")
        self.r_raw_buffer = np.zeros(self.buffer_size) # Sudah ada
        self.g_raw_buffer = np.zeros(self.buffer_size) # Sudah ada
        self.b_raw_buffer = np.zeros(self.buffer_size) # Sudah ada
        self.line_r_raw, = self.axs[2].plot(self.r_raw_buffer, color='red', label='Merah (R)')
        self.line_g_raw, = self.axs[2].plot(self.g_raw_buffer, color='green', label='Hijau (G)')
        self.line_b_raw, = self.axs[2].plot(self.b_raw_buffer, color='blue', label='Biru (B)')
        self.axs[2].grid(True)
        self.axs[2].legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def get_figure(self):
        return self.fig

    # Metode untuk mendapatkan data buffer saat ini
    def get_current_plot_data(self):
        return {
            "rppg_filtered": self.current_rppg_data.copy(),
            "rgb_r": self.r_raw_buffer.copy(),
            "rgb_g": self.g_raw_buffer.copy(),
            "rgb_b": self.b_raw_buffer.copy(),
            "resp_filtered": self.current_resp_filtered_data.copy(),
            "resp_raw": self.resp_raw_plot_buffer.copy()
            
        }

    def update_plots(self, rppg_signal_filtered, resp_signal_filtered, 
                     r_raw_value=None, g_raw_value=None, b_raw_value=None,
                     resp_raw_value=None):
        plot_updated = False

        if rppg_signal_filtered is not None and len(rppg_signal_filtered) > 0:
            # self.current_rppg_data diisi ulang dengan data yang akan diplot
            self.current_rppg_data.fill(0) # Kosongkan dulu
            data_to_plot_rppg = rppg_signal_filtered[-self.buffer_size:]
            self.current_rppg_data[-len(data_to_plot_rppg):] = data_to_plot_rppg
            self.line_rppg.set_ydata(self.current_rppg_data)
            if len(data_to_plot_rppg) > 1:
                min_val, max_val = np.min(data_to_plot_rppg), np.max(data_to_plot_rppg)
                padding = 0.1 * max(abs(min_val), abs(max_val), 0.1)
                self.axs[0].set_ylim(min_val - padding, max_val + padding if max_val > min_val else max_val + 0.5)
            plot_updated = True

        min_resp_val_combined, max_resp_val_combined = float('inf'), float('-inf')
        
        if resp_signal_filtered is not None and len(resp_signal_filtered) > 0:
            # self.current_resp_filtered_data diisi ulang
            self.current_resp_filtered_data.fill(0)
            data_to_plot_resp_filtered = resp_signal_filtered[-self.buffer_size:]
            self.current_resp_filtered_data[-len(data_to_plot_resp_filtered):] = data_to_plot_resp_filtered
            self.line_resp_filtered.set_ydata(self.current_resp_filtered_data)
            if len(data_to_plot_resp_filtered) > 1:
                min_resp_val_combined = min(min_resp_val_combined, np.min(data_to_plot_resp_filtered))
                max_resp_val_combined = max(max_resp_val_combined, np.max(data_to_plot_resp_filtered))
            plot_updated = True
        
        if resp_raw_value is not None:
            self.resp_raw_plot_buffer = np.roll(self.resp_raw_plot_buffer, -1)
            self.resp_raw_plot_buffer[-1] = resp_raw_value
            self.line_resp_raw.set_ydata(self.resp_raw_plot_buffer)
            if np.any(self.resp_raw_plot_buffer):
                 min_resp_val_combined = min(min_resp_val_combined, np.min(self.resp_raw_plot_buffer))
                 max_resp_val_combined = max(max_resp_val_combined, np.max(self.resp_raw_plot_buffer))
            plot_updated = True

        if min_resp_val_combined != float('inf') and max_resp_val_combined != float('-inf'):
            # ... (logika set_ylim untuk axs[1] tetap sama) ...
            if max_resp_val_combined > min_resp_val_combined : #
                padding_resp = 0.1 * max(abs(min_resp_val_combined), abs(max_resp_val_combined), 0.1) #
                self.axs[1].set_ylim(min_resp_val_combined - padding_resp, max_resp_val_combined + padding_resp if max_resp_val_combined > min_resp_val_combined else max_resp_val_combined + 0.5) #
            else: #
                self.axs[1].set_ylim(min_resp_val_combined - 0.5, max_resp_val_combined + 0.5) #


        min_rgb_val, max_rgb_val = 256, 0
        # ... (Logika update buffer r_raw_buffer, g_raw_buffer, b_raw_buffer dan plotnya tetap sama) ...
        if r_raw_value is not None: #
            self.r_raw_buffer = np.roll(self.r_raw_buffer, -1) #
            self.r_raw_buffer[-1] = r_raw_value #
            self.line_r_raw.set_ydata(self.r_raw_buffer) #
            if np.any(self.r_raw_buffer): #
                min_rgb_val = min(min_rgb_val, np.min(self.r_raw_buffer[self.r_raw_buffer != 0])) #
                max_rgb_val = max(max_rgb_val, np.max(self.r_raw_buffer)) #
            plot_updated = True #
            
        if g_raw_value is not None: #
            self.g_raw_buffer = np.roll(self.g_raw_buffer, -1) #
            self.g_raw_buffer[-1] = g_raw_value #
            self.line_g_raw.set_ydata(self.g_raw_buffer) #
            if np.any(self.g_raw_buffer): #
                min_rgb_val = min(min_rgb_val, np.min(self.g_raw_buffer[self.g_raw_buffer != 0])) #
                max_rgb_val = max(max_rgb_val, np.max(self.g_raw_buffer)) #
            plot_updated = True #

        if b_raw_value is not None: #
            self.b_raw_buffer = np.roll(self.b_raw_buffer, -1) #
            self.b_raw_buffer[-1] = b_raw_value #
            self.line_b_raw.set_ydata(self.b_raw_buffer) #
            if np.any(self.b_raw_buffer): #
                min_rgb_val = min(min_rgb_val, np.min(self.b_raw_buffer[self.b_raw_buffer != 0])) #
                max_rgb_val = max(max_rgb_val, np.max(self.b_raw_buffer)) #
            plot_updated = True #
        
        if r_raw_value is not None or g_raw_value is not None or b_raw_value is not None: #
            if max_rgb_val > min_rgb_val and max_rgb_val != 0 : #
                padding_rgb = 10  #
                self.axs[2].set_ylim(max(0, min_rgb_val - padding_rgb), min(255, max_rgb_val + padding_rgb)) #
            else: #
                 self.axs[2].set_ylim(0,256) #

    def clear_plots(self):
        # ... (logika clear_plots tetap sama, pastikan semua buffer di-reset) ...
        self.current_rppg_data.fill(0) #
        self.line_rppg.set_ydata(self.current_rppg_data) #
        self.axs[0].set_ylim(-1, 1) #
        
        self.current_resp_filtered_data.fill(0) #
        self.line_resp_filtered.set_ydata(self.current_resp_filtered_data) #
        self.resp_raw_plot_buffer.fill(0) #
        self.line_resp_raw.set_ydata(self.resp_raw_plot_buffer) #
        self.axs[1].set_ylim(-1, 1) #
        
        self.r_raw_buffer.fill(0) #
        self.g_raw_buffer.fill(0) #
        self.b_raw_buffer.fill(0) #
        self.line_r_raw.set_ydata(self.r_raw_buffer) #
        self.line_g_raw.set_ydata(self.g_raw_buffer) #
        self.line_b_raw.set_ydata(self.b_raw_buffer) #
        self.axs[2].set_ylim(0, 256) #
# visualization.py
import matplotlib.pyplot as plt
import numpy as np

class RealtimePlotter:
    def __init__(self, buffer_size):
        """
        Inisialisasi plotter untuk embedding.
        Args:
            buffer_size (int): Ukuran buffer untuk plotting sinyal.
        """
        self.buffer_size = buffer_size
        
        # Buat Figure dan Axes, figsize mungkin perlu disesuaikan agar pas di GUI
        # Ubah menjadi 3 baris, 1 kolom
        self.fig, self.axs = plt.subplots(3, 1, figsize=(6, 7)) # figsize (width, height) in inches, tinggi ditambah
        self.fig.suptitle("Sinyal Fisiologis & Mentah RGB")

        # Plot rPPG
        self.axs[0].set_title("Sinyal rPPG Terfilter")
        self.axs[0].set_xlabel("Sampel")
        self.axs[0].set_ylabel("Amplitudo")
        self.line_rppg, = self.axs[0].plot(np.zeros(self.buffer_size), color='blue', label='rPPG')
        self.axs[0].grid(True)
        self.axs[0].legend(loc='upper right')

        # Plot Respirasi
        self.axs[1].set_title("Sinyal Respirasi Terfilter")
        self.axs[1].set_xlabel("Sampel")
        self.axs[1].set_ylabel("Amplitudo")
        self.line_resp, = self.axs[1].plot(np.zeros(self.buffer_size), color='green', label='Respirasi')
        self.axs[1].grid(True)
        self.axs[1].legend(loc='upper right')

        # Plot Sinyal RGB Mentah BARU
        self.axs[2].set_title("Sinyal RGB Mentah dari ROI Wajah")
        self.axs[2].set_xlabel("Sampel")
        self.axs[2].set_ylabel("Intensitas Rata-rata")
        self.line_r_raw, = self.axs[2].plot(np.zeros(self.buffer_size), color='red', label='Merah (R)')
        self.line_g_raw, = self.axs[2].plot(np.zeros(self.buffer_size), color='green', label='Hijau (G)')
        self.line_b_raw, = self.axs[2].plot(np.zeros(self.buffer_size), color='blue', label='Biru (B)')
        self.axs[2].grid(True)
        self.axs[2].legend(loc='upper right') # Menampilkan legenda R, G, B

        # Buffer untuk menyimpan data sinyal RGB mentah (hanya untuk plotting di sini)
        self.r_raw_buffer = np.zeros(self.buffer_size)
        self.g_raw_buffer = np.zeros(self.buffer_size)
        self.b_raw_buffer = np.zeros(self.buffer_size)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Sesuaikan layout agar suptitle tidak overlap

    def get_figure(self):
        """Mengembalikan objek Figure Matplotlib."""
        return self.fig

    def update_plots(self, rppg_signal, resp_signal, r_raw_value=None, g_raw_value=None, b_raw_value=None): # Tambah argumen RGB
        """
        Memperbarui data plot. GUI akan menangani penggambaran canvas.
        Args:
            rppg_signal (np.array): Sinyal rPPG yang akan diplot.
            resp_signal (np.array): Sinyal respirasi yang akan diplot.
            r_raw_value (float, optional): Nilai sinyal R mentah terbaru.
            g_raw_value (float, optional): Nilai sinyal G mentah terbaru.
            b_raw_value (float, optional): Nilai sinyal B mentah terbaru.
        """
        plot_updated = False

        if rppg_signal is not None and len(rppg_signal) > 0:
            plot_data_rppg = np.zeros(self.buffer_size)
            data_to_plot_rppg = rppg_signal[-self.buffer_size:]
            plot_data_rppg[-len(data_to_plot_rppg):] = data_to_plot_rppg
            self.line_rppg.set_ydata(plot_data_rppg)

            if len(data_to_plot_rppg) > 1:
                min_val, max_val = np.min(data_to_plot_rppg), np.max(data_to_plot_rppg)
                padding_rppg = 0.1 * max(abs(min_val), abs(max_val), 0.1)
                self.axs[0].set_ylim(min_val - padding_rppg, max_val + padding_rppg if max_val > min_val else max_val + 0.5)
            plot_updated = True

        if resp_signal is not None and len(resp_signal) > 0:
            plot_data_resp = np.zeros(self.buffer_size)
            data_to_plot_resp = resp_signal[-self.buffer_size:]
            plot_data_resp[-len(data_to_plot_resp):] = data_to_plot_resp
            self.line_resp.set_ydata(plot_data_resp)

            if len(data_to_plot_resp) > 1:
                min_val, max_val = np.min(data_to_plot_resp), np.max(data_to_plot_resp)
                padding_resp = 0.1 * max(abs(min_val), abs(max_val), 0.1)
                self.axs[1].set_ylim(min_val - padding_resp, max_val + padding_resp if max_val > min_val else max_val + 0.5)
            plot_updated = True
        
        # Update plot RGB Mentah
        min_rgb_val, max_rgb_val = 256, 0 # Inisialisasi untuk mencari min/max keseluruhan channel RGB

        if r_raw_value is not None:
            self.r_raw_buffer = np.roll(self.r_raw_buffer, -1)
            self.r_raw_buffer[-1] = r_raw_value
            self.line_r_raw.set_ydata(self.r_raw_buffer)
            min_rgb_val = min(min_rgb_val, np.min(self.r_raw_buffer))
            max_rgb_val = max(max_rgb_val, np.max(self.r_raw_buffer))
            plot_updated = True
            
        if g_raw_value is not None:
            self.g_raw_buffer = np.roll(self.g_raw_buffer, -1)
            self.g_raw_buffer[-1] = g_raw_value
            self.line_g_raw.set_ydata(self.g_raw_buffer)
            min_rgb_val = min(min_rgb_val, np.min(self.g_raw_buffer))
            max_rgb_val = max(max_rgb_val, np.max(self.g_raw_buffer))
            plot_updated = True

        if b_raw_value is not None:
            self.b_raw_buffer = np.roll(self.b_raw_buffer, -1)
            self.b_raw_buffer[-1] = b_raw_value
            self.line_b_raw.set_ydata(self.b_raw_buffer)
            min_rgb_val = min(min_rgb_val, np.min(self.b_raw_buffer))
            max_rgb_val = max(max_rgb_val, np.max(self.b_raw_buffer))
            plot_updated = True
        
        if r_raw_value is not None or g_raw_value is not None or b_raw_value is not None:
             # Atur Y-limits untuk plot RGB, misalnya 0-255 atau dinamis
            # self.axs[2].set_ylim(0, 256) # Statis
            if max_rgb_val > min_rgb_val : # Cek jika ada data valid
                padding_rgb = 10 # Beri sedikit padding atas dan bawah (misal 10 unit intensitas)
                self.axs[2].set_ylim(max(0, min_rgb_val - padding_rgb), min(255, max_rgb_val + padding_rgb))
            else: # Fallback jika hanya satu nilai atau tidak ada data
                 self.axs[2].set_ylim(0,256)


    def clear_plots(self):
        """Membersihkan plot untuk penggunaan kembali."""
        self.line_rppg.set_ydata(np.zeros(self.buffer_size))
        self.line_resp.set_ydata(np.zeros(self.buffer_size))
        self.axs[0].set_ylim(-1, 1)
        self.axs[1].set_ylim(-1, 1)
        
        # Clear plot RGB mentah
        self.r_raw_buffer.fill(0)
        self.g_raw_buffer.fill(0)
        self.b_raw_buffer.fill(0)
        self.line_r_raw.set_ydata(self.r_raw_buffer)
        self.line_g_raw.set_ydata(self.g_raw_buffer)
        self.line_b_raw.set_ydata(self.b_raw_buffer)
        self.axs[2].set_ylim(0, 256) # Reset batas default untuk RGB
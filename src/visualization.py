# visualization.py
import matplotlib.pyplot as plt
import numpy as np

# SIGNAL_BUFFER_SIZE akan dioper dari gui.py/main.py

class RealtimePlotter:
    def __init__(self, buffer_size):
        """
        Inisialisasi plotter untuk embedding.
        Args:
            buffer_size (int): Ukuran buffer untuk plotting sinyal.
        """
        self.buffer_size = buffer_size
        
        # Buat Figure dan Axes, figsize mungkin perlu disesuaikan agar pas di GUI
        self.fig, self.axs = plt.subplots(2, 1, figsize=(6, 5)) # figsize (width, height) in inches
        self.fig.suptitle("Sinyal Fisiologis")

        # Plot rPPG
        self.axs[0].set_title("Sinyal rPPG Terfilter")
        self.axs[0].set_xlabel("Sampel")
        self.axs[0].set_ylabel("Amplitudo")
        self.line_rppg, = self.axs[0].plot(np.zeros(self.buffer_size), color='blue')
        self.axs[0].grid(True)

        # Plot Respirasi
        self.axs[1].set_title("Sinyal Respirasi Terfilter")
        self.axs[1].set_xlabel("Sampel")
        self.axs[1].set_ylabel("Amplitudo")
        self.line_resp, = self.axs[1].plot(np.zeros(self.buffer_size), color='green')
        self.axs[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Sesuaikan layout agar suptitle tidak overlap

    def get_figure(self):
        """Mengembalikan objek Figure Matplotlib."""
        return self.fig

    def update_plots(self, rppg_signal, resp_signal):
        """
        Memperbarui data plot. GUI akan menangani penggambaran canvas.
        Args:
            rppg_signal (np.array): Sinyal rPPG yang akan diplot.
            resp_signal (np.array): Sinyal respirasi yang akan diplot.
        """
        plot_updated = False

        if rppg_signal is not None and len(rppg_signal) > 0:
            plot_data_rppg = np.zeros(self.buffer_size)
            data_to_plot_rppg = rppg_signal[-self.buffer_size:]
            plot_data_rppg[-len(data_to_plot_rppg):] = data_to_plot_rppg
            self.line_rppg.set_ydata(plot_data_rppg)

            if len(data_to_plot_rppg) > 1:
                min_val, max_val = np.min(data_to_plot_rppg), np.max(data_to_plot_rppg)
                padding_rppg = 0.1 * max(abs(min_val), abs(max_val), 0.1) # Hindari padding 0
                self.axs[0].set_ylim(min_val - padding_rppg, max_val + padding_rppg if max_val > min_val else max_val + 0.5)
            plot_updated = True

        if resp_signal is not None and len(resp_signal) > 0:
            plot_data_resp = np.zeros(self.buffer_size)
            data_to_plot_resp = resp_signal[-self.buffer_size:]
            plot_data_resp[-len(data_to_plot_resp):] = data_to_plot_resp
            self.line_resp.set_ydata(plot_data_resp)

            if len(data_to_plot_resp) > 1:
                min_val, max_val = np.min(data_to_plot_resp), np.max(data_to_plot_resp)
                padding_resp = 0.1 * max(abs(min_val), abs(max_val), 0.1) # Hindari padding 0
                self.axs[1].set_ylim(min_val - padding_resp, max_val + padding_resp if max_val > min_val else max_val + 0.5)
            plot_updated = True
        
        # GUI akan memanggil canvas.draw_idle()
        # if plot_updated and hasattr(self.fig, 'canvas') and self.fig.canvas:
        #     try:
        #         self.fig.canvas.draw_idle()
        #     except Exception as e:
        #         print(f"Error saat self-drawing plot Matplotlib: {e}")

    def clear_plots(self):
        """Membersihkan plot untuk penggunaan kembali."""
        self.line_rppg.set_ydata(np.zeros(self.buffer_size))
        self.line_resp.set_ydata(np.zeros(self.buffer_size))
        self.axs[0].set_ylim(-1, 1) # Reset batas default
        self.axs[1].set_ylim(-1, 1) # Reset batas default
        # Judul dan label tetap ada

    # Tidak perlu metode close() yang rumit jika GUI menangani penghancuran canvas.
    # plt.close(self.fig) akan menutup figure sepenuhnya.
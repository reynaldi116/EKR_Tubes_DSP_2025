# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time

# Impor modul-modul Anda
from camera.video_capture import VideoCapture
from signal_processing.signal_processing import SignalProcessor, SIGNAL_BUFFER_SIZE, RESP_LOWCUT, RESP_HIGHCUT
from visualization.visualization import RealtimePlotter
from utils.utils import detect_face_roi, get_roi_pixels

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Parameter
USE_DUMMY_RESP_SIGNAL = True
DUMMY_RESP_ROI_MOVEMENT_SIM = 0.1

# Konstanta untuk ukuran tampilan video di GUI
VIDEO_DISPLAY_WIDTH = 640  # Lebar tampilan video (piksel)
VIDEO_DISPLAY_HEIGHT = 480 # Tinggi tampilan video (piksel)

class AppGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pengukuran Fisiologis Terintegrasi")
        self.geometry("1200x750") # Sedikit lebih tinggi untuk padding

        # Tambahkan konstanta ukuran video sebagai atribut instance jika ingin diakses dengan self
        self.VIDEO_DISPLAY_WIDTH = VIDEO_DISPLAY_WIDTH
        self.VIDEO_DISPLAY_HEIGHT = VIDEO_DISPLAY_HEIGHT

        self.processing_fps = 0
        self.frame_count_fps_calc = 0
        self.start_time_fps_calc = time.time()

        self.video_stream = None
        self.processor = None
        self.plotter = None
        self.plot_canvas_agg = None
        self.plot_canvas_widget = None

        self.processing_thread = None
        self.is_processing = False
        self.effective_fps = 30.0

        self.main_left_frame = ttk.Frame(self)
        self.main_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.main_right_frame = ttk.Frame(self)
        self.main_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_display_frame = ttk.LabelFrame(self.main_left_frame, text="Feed Video")
        # Beri ukuran minimum pada frame video agar tidak kolaps jika video belum ada
        self.video_display_frame.config(width=self.VIDEO_DISPLAY_WIDTH + 20, height=self.VIDEO_DISPLAY_HEIGHT + 20)
        self.video_display_frame.pack_propagate(False) # Mencegah frame menyesuaikan ukuran ke kontennya
        self.video_display_frame.pack(pady=5, padx=5, fill="both", expand=True)
        
        self.video_label = ttk.Label(self.video_display_frame)
        # video_label akan di-pack di tengah frame video_display_frame
        self.video_label.pack(pady=5, padx=5, anchor=tk.CENTER)


        self.data_frame = ttk.LabelFrame(self.main_left_frame, text="Data Fisiologis")
        self.data_frame.pack(pady=10, padx=5, fill="x")

        self.bpm_label = ttk.Label(self.data_frame, text="BPM (rPPG): --", font=("Helvetica", 12))
        self.bpm_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.rpm_label = ttk.Label(self.data_frame, text="RPM (Resp): --", font=("Helvetica", 12))
        self.rpm_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        self.gui_fps_label = ttk.Label(self.data_frame, text="GUI FPS: --", font=("Helvetica", 9))
        self.gui_fps_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        self.processing_fps_label = ttk.Label(self.data_frame, text="Processing FPS: --", font=("Helvetica", 9))
        self.processing_fps_label.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        self.control_frame = ttk.Frame(self.main_left_frame)
        self.control_frame.pack(pady=10, padx=5, fill="x")

        self.start_button = ttk.Button(self.control_frame, text="Mulai", command=self.start_processing)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(self.control_frame, text="Berhenti", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=5)

        self.plot_display_frame = ttk.LabelFrame(self.main_right_frame, text="Plot Sinyal")
        self.plot_display_frame.pack(pady=5, padx=5, fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Inisialisasi dengan gambar placeholder untuk video_label
        placeholder_img = Image.new('RGB', (self.VIDEO_DISPLAY_WIDTH, self.VIDEO_DISPLAY_HEIGHT), color = 'grey')
        imgtk_placeholder = ImageTk.PhotoImage(image=placeholder_img)
        self.video_label.imgtk = imgtk_placeholder
        self.video_label.config(image=imgtk_placeholder)


    def initialize_processing_components(self):
        try:
            self.video_stream = VideoCapture(device_id=0)
            if self.video_stream.fps is None or self.video_stream.fps <= 0:
                messagebox.showwarning("Peringatan FPS", "FPS kamera tidak valid. Menggunakan nilai default 30 FPS.")
                self.effective_fps = 30.0
            else:
                self.effective_fps = self.video_stream.fps
            
            self.processor = SignalProcessor(fs=self.effective_fps, buffer_size=SIGNAL_BUFFER_SIZE)
            self.plotter = RealtimePlotter(buffer_size=SIGNAL_BUFFER_SIZE)
            
            if self.plot_canvas_widget:
                self.plot_canvas_widget.destroy()
                self.plot_canvas_widget = None
                self.plot_canvas_agg = None

            self.plot_canvas_agg = FigureCanvasTkAgg(self.plotter.get_figure(), master=self.plot_display_frame)
            self.plot_canvas_agg.draw()
            self.plot_canvas_widget = self.plot_canvas_agg.get_tk_widget()
            self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            return True
        except IOError as e:
            messagebox.showerror("Error Kamera", f"Tidak dapat membuka kamera: {e}")
            return False
        except Exception as e:
            messagebox.showerror("Error Inisialisasi", f"Gagal menginisialisasi komponen: {e}")
            return False

    def start_processing(self):
        if not self.initialize_processing_components():
            return

        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        self.update_gui_fps_display()

    def stop_processing(self, called_on_exit=False):
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        if self.video_stream:
            self.video_stream.release()
            self.video_stream = None
        
        if self.plot_canvas_widget and not called_on_exit:
            self.plot_canvas_widget.destroy()
            self.plot_canvas_widget = None
            self.plot_canvas_agg = None
        
        if self.plotter and not called_on_exit:
            self.plotter.clear_plots()
            if self.plot_canvas_agg:
                self.plot_canvas_agg.draw()
        
        if not called_on_exit:
            # Kembali ke placeholder saat berhenti
            placeholder_img = Image.new('RGB', (self.VIDEO_DISPLAY_WIDTH, self.VIDEO_DISPLAY_HEIGHT), color = 'grey')
            imgtk_placeholder = ImageTk.PhotoImage(image=placeholder_img)
            self.video_label.imgtk = imgtk_placeholder
            self.video_label.config(image=imgtk_placeholder)

            self.bpm_label.config(text="BPM (rPPG): --")
            self.rpm_label.config(text="RPM (Resp): --")
            self.processing_fps_label.config(text="Processing FPS: --")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        print("Pemrosesan dihentikan.")

    def _process_loop(self):
        frame_count_proc = 0
        start_time_proc = time.time()
        internal_processing_fps = 0
        dummy_resp_time_counter = 0.0

        while self.is_processing and self.video_stream:
            ret, frame = self.video_stream.get_frame()
            if not ret:
                self.after(0, lambda: messagebox.showerror("Stream Error", "Gagal mendapatkan frame."))
                self.after(10, self.stop_processing)
                break
            
            processed_frame_for_display = frame.copy()

            face_roi_coords = detect_face_roi(frame)
            rppg_signal_value = 0.0
            if face_roi_coords is not None:
                x, y, w, h = face_roi_coords
                cv2.rectangle(processed_frame_for_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = get_roi_pixels(frame, face_roi_coords)
                if face_roi.size > 0 and len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
                    rppg_signal_value = np.mean(face_roi[:, :, 1])
                else:
                    rppg_signal_value = np.mean(frame[:,:,1])
            else:
                rppg_signal_value = np.mean(frame[:,:,1])

            filtered_rppg, bpm = self.processor.process_rppg(rppg_signal_value)

            if USE_DUMMY_RESP_SIGNAL:
                target_resp_freq = (RESP_LOWCUT + RESP_HIGHCUT) / 2.0
                resp_signal_value = DUMMY_RESP_ROI_MOVEMENT_SIM * np.sin(2 * np.pi * target_resp_freq * dummy_resp_time_counter) + \
                                    (np.random.rand() - 0.5) * 0.05
                dummy_resp_time_counter += (1.0 / self.effective_fps)
            else:
                resp_signal_value = 0.0

            filtered_resp, rpm = self.processor.process_respiration(resp_signal_value)

            # --- PERBAIKAN: Resize frame untuk GUI ke ukuran konstan ---
            if processed_frame_for_display is not None and processed_frame_for_display.shape[0] > 0 and processed_frame_for_display.shape[1] > 0:
                # Pertahankan aspek rasio asli saat me-resize agar pas di self.VIDEO_DISPLAY_WIDTH, self.VIDEO_DISPLAY_HEIGHT
                original_h, original_w = processed_frame_for_display.shape[:2]
                target_w, target_h = self.VIDEO_DISPLAY_WIDTH, self.VIDEO_DISPLAY_HEIGHT
                
                aspect_ratio_original = original_w / original_h
                aspect_ratio_target = target_w / target_h

                if aspect_ratio_original > aspect_ratio_target: # Frame asli lebih lebar
                    new_w = target_w
                    new_h = int(new_w / aspect_ratio_original)
                else: # Frame asli lebih tinggi atau sama
                    new_h = target_h
                    new_w = int(new_h * aspect_ratio_original)
                
                # Pastikan new_w dan new_h > 0
                if new_w > 0 and new_h > 0:
                    sized_frame_for_gui_content = cv2.resize(processed_frame_for_display, (new_w, new_h))
                    # Buat canvas kosong (hitam atau abu-abu) seukuran target display
                    sized_frame_for_gui = np.full((target_h, target_w, 3), 60, dtype=np.uint8) # Latar belakang abu-abu
                    # Tempelkan frame yang sudah diresize ke tengah canvas (letterboxing/pillarboxing)
                    x_offset = (target_w - new_w) // 2
                    y_offset = (target_h - new_h) // 2
                    sized_frame_for_gui[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = sized_frame_for_gui_content
                else:
                    sized_frame_for_gui = np.full((target_h, target_w, 3), 0, dtype=np.uint8) # Fallback frame hitam
            else:
                sized_frame_for_gui = np.full((self.VIDEO_DISPLAY_HEIGHT, self.VIDEO_DISPLAY_WIDTH, 3), 0, dtype=np.uint8) # Frame hitam jika error
            # -----------------------------------------------------------
            
            self.after(0, self._update_gui_elements, sized_frame_for_gui, bpm, rpm, internal_processing_fps)

            if self.plotter and self.plot_canvas_agg:
                self.plotter.update_plots(
                    filtered_rppg if len(filtered_rppg) > 0 else self.processor.get_raw_rppg_signal_for_plot(),
                    filtered_resp if len(filtered_resp) > 0 else self.processor.get_raw_resp_signal_for_plot()
                )
                self.after(0, lambda: self.plot_canvas_agg.draw_idle() if self.plot_canvas_agg and self.winfo_exists() else None)
            
            frame_count_proc += 1
            elapsed_time_proc = time.time() - start_time_proc
            if elapsed_time_proc >= 1.0:
                internal_processing_fps = frame_count_proc / elapsed_time_proc
                frame_count_proc = 0
                start_time_proc = time.time()

        print("Loop pemrosesan GUI berakhir.")

    def _update_gui_elements(self, frame_cv_display, bpm, rpm, proc_fps):
        if not self.is_processing and not self.winfo_exists():
             # Jika tidak sedang memproses, jangan update dengan frame kamera, biarkan placeholder
            if not self.is_processing:
                return
        
        # Jika window sudah tidak ada (misalnya ditutup saat proses masih berjalan di thread)
        if not self.winfo_exists():
            return

        # frame_cv_display sudah diresize ke ukuran konstan (VIDEO_DISPLAY_WIDTH, VIDEO_DISPLAY_HEIGHT)
        # dengan letterboxing/pillarboxing
        img = cv2.cvtColor(frame_cv_display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.bpm_label.config(text=f"BPM (rPPG): {bpm:.2f}")
        self.rpm_label.config(text=f"RPM (Resp): {rpm:.2f}")
        self.processing_fps_label.config(text=f"Processing FPS: {proc_fps:.2f}")

    def update_gui_fps_display(self):
        if self.winfo_exists(): # Hanya update jika window masih ada
            if self.is_processing : # Hanya update GUI FPS jika sedang memproses
                self.frame_count_fps_calc +=1
                elapsed_time = time.time() - self.start_time_fps_calc
                if elapsed_time >= 1.0:
                    current_gui_fps = self.frame_count_fps_calc / elapsed_time
                    self.gui_fps_label.config(text=f"GUI FPS: {current_gui_fps:.2f}")
                    self.frame_count_fps_calc = 0
                    self.start_time_fps_calc = time.time()
            self.after(50, self.update_gui_fps_display)


    def on_closing(self):
        if self.is_processing:
            if messagebox.askokcancel("Keluar", "Pemrosesan sedang berjalan. Apakah Anda yakin ingin keluar?"):
                self.stop_processing(called_on_exit=True)
                if self.plot_canvas_widget:
                    self.plot_canvas_widget.destroy()
                plt.close('all')
                self.destroy()
            else:
                return
        else:
            if self.plot_canvas_widget:
                self.plot_canvas_widget.destroy()
            plt.close('all')
            self.destroy()

# Bagian if __name__ == "__main__": dipindahkan ke main.py
# if __name__ == "__main__":
#     app = AppGUI()
#     app.mainloop()
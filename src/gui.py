# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog # Tambahkan filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os # Tambahkan os untuk path direktori

from video_capture import VideoCapture
from signal_processing import SignalProcessor, SIGNAL_BUFFER_SIZE
# Pastikan visualization.py sudah dimodifikasi untuk 3 plot
from visualization import RealtimePlotter
from utils import FaceDetectorMP, get_roi_pixels
from pose_respiration_tracker import PoseRespirationTracker

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

VIDEO_DISPLAY_WIDTH = 640
VIDEO_DISPLAY_HEIGHT = 480

class AppGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RPPG & Pernapasan (MediaPipe Face & Pose)")
        self.geometry("1250x750") # Mungkin perlu diperbesar jika plotnya jadi lebih tinggi

        self.VIDEO_DISPLAY_WIDTH = VIDEO_DISPLAY_WIDTH
        self.VIDEO_DISPLAY_HEIGHT = VIDEO_DISPLAY_HEIGHT

        self.processing_fps = 0.0
        self.frame_count_fps_calc = 0
        self.start_time_fps_calc = time.time()

        self.video_stream = None
        self.processor = None
        self.plotter = None
        self.plot_canvas_agg = None
        self.plot_canvas_widget = None
        
        self.face_detector_mp = None
        self.pose_tracker = None     
        self.raw_resp_debug_label = None

        self.processing_thread = None
        self.is_processing = False
        self.effective_fps = 10.0 # Sesuai dengan yang Anda berikan

        self.bpm_history = []
        self.rpm_history = []
        self.rate_history_size = 15 # Sesuai dengan yang Anda berikan

        # Folder untuk menyimpan plot
        self.plot_save_path = "saved_plots"
        if not os.path.exists(self.plot_save_path):
            os.makedirs(self.plot_save_path)
            print(f"Folder '{self.plot_save_path}' telah dibuat.")

        self.main_left_frame = ttk.Frame(self)
        self.main_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.main_right_frame = ttk.Frame(self)
        self.main_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._setup_left_panel()
        self._setup_right_panel()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._initialize_video_placeholder()

    def _setup_left_panel(self):
        self.video_display_frame = ttk.LabelFrame(self.main_left_frame, text="Feed Video")
        self.video_display_frame.config(width=self.VIDEO_DISPLAY_WIDTH + 20, height=self.VIDEO_DISPLAY_HEIGHT + 20)
        self.video_display_frame.pack_propagate(False)
        self.video_display_frame.pack(pady=5, padx=5, fill="both", expand=True)
        self.video_label = ttk.Label(self.video_display_frame)
        self.video_label.pack(pady=5, padx=5, anchor=tk.CENTER)

        self.data_frame = ttk.LabelFrame(self.main_left_frame, text="Data Fisiologis")
        self.data_frame.pack(pady=10, padx=5, fill="x")
        self.bpm_label = ttk.Label(self.data_frame, text="BPM (rPPG): --", font=("Helvetica", 12))
        self.bpm_label.grid(row=0, column=0, padx=10, pady=3, sticky="w")
        self.rpm_label = ttk.Label(self.data_frame, text="RPM (Resp): --", font=("Helvetica", 12))
        self.rpm_label.grid(row=0, column=1, padx=10, pady=3, sticky="w")
        self.processing_fps_label = ttk.Label(self.data_frame, text="Processing FPS: --", font=("Helvetica", 9))
        self.processing_fps_label.grid(row=1, column=0, padx=10, pady=3, sticky="w")
        self.gui_fps_label = ttk.Label(self.data_frame, text="GUI FPS: --", font=("Helvetica", 9))
        self.gui_fps_label.grid(row=1, column=1, padx=10, pady=3, sticky="w")
        self.raw_resp_debug_label = ttk.Label(self.data_frame, text="Raw Resp Motion: --", font=("Helvetica", 9))
        self.raw_resp_debug_label.grid(row=2, column=0, columnspan=2, padx=10, pady=3, sticky="w")

        self.control_frame = ttk.Frame(self.main_left_frame)
        self.control_frame.pack(pady=10, padx=5, fill="x")
        self.start_button = ttk.Button(self.control_frame, text="Mulai", command=lambda: self.start_processing())
        self.start_button.pack(side="left", padx=5, pady=5)
        self.stop_button = ttk.Button(self.control_frame, text="Berhenti", command=lambda: self.stop_processing(), state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=5, pady=5)

        # Tombol baru untuk menyimpan plot
        self.save_plot_button = ttk.Button(self.control_frame, text="Simpan Plot RGB", command=self.save_rgb_plot)
        self.save_plot_button.pack(side="left", padx=5, pady=5)
        self.save_plot_button.config(state=tk.DISABLED) # Awalnya disable

    def _setup_right_panel(self):
        self.plot_display_frame = ttk.LabelFrame(self.main_right_frame, text="Plot Sinyal")
        self.plot_display_frame.pack(pady=5, padx=5, fill="both", expand=True)

    def _initialize_video_placeholder(self):
        placeholder_img = Image.new('RGB', (self.VIDEO_DISPLAY_WIDTH, self.VIDEO_DISPLAY_HEIGHT), color = (128, 128, 128))
        self.imgtk_placeholder_ref = ImageTk.PhotoImage(image=placeholder_img)
        self.video_label.imgtk = self.imgtk_placeholder_ref
        self.video_label.config(image=self.imgtk_placeholder_ref)

    def initialize_processing_components(self):
        try:
            self.video_stream = VideoCapture(device_id=0)
            current_cam_fps = self.video_stream.fps if self.video_stream.fps and self.video_stream.fps > 0 else self.effective_fps
            if not (self.video_stream.fps and self.video_stream.fps > 0):
                 messagebox.showwarning("Peringatan FPS Kamera",
                                       f"FPS kamera tidak valid ({self.video_stream.fps}). Menggunakan nilai {self.effective_fps} FPS untuk pemrosesan.")
            else:
                 print(f"Kamera FPS terdeteksi: {current_cam_fps}. Pemrosesan akan menggunakan fs={self.effective_fps}")

            self.processor = SignalProcessor(fs=self.effective_fps, buffer_size=SIGNAL_BUFFER_SIZE)
            self.plotter = RealtimePlotter(buffer_size=SIGNAL_BUFFER_SIZE)
            
            self.face_detector_mp = FaceDetectorMP(model_selection=0)
            self.pose_tracker = PoseRespirationTracker(model_complexity=1)
            
            if self.plot_canvas_widget: self.plot_canvas_widget.destroy()
            self.plot_canvas_agg = FigureCanvasTkAgg(self.plotter.get_figure(), master=self.plot_display_frame)
            self.plot_canvas_agg.draw()
            self.plot_canvas_widget = self.plot_canvas_agg.get_tk_widget()
            self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            return True
        except Exception as e:
            messagebox.showerror("Error Inisialisasi", f"Gagal menginisialisasi komponen: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_processing(self):
        if not self.initialize_processing_components(): return
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_plot_button.config(state=tk.NORMAL) # Enable tombol simpan
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        self.update_gui_fps_display()

    def stop_processing(self, called_on_exit=False):
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.5)
        if self.video_stream:
            self.video_stream.release(); self.video_stream = None
        
        if not called_on_exit:
            if self.plot_canvas_widget:
                self.plot_canvas_widget.destroy(); self.plot_canvas_widget = None; self.plot_canvas_agg = None
            if self.plotter: self.plotter.clear_plots()
            self._initialize_video_placeholder()
            self.bpm_label.config(text="BPM (rPPG): --"); self.rpm_label.config(text="RPM (Resp): --")
            self.processing_fps_label.config(text="Processing FPS: --"); self.gui_fps_label.config(text="GUI FPS: --")
            if self.raw_resp_debug_label: self.raw_resp_debug_label.config(text="Raw Resp Motion: --")
            self.bpm_history = []; self.rpm_history = []
            self.start_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
            self.save_plot_button.config(state=tk.DISABLED) # Disable tombol simpan
        print("Pemrosesan dihentikan.")

    def _process_loop(self):
        frame_count_proc_fps = 0
        start_time_proc_fps = time.time()
        current_processing_fps = 0.0
        
        r_signal_value, g_signal_value, b_signal_value = 0.0, 0.0, 0.0
        target_frame_duration = 1.0 / self.effective_fps

        while self.is_processing and self.video_stream:
            loop_start_time = time.time()

            ret, frame_original_bgr = self.video_stream.get_frame()
            if not ret:
                if self.is_processing: self.after(0, lambda: messagebox.showerror("Stream Error", "Gagal mendapatkan frame."))
                self.is_processing = False; break
            
            frame_original_rgb_mp = cv2.cvtColor(frame_original_bgr, cv2.COLOR_BGR2RGB) # Untuk MediaPipe
            processed_frame_for_drawing = frame_original_bgr.copy()

            face_bbox = self.face_detector_mp.detect_face_bounding_box(frame_original_rgb_mp)
            
            r_fallback = np.mean(frame_original_bgr[:, :, 2])
            g_fallback = np.mean(frame_original_bgr[:, :, 1])
            b_fallback = np.mean(frame_original_bgr[:, :, 0])
            r_signal_value, g_signal_value, b_signal_value = r_fallback, g_fallback, b_fallback
            
            if face_bbox is not None:
                cv2.rectangle(processed_frame_for_drawing,
                              (face_bbox[0], face_bbox[1]),
                              (face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]),
                              (0, 255, 0), 2)
                face_roi_pixels = get_roi_pixels(frame_original_bgr, face_bbox)
                if face_roi_pixels.size > 0 and len(face_roi_pixels.shape) == 3:
                    r_signal_value = np.mean(face_roi_pixels[:, :, 2]) # Red
                    g_signal_value = np.mean(face_roi_pixels[:, :, 1]) # Green
                    b_signal_value = np.mean(face_roi_pixels[:, :, 0]) # Blue
            
            filtered_rppg, bpm_current = self.processor.process_rppg(g_signal_value)

            raw_resp_motion_signal, pose_detected = self.pose_tracker.get_respiration_signal_and_draw_landmarks(
                frame_original_rgb_mp, processed_frame_for_drawing
            )
            filtered_resp, rpm_current = self.processor.process_respiration(raw_resp_motion_signal)
            
            averaged_bpm = bpm_current; averaged_rpm = rpm_current
            if bpm_current > 0:
                self.bpm_history.append(bpm_current)
                if len(self.bpm_history) > self.rate_history_size: self.bpm_history.pop(0)
                if self.bpm_history: averaged_bpm = np.mean(self.bpm_history)
            if rpm_current > 0:
                self.rpm_history.append(rpm_current)
                if len(self.rpm_history) > self.rate_history_size: self.rpm_history.pop(0)
                if self.rpm_history: averaged_rpm = np.mean(self.rpm_history)

            frame_for_gui_display = self._prepare_frame_for_display(processed_frame_for_drawing)
            
            if self.winfo_exists():
                self.after(0, self._update_gui_data, frame_for_gui_display, averaged_bpm, averaged_rpm, current_processing_fps, raw_resp_motion_signal)

            if self.plotter and self.plot_canvas_agg and self.winfo_exists():
                rppg_plot_data = filtered_rppg if len(filtered_rppg) > 0 else self.processor.get_raw_rppg_signal_for_plot()
                resp_plot_data = filtered_resp if len(filtered_resp) > 0 else self.processor.get_raw_resp_signal_for_plot()
                
                self.plotter.update_plots(rppg_plot_data, resp_plot_data,
                                          r_raw_value=r_signal_value,
                                          g_raw_value=g_signal_value,
                                          b_raw_value=b_signal_value)
                self.after(0, lambda: self.plot_canvas_agg.draw_idle() if self.plot_canvas_agg and self.winfo_exists() else None)
            
            frame_count_proc_fps += 1
            elapsed_time_proc_cycle = time.time() - start_time_proc_fps
            if elapsed_time_proc_cycle >= 1.0:
                current_processing_fps = frame_count_proc_fps / elapsed_time_proc_cycle
                frame_count_proc_fps = 0; start_time_proc_fps = time.time()
            
            loop_processing_time = time.time() - loop_start_time
            sleep_time = target_frame_duration - loop_processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if self.winfo_exists() and not self.is_processing:
            self.after(0, self.stop_processing)
        print("Loop pemrosesan GUI berakhir.")

    # --- Metode Baru untuk Menyimpan Plot ---
    def save_rgb_plot(self):
        if not self.is_processing:
            messagebox.showwarning("Simpan Plot", "Pemrosesan tidak sedang berjalan. Tidak ada plot untuk disimpan.")
            return

        if self.plotter and self.plotter.get_figure():
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Perbaikan: Pastikan path join benar, dan folder ada
                filename = os.path.join(self.plot_save_path, f"rgb_plot_{timestamp}.png")
                
                figure_to_save = self.plotter.get_figure()
                figure_to_save.savefig(filename, dpi=150) # Tambahkan dpi untuk kualitas
                messagebox.showinfo("Simpan Plot", f"Plot RGB berhasil disimpan sebagai:\n{filename}")
            except Exception as e:
                messagebox.showerror("Simpan Plot Error", f"Gagal menyimpan plot: {e}\nPastikan folder '{self.plot_save_path}' ada dan dapat ditulis.")
                import traceback
                traceback.print_exc()
        else:
            messagebox.showerror("Simpan Plot Error", "Objek plotter atau figure tidak tersedia.")

    def _prepare_frame_for_display(self, frame_to_display):
        if frame_to_display is None or frame_to_display.size == 0:
            return np.full((self.VIDEO_DISPLAY_HEIGHT, self.VIDEO_DISPLAY_WIDTH, 3), 128, dtype=np.uint8)
        
        original_h, original_w = frame_to_display.shape[:2]
        target_w, target_h = self.VIDEO_DISPLAY_WIDTH, self.VIDEO_DISPLAY_HEIGHT
        
        aspect_ratio_original = original_w / original_h
        aspect_ratio_target = target_w / target_h

        if aspect_ratio_original > aspect_ratio_target:
            new_w = target_w
            new_h = int(target_w / aspect_ratio_original)
        else:
            new_h = target_h
            new_w = int(target_h * aspect_ratio_original)

        if new_w <= 0 or new_h <= 0:
             return np.full((target_h, target_w, 3), 128, dtype=np.uint8)

        resized_content = cv2.resize(frame_to_display, (new_w, new_h))
        
        full_sized_frame = np.full((target_h, target_w, 3), (128,128,128), dtype=np.uint8)
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        full_sized_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_content
        return full_sized_frame

    def _update_gui_data(self, frame_cv_display, bpm_to_display, rpm_to_display, proc_fps, raw_resp_signal_val):
        if not self.winfo_exists(): return

        img = cv2.cvtColor(frame_cv_display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        
        self.imgtk_display_ref = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = self.imgtk_display_ref
        self.video_label.config(image=self.imgtk_display_ref)
        
        self.bpm_label.config(text=f"BPM (rPPG): {bpm_to_display:.1f}")
        self.rpm_label.config(text=f"RPM (Resp): {rpm_to_display:.1f}")
        self.processing_fps_label.config(text=f"Processing FPS: {proc_fps:.2f}")
        if self.raw_resp_debug_label: self.raw_resp_debug_label.config(text=f"Raw Resp Motion: {raw_resp_signal_val:.4f}")

    def update_gui_fps_display(self):
        if self.winfo_exists():
            self.frame_count_fps_calc +=1
            elapsed_time = time.time() - self.start_time_fps_calc
            if elapsed_time >= 1.0:
                current_gui_fps = self.frame_count_fps_calc / elapsed_time
                if self.winfo_exists(): self.gui_fps_label.config(text=f"GUI FPS: {current_gui_fps:.2f}")
                self.frame_count_fps_calc = 0
                self.start_time_fps_calc = time.time()
            
            if self.is_processing or self.winfo_exists():
                 self.after(50, self.update_gui_fps_display)

    def on_closing(self):
        if self.is_processing:
            user_choice = messagebox.askyesnocancel("Keluar", "Pemrosesan sedang berjalan. Hentikan dan keluar?")
            if user_choice is True:
                self.is_processing = False
                if self.processing_thread and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=0.5)
                self._cleanup_resources()
                self.destroy()
            elif user_choice is False:
                # Jika pengguna memilih 'No' untuk "Hentikan dan keluar?", berarti mereka ingin keluar tanpa menghentikan.
                # Ini biasanya berarti langsung destroy. Atau, jika maksudnya 'Cancel', maka tidak melakukan apa-apa.
                # Berdasarkan logika askyesnocancel, 'No' berarti keluar juga, sama seperti 'Yes'.
                # Jika ingin 'Cancel' tidak melakukan apa-apa, maka hanya 'Yes' yang destroy.
                self._cleanup_resources()
                self.destroy()
            # Jika user_choice is None (Cancel), tidak melakukan apa-apa
        else:
            self._cleanup_resources()
            self.destroy()

    def _cleanup_resources(self):
        print("Membersihkan resources...")
        if self.video_stream: self.video_stream.release()
        if self.face_detector_mp: self.face_detector_mp.close()
        if self.pose_tracker: self.pose_tracker.close()      
        if self.plot_canvas_widget: self.plot_canvas_widget.destroy()
        plt.close('all')
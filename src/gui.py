# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
from video_capture import VideoCapture
from signal_processing import SignalProcessor, SIGNAL_BUFFER_SIZE
from visualization import RealtimePlotter # Pastikan ini versi yang menampilkan semua 4 sinyal dalam 3 subplot & get_current_plot_data()
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
        self.geometry("1250x750")

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
        self.effective_fps = 30.0 
        print(f"Target effective FPS set to: {self.effective_fps}")

        self.bpm_history = []
        self.rpm_history = []
        self.rate_history_size = 15 

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
        print("AppGUI Initialized")

    def _setup_left_panel(self):
        self.video_display_frame = ttk.LabelFrame(self.main_left_frame, text="Feed Video")
        self.video_display_frame.config(width=self.VIDEO_DISPLAY_WIDTH + 20, height=self.VIDEO_DISPLAY_HEIGHT + 20)
        self.video_display_frame.pack_propagate(False)
        self.video_display_frame.pack(pady=5, padx=5, fill="both", expand=True)
        self.video_label = ttk.Label(self.video_display_frame)
        self.video_label.pack(pady=5, padx=5, anchor=tk.CENTER)
        print("Video Label Created")

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
        
        # Tombol untuk menyimpan plot dengan layout 4 subplot terpisah
        self.save_custom_layout_button = ttk.Button(self.control_frame, text="Simpan Plot (4 Subplot)", command=self.save_plot_with_custom_layout)
        self.save_custom_layout_button.pack(side="left", padx=5, pady=5)
        self.save_custom_layout_button.config(state=tk.DISABLED)


    def _setup_right_panel(self):
        self.plot_display_frame = ttk.LabelFrame(self.main_right_frame, text="Plot Sinyal")
        self.plot_display_frame.pack(pady=5, padx=5, fill="both", expand=True)


    def _initialize_video_placeholder(self):
        print("Initializing video placeholder...")
        try:
            placeholder_img = Image.new('RGB', (self.VIDEO_DISPLAY_WIDTH, self.VIDEO_DISPLAY_HEIGHT), color = (128, 128, 128))
            self.imgtk_placeholder_ref = ImageTk.PhotoImage(image=placeholder_img)
            if hasattr(self, 'video_label') and self.video_label:
                 self.video_label.imgtk = self.imgtk_placeholder_ref
                 self.video_label.config(image=self.imgtk_placeholder_ref)
                 print("Video placeholder set.")
            else:
                print("Error: video_label not initialized before placeholder.")
        except Exception as e:
            print(f"Error initializing placeholder: {e}")


    def initialize_processing_components(self):
        print("Initializing processing components...")
        try:
            self.video_stream = VideoCapture(device_id=0) 
            print(f"VideoCapture opened: {self.video_stream.cap.isOpened() if self.video_stream and self.video_stream.cap else 'N/A'}")
            
            actual_cam_fps = self.video_stream.fps if self.video_stream.fps and self.video_stream.fps > 0 else None
            if not actual_cam_fps:
                 messagebox.showwarning("Peringatan FPS Kamera",
                                       f"FPS kamera tidak valid ({self.video_stream.fps}). Pemrosesan akan menggunakan target fs={self.effective_fps} FPS.")
            else:
                 print(f"Kamera FPS terdeteksi: {actual_cam_fps}. Pemrosesan akan menggunakan target fs={self.effective_fps}")

            self.processor = SignalProcessor(fs=self.effective_fps, buffer_size=SIGNAL_BUFFER_SIZE)
            self.plotter = RealtimePlotter(buffer_size=SIGNAL_BUFFER_SIZE) # visualization.py harus menampilkan 3 subplot (resp raw & filtered ditumpuk)
            self.face_detector_mp = FaceDetectorMP(model_selection=0)
            self.pose_tracker = PoseRespirationTracker(model_complexity=1)
            
            if self.plot_canvas_widget: self.plot_canvas_widget.destroy()
            self.plot_canvas_agg = FigureCanvasTkAgg(self.plotter.get_figure(), master=self.plot_display_frame)
            self.plot_canvas_agg.draw()
            self.plot_canvas_widget = self.plot_canvas_agg.get_tk_widget()
            self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            print("Processing components initialized.")
            return True
        except Exception as e:
            messagebox.showerror("Error Inisialisasi", f"Gagal menginisialisasi komponen: {e}")
            import traceback
            traceback.print_exc()
            print(f"Error in initialize_processing_components: {e}")
            return False


    def start_processing(self):
        print("Start processing called.")
        if not self.initialize_processing_components():
            print("Initialization failed. Cannot start processing.")
            return
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_custom_layout_button.config(state=tk.NORMAL) # Enable tombol simpan kustom
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        self.update_gui_fps_display()
        print("Processing thread started.")


    def stop_processing(self, called_on_exit=False):
        print("Stop processing called.")
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            print("Joining processing thread...")
            self.processing_thread.join(timeout=1.5) 
            print("Processing thread joined.")
        if self.video_stream:
            print("Releasing video stream...")
            self.video_stream.release(); self.video_stream = None
            print("Video stream released.")
        
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
            self.save_custom_layout_button.config(state=tk.DISABLED) # Disable tombol simpan kustom
        print("Pemrosesan dihentikan (GUI updated).")


    def _process_loop(self):
        # ... (Konten _process_loop tetap sama, pastikan semua data sinyal dikirim ke self.plotter.update_plots) ...
        print("Process loop started.")
        frame_count_proc_fps = 0
        start_time_proc_fps = time.time()
        current_processing_fps = 0.0
        
        r_signal_value, g_signal_value, b_signal_value = 0.0, 0.0, 0.0
        averaged_bpm, averaged_rpm = 0.0, 0.0
        raw_resp_motion_signal = 0.0 

        target_frame_duration = 1.0 / self.effective_fps
        frame_counter = 0
        
        while self.is_processing and self.video_stream:
            loop_start_time = time.time()
            frame_counter +=1

            ret, frame_original_bgr = self.video_stream.get_frame()
            if not ret or frame_original_bgr is None:
                print(f"Loop {frame_counter}: Failed to get frame or frame is None. Stopping. Ret: {ret}")
                if self.is_processing: self.after(0, lambda: messagebox.showerror("Stream Error", "Gagal mendapatkan frame atau frame kosong."))
                self.is_processing = False; break
            
            frame_original_rgb_mp = cv2.cvtColor(frame_original_bgr, cv2.COLOR_BGR2RGB)
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
                    r_signal_value = np.mean(face_roi_pixels[:, :, 2])
                    g_signal_value = np.mean(face_roi_pixels[:, :, 1])
                    b_signal_value = np.mean(face_roi_pixels[:, :, 0])
            
            filtered_rppg, bpm_current = self.processor.process_rppg(g_signal_value)
            raw_resp_motion_signal, pose_detected = self.pose_tracker.get_respiration_signal_and_draw_landmarks(
                frame_original_rgb_mp, processed_frame_for_drawing
            )
            filtered_resp, rpm_current = self.processor.process_respiration(raw_resp_motion_signal)
            
            current_bpm_to_average = bpm_current
            current_rpm_to_average = rpm_current

            if current_bpm_to_average > 0:
                self.bpm_history.append(current_bpm_to_average)
                if len(self.bpm_history) > self.rate_history_size: self.bpm_history.pop(0)
                if self.bpm_history: averaged_bpm = np.mean(self.bpm_history)
            elif not self.bpm_history: averaged_bpm = 0.0
            
            if current_rpm_to_average > 0:
                self.rpm_history.append(current_rpm_to_average)
                if len(self.rpm_history) > self.rate_history_size: self.rpm_history.pop(0)
                if self.rpm_history: averaged_rpm = np.mean(self.rpm_history)
            elif not self.rpm_history: averaged_rpm = 0.0

            frame_for_gui_display = self._prepare_frame_for_display(processed_frame_for_drawing)
            
            if self.winfo_exists():
                self.after(0, self._update_gui_data, frame_for_gui_display, averaged_bpm, averaged_rpm, current_processing_fps, raw_resp_motion_signal)

            if self.plotter and self.plot_canvas_agg and self.winfo_exists():
                rppg_plot_data_to_send = filtered_rppg if len(filtered_rppg) > 0 else self.processor.get_raw_rppg_signal_for_plot()
                resp_filtered_plot_data_to_send = filtered_resp if len(filtered_resp) > 0 else self.processor.get_raw_resp_signal_for_plot()
                
                self.plotter.update_plots(rppg_plot_data_to_send, 
                                          resp_filtered_plot_data_to_send,
                                          r_raw_value=r_signal_value,
                                          g_raw_value=g_signal_value,
                                          b_raw_value=b_signal_value,
                                          resp_raw_value=raw_resp_motion_signal) # Ini penting untuk tampilan GUI
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
        
        print("Process loop attempting to stop naturally or by flag.")
        if self.winfo_exists() and not self.is_processing:
            print("Scheduling stop_processing from _process_loop due to is_processing=False")
            self.after(0, self.stop_processing)
        print("Process loop ended.")


    # --- Metode save_plot_with_custom_layout BARU ---
    def save_plot_with_custom_layout(self):
        if not self.is_processing:
            messagebox.showwarning("Simpan Plot", "Pemrosesan tidak sedang berjalan. Tidak ada plot untuk disimpan.")
            return

        if self.plotter and hasattr(self.plotter, 'get_current_plot_data'):
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                current_data = self.plotter.get_current_plot_data()

                # Buat Figure BARU dengan 4 subplot untuk disimpan
                # (Jumlah subplot dan figsize mungkin perlu disesuaikan)
                fig_to_save, axs_to_save = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True) 
                fig_to_save.suptitle("Analisis Sinyal Fisiologis & Mentah (Disimpan)", fontsize=14, y=1.02)


                # 1. Plot Sinyal rPPG Terfilter
                axs_to_save[0].plot(current_data["rppg_filtered"], color='purple', label='rPPG Terfilter')
                axs_to_save[0].set_title("Sinyal rPPG Terfilter")
                axs_to_save[0].set_xlabel("Sampel")
                axs_to_save[0].set_ylabel("Amplitudo")
                axs_to_save[0].grid(True)
                axs_to_save[0].legend(loc='upper right')
                if np.any(current_data["rppg_filtered"]):
                    min_val, max_val = np.min(current_data["rppg_filtered"]), np.max(current_data["rppg_filtered"])
                    padding = 0.1 * max(abs(min_val), abs(max_val), 0.1) if max_val != min_val else 0.5
                    axs_to_save[0].set_ylim(min_val - padding, max_val + padding)
                else:
                    axs_to_save[0].set_ylim(-1,1)

                # 2. Plot Sinyal Respirasi Mentah (Sebelum Filter)
                axs_to_save[1].plot(current_data["resp_raw"], color='cyan', label='Respirasi Mentah')
                axs_to_save[1].set_title("Sinyal Respirasi Mentah (Sebelum Filter)")
                axs_to_save[1].set_xlabel("Sampel")
                axs_to_save[1].set_ylabel("Amplitudo Mentah")
                axs_to_save[1].grid(True)
                axs_to_save[1].legend(loc='upper right')
                if np.any(current_data["resp_raw"]):
                    min_val, max_val = np.min(current_data["resp_raw"]), np.max(current_data["resp_raw"])
                    padding_resp = 0.1 * max(abs(min_val), abs(max_val), 0.1) if max_val != min_val else 0.5
                    axs_to_save[1].set_ylim(min_val - padding_resp, max_val + padding_resp)
                else:
                     axs_to_save[1].set_ylim(np.min(current_data["resp_raw"]) -0.5 if np.any(current_data["resp_raw"]) else -1,
                                         np.max(current_data["resp_raw"]) + 0.5 if np.any(current_data["resp_raw"]) else 1)


                # 3. Plot Sinyal Respirasi Terfilter
                axs_to_save[2].plot(current_data["resp_filtered"], color='orange', label='Respirasi Terfilter')
                axs_to_save[2].set_title("Sinyal Respirasi Terfilter")
                axs_to_save[2].set_xlabel("Sampel")
                axs_to_save[2].set_ylabel("Amplitudo Terfilter")
                axs_to_save[2].grid(True)
                axs_to_save[2].legend(loc='upper right')
                if np.any(current_data["resp_filtered"]):
                    min_val, max_val = np.min(current_data["resp_filtered"]), np.max(current_data["resp_filtered"])
                    padding = 0.1 * max(abs(min_val), abs(max_val), 0.1) if max_val != min_val else 0.5
                    axs_to_save[2].set_ylim(min_val - padding, max_val + padding)
                else:
                    axs_to_save[2].set_ylim(-1,1)

                # 4. Plot Sinyal RGB Mentah
                axs_to_save[3].plot(current_data["rgb_r"], color='red', label='Merah (R)')
                axs_to_save[3].plot(current_data["rgb_g"], color='green', label='Hijau (G)')
                axs_to_save[3].plot(current_data["rgb_b"], color='blue', label='Biru (B)')
                axs_to_save[3].set_title("Sinyal RGB Mentah dari ROI Wajah")
                axs_to_save[3].set_xlabel("Sampel")
                axs_to_save[3].set_ylabel("Intensitas Rata-rata")
                axs_to_save[3].grid(True)
                axs_to_save[3].legend(loc='upper right')
                min_r, max_r = (np.min(current_data["rgb_r"]), np.max(current_data["rgb_r"])) if np.any(current_data["rgb_r"]) else (0,0)
                min_g, max_g = (np.min(current_data["rgb_g"]), np.max(current_data["rgb_g"])) if np.any(current_data["rgb_g"]) else (0,0)
                min_b, max_b = (np.min(current_data["rgb_b"]), np.max(current_data["rgb_b"])) if np.any(current_data["rgb_b"]) else (0,0)
                min_rgb_overall = min(min_r, min_g, min_b)
                max_rgb_overall = max(max_r, max_g, max_b)

                if max_rgb_overall > min_rgb_overall and max_rgb_overall > 0 :
                     padding_rgb = 10 
                     axs_to_save[3].set_ylim(max(0, min_rgb_overall - padding_rgb), min(255, max_rgb_overall + padding_rgb))
                else:
                     axs_to_save[3].set_ylim(0,256)

                # fig_to_save.tight_layout(rect=[0, 0.03, 1, 0.95]) # constrained_layout=True lebih disarankan

                filename = os.path.join(self.plot_save_path, f"separated_signals_plot_{timestamp}.png")
                fig_to_save.savefig(filename, dpi=150)
                plt.close(fig_to_save) 

                messagebox.showinfo("Simpan Plot", f"Plot dengan 4 subplot terpisah berhasil disimpan sebagai:\n{filename}")

            except Exception as e:
                messagebox.showerror("Simpan Plot Error", f"Gagal menyimpan plot kustom: {e}")
                import traceback
                traceback.print_exc()
                if 'fig_to_save' in locals() and plt.fignum_exists(fig_to_save.number):
                    plt.close(fig_to_save)
        else:
            messagebox.showerror("Simpan Plot Error", "Objek plotter tidak tersedia atau metode get_current_plot_data tidak ditemukan.")


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

        try:
            resized_content = cv2.resize(frame_to_display, (new_w, new_h))
        except Exception as e:
            print(f"Error resizing frame in _prepare_frame_for_display: {e}")
            return np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        
        full_sized_frame = np.full((target_h, target_w, 3), (128,128,128), dtype=np.uint8)
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        full_sized_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_content
        return full_sized_frame


    def _update_gui_data(self, frame_cv_display, bpm_to_display, rpm_to_display, proc_fps, raw_resp_signal_val):
        if not self.winfo_exists(): return
        if frame_cv_display is None:
            print("Error in _update_gui_data: frame_cv_display is None. Skipping update.")
            return
            
        try:
            img = cv2.cvtColor(frame_cv_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            self.imgtk_display_ref = ImageTk.PhotoImage(image=img_pil)
            
            if hasattr(self, 'video_label') and self.video_label.winfo_exists():
                 self.video_label.imgtk = self.imgtk_display_ref
                 self.video_label.config(image=self.imgtk_display_ref)
            else:
                print("Error: video_label not available or destroyed during GUI update.")

            self.bpm_label.config(text=f"BPM (rPPG): {bpm_to_display:.1f}")
            self.rpm_label.config(text=f"RPM (Resp): {rpm_to_display:.1f}")
            self.processing_fps_label.config(text=f"Processing FPS: {proc_fps:.2f}")
            if self.raw_resp_debug_label: self.raw_resp_debug_label.config(text=f"Raw Resp Motion: {raw_resp_signal_val:.4f}")
        except Exception as e:
            print(f"Error updating GUI data: {e}")
            import traceback
            traceback.print_exc()


    def update_gui_fps_display(self):
        if self.winfo_exists():
            self.frame_count_fps_calc +=1
            elapsed_time = time.time() - self.start_time_fps_calc
            if elapsed_time >= 1.0:
                current_gui_fps = self.frame_count_fps_calc / elapsed_time
                if self.winfo_exists(): self.gui_fps_label.config(text=f"GUI FPS: {current_gui_fps:.2f}")
                self.frame_count_fps_calc = 0
                self.start_time_fps_calc = time.time()
            
            if self.is_processing :
                 self.after(50, self.update_gui_fps_display)


    def on_closing(self):
        print("WM_DELETE_WINDOW called (on_closing).")
        if self.is_processing:
            user_choice = messagebox.askyesnocancel("Keluar", "Pemrosesan sedang berjalan. Hentikan dan keluar?")
            if user_choice is True:
                print("User chose Yes to stop and exit.")
                self.is_processing = False
                if self.processing_thread and self.processing_thread.is_alive():
                    print("Waiting for processing thread to join (on_closing)...")
                    self.processing_thread.join(timeout=1.0)
                    print("Processing thread joined (on_closing).")
                self._cleanup_resources()
                self.destroy()
            elif user_choice is False:
                print("User chose No (interpreted as exit now, cleanup and destroy).")
                self.is_processing = False 
                self._cleanup_resources()
                self.destroy()
            else: 
                print("User chose Cancel. Not exiting.")
                return
        else:
            print("Not processing. Cleaning up and destroying.")
            self._cleanup_resources()
            self.destroy()


    def _cleanup_resources(self):
        print("Cleaning up resources...")
        if self.video_stream and hasattr(self.video_stream, 'cap') and self.video_stream.cap and self.video_stream.cap.isOpened():
            print("Releasing video_stream in cleanup...")
            self.video_stream.release()
        self.video_stream = None
        
        if self.face_detector_mp: self.face_detector_mp.close()
        if self.pose_tracker: self.pose_tracker.close()      
        
        if self.plot_canvas_widget:
            print("Destroying plot_canvas_widget...")
            self.plot_canvas_widget.destroy()
            self.plot_canvas_widget = None
        self.plot_canvas_agg = None
        
        print("Closing all matplotlib figures...")
        plt.close('all')
        print("Resources cleaned up.")
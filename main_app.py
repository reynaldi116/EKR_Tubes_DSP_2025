import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

class CameraApp:
    def __init__(self, window, window_title="Real-time Camera Feed"):
        self.window = window
        self.window.title(window_title)

        # Inisialisasi kamera
        self.vid = cv2.VideoCapture(0) # 0 adalah indeks default untuk webcam
        if not self.vid.isOpened():
            raise ValueError("Tidak dapat membuka kamera. Pastikan kamera terhubung dan tidak sedang digunakan oleh aplikasi lain.")

        # Ambil lebar dan tinggi frame
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Buat Canvas Tkinter untuk menampilkan video
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Tombol untuk keluar
        self.btn_quit = tk.Button(window, text="Quit", width=10, command=self.quit_app)
        self.btn_quit.pack(pady=10)

        self.delay = 15 # milliseconds, untuk update frame
        self.running = True
        self.update_thread = threading.Thread(target=self.update_frame)
        self.update_thread.daemon = True # Agar thread berhenti ketika aplikasi utama berhenti
        self.update_thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.quit_app) # Menangani penutupan jendela

    def update_frame(self):
        while self.running:
            ret, frame = self.vid.read()
            if ret:
                # Konversi frame OpenCV (BGR) ke RGB
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                print("Gagal mengambil frame.")
                break
            time.sleep(self.delay / 1000.0) # Konversi delay ke detik

    def quit_app(self):
        self.running = False
        self.vid.release() # Lepaskan kamera
        self.window.destroy() # Tutup jendela Tkinter

# Jalankan aplikasi
if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
# video_capture.py
import cv2

class VideoCapture:
    def __init__(self, device_id=0):
        """
        Inisialisasi penangkap video dari perangkat kamera.

        Args:
            device_id (int): ID kamera (biasanya 0 untuk kamera bawaan/default).
        """
        self.cap = cv2.VideoCapture(device_id)  # Buka stream video dari kamera
        if not self.cap.isOpened():
            raise IOError("Tidak dapat membuka kamera.")  # Error jika kamera gagal dibuka
        
        # Ambil resolusi frame kamera
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ambil FPS kamera (frame per detik)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Kamera dibuka: {self.width}x{self.height} @ {self.fps} FPS")

    def get_frame(self):
        """
        Membaca satu frame dari kamera.

        Returns:
            tuple: (ret, frame)
                - ret (bool): True jika pembacaan frame berhasil
                - frame (np.array): Frame gambar dalam format BGR (OpenCV default)
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Melepaskan resource kamera saat tidak digunakan lagi."""
        self.cap.release()

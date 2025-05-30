# video_capture.py
import cv2

class VideoCapture:
    def __init__(self, device_id=0):
        """
        Inisialisasi penangkap video.
        Args:
            device_id (int): ID perangkat kamera (biasanya 0 untuk kamera default).
        """
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise IOError("Tidak dapat membuka kamera.")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Kamera dibuka: {self.width}x{self.height} @ {self.fps} FPS")

    def get_frame(self):
        """
        Membaca satu frame dari kamera.
        Returns:
            tuple: (ret, frame) dimana ret adalah boolean (True jika berhasil)
                   dan frame adalah gambar (NumPy array).
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Melepaskan perangkat kamera."""
        self.cap.release()
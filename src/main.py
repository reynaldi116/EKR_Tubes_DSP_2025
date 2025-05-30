# main.py
import gui # Impor modul gui yang berisi kelas AppGUI
from signal_processing import signal_processing # Untuk SIGNAL_BUFFER_SIZE, dll. jika ada konfigurasi global.
                         # Namun, gui.py sudah mengimpornya.
                         # Bisa dihapus jika tidak ada setup global dari main.py.

if __name__ == "__main__":
    """
    Titik masuk utama aplikasi Pengukuran Fisiologis.
    Script ini menginisialisasi dan menjalankan antarmuka pengguna grafis (GUI)
    yang didefinisikan dalam modul 'gui'.
    """
    print(f"Memulai aplikasi dari main.py...")
    print(f"Menggunakan SIGNAL_BUFFER_SIZE: {signal_processing.SIGNAL_BUFFER_SIZE}") # Contoh akses konstanta

    # Membuat instance dari AppGUI yang didefinisikan di gui.py
    app = gui.AppGUI()
    
    # Memulai event loop utama Tkinter.
    # Ini akan membuat GUI terlihat dan interaktif hingga window ditutup.
    app.mainloop()

    print("Aplikasi ditutup.")
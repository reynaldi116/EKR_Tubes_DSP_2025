# main.py

import gui  # Mengimpor modul 'gui' yang berisi kelas AppGUI untuk membuat GUI aplikasi
import signal_processing  # Mengimpor modul signal_processing yang berisi konfigurasi seperti SIGNAL_BUFFER_SIZE

if __name__ == "__main__":
    """
    Titik masuk utama aplikasi Pengukuran Fisiologis.
    Script ini bertugas menginisialisasi dan menjalankan GUI aplikasi.
    """
    
    print(f"Memulai aplikasi dari main.py...")  # Menandai awal eksekusi aplikasi di console
    
    # Menampilkan nilai konstanta SIGNAL_BUFFER_SIZE dari modul signal_processing
    # Ini sebagai contoh akses ke konfigurasi global dari modul lain
    print(f"Menggunakan SIGNAL_BUFFER_SIZE: {signal_processing.SIGNAL_BUFFER_SIZE}")
    
    # Membuat objek aplikasi GUI menggunakan kelas AppGUI dari modul gui
    app = gui.AppGUI()
    
    # Memulai event loop utama Tkinter
    # Ini akan menjalankan GUI dan membuat aplikasi tetap responsif sampai window ditutup
    app.mainloop()
    
    # Ketika GUI ditutup dan loop berhenti, cetak pesan penutupan aplikasi
    print("Aplikasi ditutup.")
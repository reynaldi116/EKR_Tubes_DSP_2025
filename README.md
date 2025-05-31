# 🤖 DSP-FINAL_PROJECT_2025
## Digital Signal Processing IF3024
## Course instructor = Martin Clinton Tosima Manullang, S.T., M.T.
## EN-Version
## Project Overview
This project is an application that integrates video capture, real-time audio signal processing, and interactive visualization within a graphical user interface (GUI). Developed as part of the Digital Signal Processing (DSP) course assignment, it demonstrates practical implementation of fundamental DSP concepts using the Python programming language. The main process involves capturing respiratory signals.

---

## 🛠️ Key Components

### 🔹 src/main.py
Manages the primary application logic and acts as the main controller connecting the GUI components.

### 🔹 src/gui.py
Contains the graphical user interface implementation using Python libraries, providing buttons and visual elements for user interaction.

### 🔹 src/video_capture.py
Responsible for capturing real-time video input from the camera to serve as the system’s visual input.

### 🔹 src/signal_processing.py
Implements a Butterworth bandpass filter to isolate relevant frequencies (heart rate ~0.75–4 Hz, respiration ~0.1–0.8 Hz), detrends signals via moving average to stabilize and remove drift, and computes frequency spectrum using FFT to identify dominant frequencies converted to BPM (heart rate) or RPM (respiration). It also buffers raw signals for continuous analysis.

### 🔹 src/utils.py
Utility functions supporting other modules, such as data formatting and helper routines

### 🔹 src/visualization.py
Handles the visualization of processed signals, displaying waveforms or frequency spectra

### 🔹 src/pose_respiration_tracker.py
Uses MediaPipe’s pose landmarks for left and right shoulders to calculate average vertical displacement, producing and visualizing respiratory signals based on landmark movement.

### 🔹 src/motion_tracker.py
Monitors movement of feature points within a specified region of interest (ROI) in the video to estimate respiratory signals based on vertical displacement of body motion. If feature points are insufficient or lost, re-detection ensures signal stability. Optical flow techniques detect subtle frame-to-frame movements for non-invasive respiratory pattern monitoring

---

## 🖥️ How to Run
1. Open a terminal and navigate to the project folder.
2. Clone this repository using the command:
   ```bash
   git clone http https://github.com/reynaldi116/EKR_Tubes_DSP_2025.git
   ```
3. Use Python version 3.12 or above. If unavailable, create and activate  a Conda environment:
   ```bash
   # To create a Conda environment:  
   conda create -n myenv python=3.10.6  
   conda activate myenv  
   pip install -r requirements.txt  
   ```
4. Run the application :
```bash
   cd src  
   python main_app.py  
```
5. The user interface will launch, allowing you to start real-time audio signal processing immediately.

## Real-Time Physiological Monitoring Application

## 💡 Main Features

### 🫀 Real-time Heart Rate (BPM) Estimation
- Uses **MediaPipe Face Detection** (based on the BlazeFace model via `.tflite` file) for accurate face detection.
- Implements the **POS (Plane-Orthogonal to Skin)** algorithm to extract rPPG signals from the average RGB color values in the facial Region of Interest (ROI).
- Applies signal filtering (Butterworth bandpass, detrending) and FFT analysis to compute BPM.

### 🌬️ Real-time Respiratory Rate (RPM) Estimation
- Utilizes **MediaPipe Pose Landmarker** to identify shoulder landmarks.
- Shoulder positions dynamically define the ROI in the chest/shoulder area.
- Applies **Optical Flow (Lucas-Kanade)** to track features within the ROI and analyze the average vertical motion as the raw respiratory signal.
- Filters the signal and uses FFT analysis to calculate RPM.

### 🖥️ Graphical User Interface (GUI)
- Built with **Tkinter** and `ttk` using custom themes and styling for a modern, responsive look.
- Displays real-time video feed with facial ROI and respiratory ROI/feature visualizations.
- Shows smoothed numeric values for BPM and RPM, updated periodically.
- Real-time plots of filtered rPPG and respiratory signals using embedded **Matplotlib**.
- Includes a status bar for operational feedback to the user.


---

## 👨‍💻 Contributors
Developed by EKR Team as final project of Digital Signal Processing course in 2025.

| Name             | ID Student       | Github Account                                                               |
| :-------------------------- | :-------- | :------------------------------------------------------------------------ |
| Reynaldi Cristian Simamora  | 122140116 | [reynaldi116](https://github.com/reynaldi116)                             |
| Eichal Elphindo Ginting     | 122140165 | [eichalelphindoginting](https://github.com/eichalelphindoginting)         |
| Muhammad Kaisar Teddy       | 122140058 | [Muhammad-Kaisar-Teddy](https://github.com/Muhammad-Kaisar-Teddy)         |

---
## Versi ID
## Matakuliah Pengolahan Sinyal Digital IF3024
## Dosen Pengampu : Martin Clinton Tosima Manullang, S.T., M.T. 

## 📖 Gambaran Umum Proyek
Proyek ini merupakan sebuah aplikasi yang mengintegrasikan proses pengambilan video, pemrosesan sinyal audio secara real-time, dan visualisasi interaktif dalam sebuah antarmuka grafis (GUI). Proyek ini dikembangkan sebagai bagian dari tugas mata kuliah Digital Signal Processing (DSP) dan menunjukkan penerapan nyata dari konsep-konsep dasar DSP menggunakan bahasa pemrograman Python. Untuk proses yang dilakukan adalah pengambilan sinyal respirasi, 

---

## 🛠️ Komponen Utama

### 🔹 src/main.py
Mengatur alur logika utama dan bertindak sebagai pengendali utama yang menghubungkan GUI.

### 🔹 src/gui.py
Berisi implementasi antarmuka pengguna grafis (GUI) menggunakan pustaka Python. Menyediakan tombol dan tampilan visual untuk berinteraksi dengan aplikasi.

### 🔹 src/camera/video_capture.py
Bertanggung jawab untuk menangkap video dari kamera secara real-time sebagai input visual bagi sistem.

### 🔹 src/signal_processing/signal_processing.py
Penerapan filter bandpass Butterworth untuk memisahkan frekuensi relevan (detak jantung ~0.75–4 Hz, respirasi ~0.1–0.8 Hz), menghilangkan tren (detrending) menggunakan moving average agar sinyal lebih stabil dan bebas drift, dan perhitungan spektrum frekuensi dengan FFT untuk menemukan frekuensi dominan yang kemudian dikonversi ke BPM (detak jantung) atau RPM (pernapasan), menyimpan buffer sinyal mentah untuk analisis berkelanjutan.

### 🔹 src/utils/utils.py
Fungsi-fungsi utilitas pendukung modul lain, seperti pemformatan data dan fungsi bantu lainnya.

### 🔹 src/visualization.py
Bertugas menampilkan hasil pemrosesan sinyal dalam bentuk visual, seperti gelombang suara atau spektrum frekuensi.

### 🔹 src/motion_tracker.py
Memantau perpindahan titik-titik fitur pada area tertentu (ROI) di video untuk mengestimasi sinyal pernapasan berdasarkan perubahan posisi vertikal gerakan tubuh. Bila titik fitur terlalu sedikit atau tidak dapat terlacak dengan baik, fitur akan dideteksi ulang untuk menjaga kestabilan sinyal. Teknik optical flow membantu mendeteksi pergerakan halus dari frame ke frame, sehingga bisa digunakan untuk memantau pola gerakan pernapasan secara non-invasif.

### 🔹 src/pose_respiration_tracker.py
Penerapan mediapipe untuk landmark_pose bahu kiri dan kanan, melakukan perhitungan perubahan vertikal rata-rata bahu, dan mengaplikasikan serta menghasilkan sinyal pernapasan dalam bentuk landmark tervisualisasi

---

##  Cara Menjalankan Aplikasi

1. Buka terminal dan arahkan ke folder proyek.
2. Memanggil atau menduplikat repositori ini dengan memanggil command
   ```bash
   git clone http https://github.com/reynaldi116/EKR_Tubes_DSP_2025.git
   ```
3. Menggunakan python versi 3.12 atau ke atas. Jika tidak gunakan environment conda
   ```bash
   "Untuk membuat Environment di conda menggunakan command berikut : "
   conda create -n myenv python=3.10.6
   conda activate myenv
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi dengan perintah:
   ```bash
   cd src
   python main_app.py
   ```
5. Antarmuka pengguna akan muncul dan Anda dapat langsung memulai pemrosesan sinyal audio secara real-time.
---

## Aplikasi Pemantauan Fisiologis Real-Time

##  Fitur Utama

###  Estimasi Denyut Jantung (BPM) Real-time
- Menggunakan **MediaPipe Face Detection** (berdasarkan model BlazeFace melalui file `.tflite`) untuk deteksi wajah yang akurat.
- Menerapkan algoritma **POS (Plane-Orthogonal to Skin)** untuk mengekstraksi sinyal rPPG dari nilai warna RGB rata-rata di Region of Interest (ROI) wajah.
- Menerapkan penyaringan sinyal (Butterworth bandpass, detrending) dan analisis FFT untuk menghitung BPM.
- 
###  Estimasi Laju Pernapasan (RPM) Real-time
- Menggunakan **MediaPipe Pose Landmarker** untuk mengidentifikasi titik acuan bahu.
- Posisi bahu secara dinamis menentukan ROI di area dada/bahu.
- Menerapkan **Optical Flow (Lucas-Kanade)** untuk melacak fitur dalam ROI dan menganalisis gerakan vertikal rata-rata sebagai sinyal pernapasan mentah.
- Memfilter sinyal dan menggunakan analisis FFT untuk menghitung RPM.
  
###  Antarmuka Pengguna Grafis (GUI)
- Dibuat dengan **Tkinter** dan `ttk` menggunakan tema dan gaya khusus untuk tampilan yang modern dan responsif.
- Menampilkan umpan video waktu nyata dengan ROI wajah dan ROI pernapasan/visualisasi fitur.
- Menampilkan nilai numerik yang dihaluskan untuk BPM dan RPM, diperbarui secara berkala.
- Plot waktu nyata dari rPPG yang difilter dan sinyal pernapasan menggunakan **Matplotlib** yang tertanam.
- Menyertakan bilah status untuk umpan balik operasional kepada pengguna.
---

## 👨‍💻 Kontributor
Dikembangkan oleh EKR Team sebagai Tugas Besar mata kuliah Digital Signal Processing tahun 2025.

| Nama Lengkap                | NIM       | Akun GitHub                                                               |
| :-------------------------- | :-------- | :------------------------------------------------------------------------ |
| Reynaldi Cristian Simamora  | 122140116 | [reynaldi116](https://github.com/reynaldi116)                             |
| Eichal Elphindo Ginting     | 122140165 | [eichalelphindoginting](https://github.com/eichalelphindoginting)         |
| Muhammad Kaisar Teddy       | 122140058 | [Muhammad-Kaisar-Teddy](https://github.com/Muhammad-Kaisar-Teddy)         |

## Logbook Kegiatan Proyek

| No         | Tanggal          | Aktivitas/Progress                                                                                                                                                            |
| :--------- | :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1          | 08/05/2025     | Pembentukan anggota tim.																			|
| 2          | 11/05/2025    | Pembuatan Repositori Github pengumpulan tugas, dan penambahan kontributor													|
| 3          | 25/05/2025   | Diskusi Awal Pengerjaaan																			|
| 4          | 29/05/2025     | Pengimplementasian video real time awal, dan riset metode estimasi respirai dan detak jantung											|
| 5          | 30/05/2025    | Pembentukan sinyal, dan pengembangan model respirasi dan rppg dengan model terkait, dan pemantapan tampilan GUI								|
| 6          | 31/05/2025     | Pemantapan keseluruhan kode program pembuatan laporan report															|

---


import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import sounddevice as sd
from gtts import gTTS
import os
import threading
from scipy.io.wavfile import write
import time


class SpeechModel:
    def __init__(self):
        self.model = self.create_model()
        
    def create_model(self):
        model = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            
            # Flatten and Dense layers
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(self.get_classes()))
        )
        return model
    
    def get_classes(self):
        return {
            0: "เพลง",
            1: "เสียงพูด",
            2: "เสียงดนตรี",
            3: "เสียงรบกวน"
        }
    
    def preprocess_audio(self, waveform, sample_rate):
        # ตั้งค่าพารามิเตอร์
        n_mels = 64
        n_fft = 1024
        hop_length = 512
        
        # สร้าง Mel Spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # แปลงเป็น Mel Spectrogram
        with torch.no_grad():
            mel_spec = mel_transform(waveform)
            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
            # Normalize
            mel_spec = (mel_spec + 80) / 80
            
        return mel_spec

class SpeechRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("โปรแกรมวิเคราะห์ไฟล์เสียง")
        self.root.geometry("800x600")
        
        # ตั้งค่าพารามิเตอร์
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        
        # สร้างโฟลเดอร์สำหรับเก็บไฟล์
        self.setup_folders()
        
        # ตั้งค่า GUI
        self.setup_gui()
        self.speech_model = SpeechModel()
    def setup_folders(self):
        self.output_folder = "output"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def setup_gui(self):
        # สร้าง Frame หลัก
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # สร้างสไตล์
        style = ttk.Style()
        style.configure('Custom.TButton', font=('Helvetica', 12), padding=10)
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Status.TLabel', font=('Helvetica', 12))

        # ส่วนบน - ชื่อโปรแกรม
        title_label = ttk.Label(
            main_frame,
            text="โปรแกรมวิเคราะห์ไฟล์เสียง",
            style='Title.TLabel'
        )
        title_label.pack(pady=10)

        # ส่วนควบคุม
        control_frame = ttk.LabelFrame(main_frame, text="ควบคุม", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # ปุ่มเลือกไฟล์
        file_button = ttk.Button(
            control_frame,
            text="เลือกไฟล์เสียง",
            style='Custom.TButton',
            command=self.choose_file
        )
        file_button.pack(side=tk.LEFT, padx=5)

        # แสดงชื่อไฟล์ปัจจุบัน
        self.current_file_label = ttk.Label(
            control_frame,
            text="ยังไม่ได้เลือกไฟล์",
            style='Status.TLabel'
        )
        self.current_file_label.pack(side=tk.LEFT, padx=20)

        # ส่วนแสดงข้อมูลไฟล์
        info_frame = ttk.LabelFrame(main_frame, text="ข้อมูลไฟล์เสียง", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # สร้าง Grid สำหรับข้อมูล
        self.info_labels = {}
        info_fields = [
            ("duration", "ความยาว"),
            ("duration_fmt", "ความยาว (นาที:วินาที)"),
            ("channels", "จำนวนช่องเสียง"),
            ("sample_rate", "Sample Rate"),
            ("file_size", "ขนาดไฟล์")
        ]
        
        for i, (key, label) in enumerate(info_fields):
            ttk.Label(info_frame, text=f"{label}:", style='Status.TLabel').grid(
                row=i, column=0, sticky='w', padx=5, pady=2
            )
            self.info_labels[key] = ttk.Label(info_frame, text="-", style='Status.TLabel')
            self.info_labels[key].grid(row=i, column=1, sticky='w', padx=5, pady=2)

        # ส่วนแสดงผลการวิเคราะห์
        analysis_frame = ttk.LabelFrame(main_frame, text="ผลการวิเคราะห์", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Text widget สำหรับแสดงผล
        self.result_text = tk.Text(
            analysis_frame,
            height=10,
            width=50,
            font=('Helvetica', 12)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(analysis_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

    def format_duration(self, seconds):
        minutes = int(seconds // 60)
        seconds_remaining = seconds % 60
        return f"{minutes:02d}:{seconds_remaining:05.2f}"

    def format_file_size(self, size_in_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024
        return f"{size_in_bytes:.2f} TB"

    def choose_file(self):
        file_path = filedialog.askopenfilename(
            title="เลือกไฟล์เสียง",
            filetypes=(
                ("MP3 files", "*.mp3"),
                ("WAV files", "*.wav"),
                ("All audio files", "*.wav *.mp3"),
                ("all files", "*.*")
            )
        )
        
        if file_path:
            self.process_audio_file(file_path)

    def process_audio_file(self, file_path):
        try:
            # อ่านข้อมูลไฟล์
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # โหลดไฟล์เสียง
            waveform, sample_rate = torchaudio.load(file_path)
            
            # คำนวณข้อมูล
            duration = len(waveform[0]) / sample_rate
            channels = waveform.shape[0]
            self.analyze_audio(waveform, sample_rate)
            
            # อัพเดตการแสดงผล
            self.current_file_label.configure(text=file_name)
            
            # อัพเดตข้อมูล
            self.info_labels["duration"].configure(text=f"{duration:.2f} วินาที")
            self.info_labels["duration_fmt"].configure(text=self.format_duration(duration))
            self.info_labels["channels"].configure(text=str(channels))
            self.info_labels["sample_rate"].configure(text=f"{sample_rate} Hz")
            self.info_labels["file_size"].configure(text=self.format_file_size(file_size))
            
            # เพิ่มข้อมูลในส่วนผลการวิเคราะห์
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"วิเคราะห์ไฟล์: {file_name}\n")
            self.result_text.insert(tk.END, f"ประเภทไฟล์: {os.path.splitext(file_name)[1]}\n")
            self.result_text.insert(tk.END, f"เวลาที่วิเคราะห์: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # TODO: เพิ่มการวิเคราะห์เสียงเพิ่มเติมตรงนี้
            
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
    
    def analyze_audio(self, waveform, sample_rate):
        # แบ่งเสียงเป็นช่วงๆ ละ 5 วินาที
        segment_length = 5 * sample_rate
        num_segments = len(waveform[0]) // segment_length
        
        self.result_text.insert(tk.END, "\nผลการวิเคราะห์แต่ละช่วง:\n")
        
        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = waveform[:, start:end]
            
            # Preprocess
            mel_spec = self.speech_model.preprocess_audio(segment, sample_rate)
            
            # เพิ่มมิติ batch
            mel_spec = mel_spec.unsqueeze(0)
            
            # ทำนาย
            with torch.no_grad():
                outputs = self.speech_model.model(mel_spec)
                predicted = torch.argmax(outputs, dim=1)
                
                # แปลงผลลัพธ์
                class_name = self.speech_model.get_classes()[predicted.item()]
                
                # คำนวณความมั่นใจ
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item() * 100
                
                # แสดงผล
                start_time = self.format_duration(i * 5)
                end_time = self.format_duration((i + 1) * 5)
                self.result_text.insert(tk.END, 
                    f"ช่วง {start_time} - {end_time}: {class_name} "
                    f"(ความมั่นใจ {confidence:.1f}%)\n"
                )
        
        # วิเคราะห์เพิ่มเติม
        self.analyze_audio_features(waveform, sample_rate)
    
    def analyze_audio_features(self, waveform, sample_rate):
        self.result_text.insert(tk.END, "\nการวิเคราะห์คุณลักษณะเสียง:\n")
        
        # วิเคราะห์ความดัง
        amplitude = waveform.abs().mean().item()
        self.result_text.insert(tk.END, f"ความดังเฉลี่ย: {amplitude:.3f}\n")
        
        # วิเคราะห์ช่วงความถี่
        mel_spec = self.speech_model.preprocess_audio(waveform, sample_rate)
        freq_range = {
            "ต่ำ (20-250 Hz)": mel_spec[:, :12].mean().item(),
            "กลาง (250-2000 Hz)": mel_spec[:, 12:40].mean().item(),
            "สูง (2000-20000 Hz)": mel_spec[:, 40:].mean().item()
        }
        
        self.result_text.insert(tk.END, "การกระจายความถี่:\n")
        for range_name, value in freq_range.items():
            self.result_text.insert(tk.END, f"- {range_name}: {value:.3f}\n")

def main():
    root = tk.Tk()
    app = SpeechRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
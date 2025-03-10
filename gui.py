import tkinter as tk
from tkinter import filedialog
import customtkinter
from tkinter import messagebox
from PIL import Image

import sys
import os


sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from super_resolution import SuperResolution
from config.config import CONFIG
from models.EDSR.edsr import EDSR

class UpscalerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Kiri-2x')
        self.root.geometry('700x500')

        self.root.grid_columnconfigure(0, weight=1)  
        self.root.grid_columnconfigure(1, weight=1) 

        self.main_label = customtkinter.CTkLabel(root, text='Kiri-2x', font=('monospace', 18, 'bold'))
        self.main_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.label = customtkinter.CTkLabel(root, text='Select Image or Video:')
        self.label.grid(row=1, column=0, padx=20, pady=5, sticky="w")

        self.file_path_label = customtkinter.CTkLabel(root, text='No file selected', wraplength=250)
        self.file_path_label.grid(row=2, column=0, padx=20, pady=5, sticky="n")

        self.select_btn = customtkinter.CTkButton(root, text='Browse', command=self.browse_file)
        self.select_btn.grid(row=3, column=0, padx=20, pady=5, sticky="n")

        self.preview_label = customtkinter.CTkLabel(root, text="Preview", width=200, height=200)
        self.preview_label.grid(row=4, column=0, padx=20, pady=5, sticky="n")

        self.save_label = customtkinter.CTkLabel(root, text='Select Save Path:')
        self.save_label.grid(row=1, column=1, padx=20, pady=5, sticky="n")

        self.save_path = customtkinter.CTkLabel(root, text='No save path selected', wraplength=250)
        self.save_path.grid(row=2, column=1, padx=20, pady=5, sticky="n")

        self.save_path_button = customtkinter.CTkButton(root, text='Browse Save Path', command=self.select_save_dir)
        self.save_path_button.grid(row=3, column=1, padx=20, pady=5, sticky="n")

        self.start_btn = customtkinter.CTkButton(root, text="Start Upscaling", state=customtkinter.DISABLED, command=self.start_upscaling)
        self.start_btn.grid(row=5, column=1, padx=20, pady=20, sticky="e")  

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image & video Files", "*.png;*.jpg;*.jpeg;*.mp4;*.mkv;*.avi")])
        if file_path:
            self.file_path_label.configure(text=file_path)
            self.start_btn.configure(state=tk.NORMAL)

            if file_path.endswith(('.png', '.jpg', '.jpeg')):
                self.show_image_preview(file_path)

    def show_image_preview(self, file_path):
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_ctk = customtkinter.CTkImage(light_image=img, dark_image=img, size=(200, 200))

        self.preview_label.configure(image=img_ctk)
        self.preview_label.image = img_ctk 

    def select_save_dir(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.save_path.configure(text=folder_path)
            return folder_path
    
    def start_upscaling(self):
        file_path = self.file_path_label.cget('text')
        save_path = self.save_path.cget('text')

        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            save_path = self.upscale_image(file_path, save_path)
        
        messagebox.showinfo('Upscaling Complete', 'Upscaling complete. Upscaled image' + save_path)
        self.root.quit()

    def upscale_image(self, file_path, save_path):
        model = EDSR().to(CONFIG.DEVICE)  
        save_path = os.path.join(save_path, os.path.splitext(os.path.basename(file_path))[0] + '_2x.png')
        sr = SuperResolution('weights/edsr.pth', model, CONFIG.DEVICE)
        sr.upscale(file_path, save_path)
        return save_path 

if __name__ == "__main__":
    root = customtkinter.CTk()
    app = UpscalerGUI(root)
    root.mainloop()

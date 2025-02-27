import os
import torch
import glob
import subprocess
from PIL import Image
import torchvision.transforms.functional as TF

from config.config import CONFIG
from models.EDSR.edsr import EDSR

class VideoUpscaler:
    def __init__(self, video_path, frames_dir='videos/frames', upscaled_frames_dir='videos/upscaled_frames', output_video='videos/upscaled_video.mp4'):
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.upscaled_frames_dir = upscaled_frames_dir
        self.output_video = output_video
        self.model = self.load_model()
        
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.upscaled_frames_dir, exist_ok=True)
    
    def load_model(self):
        model = EDSR().to(CONFIG.DEVICE)
        model.load_state_dict(torch.load('weights/edsr.pth'))
        model.eval()
        return model
    
    def extract_frames(self):
        ffmpeg_cmd = [
            "ffmpeg", "-i", self.video_path,
            "-vf", "fps=30",
            os.path.join(self.frames_dir, "frame_%04d.png")
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Frames extracted to {self.frames_dir}")
    
    def upscale_frames(self):
        frames = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
        
        for frame in frames:
            img = Image.open(os.path.join(self.frames_dir, frame)).convert('RGB')
            img = TF.to_tensor(img).unsqueeze(0).to(CONFIG.DEVICE)

            with torch.no_grad():
                sr_tensor = self.model(img).clamp(0, 1)
            
            sr_img = TF.to_pil_image(sr_tensor.squeeze(0))
            sr_img.save(os.path.join(self.upscaled_frames_dir, frame))
        print("Frames upscaled")
    
    def generate_video(self):
        ffmpeg_video = [
            "ffmpeg", "-framerate", "30",
            "-i", os.path.join(self.upscaled_frames_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            self.output_video
        ]
        subprocess.run(ffmpeg_video, check=True)
        print("Upscaled video generated")
    
    def cleanup(self):
        pattern = 'frame_*.png'
        for file_path in glob.glob(os.path.join(self.frames_dir, pattern)):
            os.remove(file_path)
        print("Original frames deleted")

        for file_path in glob.glob(os.path.join(self.upscaled_frames_dir, pattern)):
            os.remove(file_path)
        print("Upscaled frames deleted")
    
    def process(self):
        self.extract_frames()
        self.upscale_frames()
        self.generate_video()
        self.cleanup()

if __name__ == "__main__":
    video_upscaler = VideoUpscaler('videos/aot.mp4')
    video_upscaler.process()

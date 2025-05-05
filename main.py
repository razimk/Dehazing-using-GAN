import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import numpy as np
import threading
import os

class DehazeApp:
    def __init__(self, root):
        # Initialize main window
        self.root = root
        self.root.title("DeHaze Pro")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Create header
        header_frame = tk.Frame(root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="DeHaze Pro", font=("Arial", 24, "bold"), bg="#2c3e50", fg="white")
        title_label.pack(pady=20)
        
        # Create main content area
        content_frame = tk.Frame(root, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Options frame (left side)
        options_frame = tk.Frame(content_frame, bg="#e0e0e0", width=300)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Style for buttons
        self.button_style = {
            "font": ("Arial", 12),
            "bg": "#3498db",
            "fg": "white",
            "activebackground": "#2980b9",
            "activeforeground": "white",
            "width": 25,
            "height": 2,
            "bd": 0,
            "relief": tk.FLAT
        }
        
        # Option buttons
        option_label = tk.Label(options_frame, text="Select Mode", font=("Arial", 16, "bold"), bg="#e0e0e0")
        option_label.pack(pady=(20, 30))
        
        self.image_btn = tk.Button(options_frame, text="Image Dehaze", command=self.image_dehaze_mode, **self.button_style)
        self.image_btn.pack(pady=10)
        
        self.video_btn = tk.Button(options_frame, text="Video Dehaze", command=self.video_dehaze_mode, **self.button_style)
        self.video_btn.pack(pady=10)
        
        self.realtime_btn = tk.Button(options_frame, text="Realtime Dehaze", command=self.realtime_dehaze_mode, **self.button_style)
        self.realtime_btn.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(options_frame, text="Ready", font=("Arial", 10), bg="#e0e0e0", fg="#333")
        self.status_label.pack(pady=(50, 10))
        
        # Display frame (right side)
        self.display_frame = tk.Frame(content_frame, bg="#ffffff")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Welcome message
        welcome_label = tk.Label(self.display_frame, text="Welcome to DeHaze Pro", font=("Arial", 20, "bold"), bg="#ffffff")
        welcome_label.pack(pady=(100, 20))
        
        instruction_label = tk.Label(self.display_frame, text="Select a mode from the options panel to get started", font=("Arial", 12), bg="#ffffff")
        instruction_label.pack()
        
        # Video capture variables
        self.cap = None
        self.is_capturing = False
        self.video_thread = None
        
        # Progress bar for processing
        self.progress = ttk.Progressbar(options_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress.pack(pady=(20, 10))
        
        # Additional controls frame (initially hidden)
        self.controls_frame = tk.Frame(options_frame, bg="#e0e0e0")
        self.controls_frame.pack(pady=20, fill=tk.X)
        
        # Initially hide controls
        self.controls_frame.pack_forget()
        
        # Now that the UI is set up, load the model
        self.load_model()
        
    def load_model(self):
        # Load generator model (define the model architecture first)
        try:
            self.generator = Generator().to(self.device)
            self.generator.load_state_dict(torch.load('netG_finetuned_final.pth', map_location=self.device, weights_only=True))
            self.generator.eval()
            self.update_status("Model loaded successfully")
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}")
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def clear_display(self):
        # Clear the display frame
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        # Stop any ongoing video capture
        self.stop_capture()
        
        # Hide controls
        self.controls_frame.pack_forget()
    
    def image_dehaze_mode(self):
        self.clear_display()
        self.update_status("Image Dehaze Mode")
        
        # Create image mode layout
        header = tk.Label(self.display_frame, text="Image Dehazing", font=("Arial", 18, "bold"), bg="#ffffff")
        header.pack(pady=(20, 30))
        
        # Create frames for original and dehazed images
        images_frame = tk.Frame(self.display_frame, bg="#ffffff")
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        self.original_frame = tk.LabelFrame(images_frame, text="Original Image", bg="#ffffff", font=("Arial", 12))
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.original_label = tk.Label(self.original_frame, bg="#f5f5f5")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.dehazed_frame = tk.LabelFrame(images_frame, text="Dehazed Image", bg="#ffffff", font=("Arial", 12))
        self.dehazed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.dehazed_label = tk.Label(self.dehazed_frame, bg="#f5f5f5")
        self.dehazed_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add button to browse for image
        self.controls_frame.pack(pady=20, fill=tk.X)
        
        browse_btn = tk.Button(self.controls_frame, text="Browse Image", command=self.browse_image, 
                             font=("Arial", 12), bg="#27ae60", fg="white", width=15, height=1)
        browse_btn.pack(pady=5)
        
        save_btn = tk.Button(self.controls_frame, text="Save Result", command=self.save_dehazed_image,
                           font=("Arial", 12), bg="#e74c3c", fg="white", width=15, height=1)
        save_btn.pack(pady=5)
        save_btn.config(state=tk.DISABLED)  # Initially disabled
        self.save_btn = save_btn
    
    def video_dehaze_mode(self):
        self.clear_display()
        self.update_status("Video Dehaze Mode")
        
        header = tk.Label(self.display_frame, text="Video Dehazing", font=("Arial", 18, "bold"), bg="#ffffff")
        header.pack(pady=(20, 30))
        
        # Create frames for original and dehazed videos
        videos_frame = tk.Frame(self.display_frame, bg="#ffffff")
        videos_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholders for video frames
        self.original_video_frame = tk.LabelFrame(videos_frame, text="Original Video", bg="#ffffff", font=("Arial", 12))
        self.original_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.original_video_label = tk.Label(self.original_video_frame, bg="#f5f5f5")
        self.original_video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.dehazed_video_frame = tk.LabelFrame(videos_frame, text="Dehazed Video", bg="#ffffff", font=("Arial", 12))
        self.dehazed_video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.dehazed_video_label = tk.Label(self.dehazed_video_frame, bg="#f5f5f5")
        self.dehazed_video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add controls
        self.controls_frame.pack(pady=20, fill=tk.X)
        
        browse_video_btn = tk.Button(self.controls_frame, text="Browse Video", command=self.browse_video,
                                   font=("Arial", 12), bg="#27ae60", fg="white", width=15, height=1)
        browse_video_btn.pack(pady=5)
        
        self.play_btn = tk.Button(self.controls_frame, text="Play", command=self.play_video,
                              font=("Arial", 12), bg="#3498db", fg="white", width=15, height=1)
        self.play_btn.pack(pady=5)
        self.play_btn.config(state=tk.DISABLED)  # Initially disabled
        
        self.stop_btn = tk.Button(self.controls_frame, text="Stop", command=self.stop_video,
                              font=("Arial", 12), bg="#e74c3c", fg="white", width=15, height=1)
        self.stop_btn.pack(pady=5)
        self.stop_btn.config(state=tk.DISABLED)  # Initially disabled
        
        save_video_btn = tk.Button(self.controls_frame, text="Save Result", command=self.save_dehazed_video,
                                 font=("Arial", 12), bg="#f39c12", fg="white", width=15, height=1)
        save_video_btn.pack(pady=5)
        save_video_btn.config(state=tk.DISABLED)  # Initially disabled
        self.save_video_btn = save_video_btn
    
    def realtime_dehaze_mode(self):
        self.clear_display()
        self.update_status("Realtime Dehaze Mode")
        
        header = tk.Label(self.display_frame, text="Realtime Dehazing", font=("Arial", 18, "bold"), bg="#ffffff")
        header.pack(pady=(20, 30))
        
        # Create frames for original and dehazed webcam feeds
        feeds_frame = tk.Frame(self.display_frame, bg="#ffffff")
        feeds_frame.pack(fill=tk.BOTH, expand=True)
        
        self.webcam_frame = tk.LabelFrame(feeds_frame, text="Webcam Feed", bg="#ffffff", font=("Arial", 12))
        self.webcam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.webcam_label = tk.Label(self.webcam_frame, bg="#f5f5f5")
        self.webcam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.realtime_dehazed_frame = tk.LabelFrame(feeds_frame, text="Dehazed Feed", bg="#ffffff", font=("Arial", 12))
        self.realtime_dehazed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.realtime_dehazed_label = tk.Label(self.realtime_dehazed_frame, bg="#f5f5f5")
        self.realtime_dehazed_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add controls
        self.controls_frame.pack(pady=20, fill=tk.X)
        
        self.start_webcam_btn = tk.Button(self.controls_frame, text="Start Webcam", command=self.start_webcam,
                                       font=("Arial", 12), bg="#27ae60", fg="white", width=15, height=1)
        self.start_webcam_btn.pack(pady=5)
        
        self.stop_webcam_btn = tk.Button(self.controls_frame, text="Stop Webcam", command=self.stop_webcam,
                                      font=("Arial", 12), bg="#e74c3c", fg="white", width=15, height=1)
        self.stop_webcam_btn.pack(pady=5)
        self.stop_webcam_btn.config(state=tk.DISABLED)  # Initially disabled
        
        self.screenshot_btn = tk.Button(self.controls_frame, text="Take Screenshot", command=self.take_screenshot,
                                     font=("Arial", 12), bg="#f39c12", fg="white", width=15, height=1)
        self.screenshot_btn.pack(pady=5)
        self.screenshot_btn.config(state=tk.DISABLED)  # Initially disabled
    
    def browse_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        
        if not image_path:
            return
        
        self.update_status("Processing image...")
        self.progress.start()
        
        try:
            # Load and display original image
            original_img = Image.open(image_path)
            self.original_pil = original_img.copy()
            
            # Resize for display while maintaining aspect ratio
            display_img = self.resize_for_display(original_img, 400)
            photo = ImageTk.PhotoImage(display_img)
            
            self.original_label.config(image=photo)
            self.original_label.image = photo  # Keep a reference
            
            # Process the image through the generator
            input_tensor = self.preprocess(original_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_tensor = self.generator(input_tensor)
            
            # Denormalize and convert tensor to image
            output_tensor = self.denormalize(output_tensor.squeeze(0)).cpu()
            output_img = transforms.ToPILImage()(output_tensor)
            self.dehazed_pil = output_img.copy()
            
            # Resize for display
            display_output = self.resize_for_display(output_img, 400)
            output_photo = ImageTk.PhotoImage(display_output)
            
            self.dehazed_label.config(image=output_photo)
            self.dehazed_label.image = output_photo  # Keep a reference
            
            self.save_btn.config(state=tk.NORMAL)  # Enable save button
            self.update_status("Image processed successfully")
            
        except Exception as e:
            self.update_status(f"Error processing image: {str(e)}")
        finally:
            self.progress.stop()
    
    def save_dehazed_image(self):
        if not hasattr(self, 'dehazed_pil'):
            self.update_status("No processed image to save")
            return
        
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        save_path = filedialog.asksaveasfilename(title="Save Dehazed Image", 
                                               defaultextension=".png", 
                                               filetypes=filetypes)
        
        if not save_path:
            return
        
        try:
            self.dehazed_pil.save(save_path)
            self.update_status(f"Image saved to {save_path}")
        except Exception as e:
            self.update_status(f"Error saving image: {str(e)}")
    
    def browse_video(self):
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        self.video_path = filedialog.askopenfilename(title="Select Video", filetypes=filetypes)
        
        if not self.video_path:
            return
        
        self.update_status(f"Video selected: {os.path.basename(self.video_path)}")
        self.play_btn.config(state=tk.NORMAL)  # Enable play button
    
    def play_video(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            self.update_status("No video selected")
            return
        
        # Disable play button, enable stop button
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_video_btn.config(state=tk.NORMAL)
        
        # Start video processing in a separate thread
        self.is_capturing = True
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.update_status("Error: Could not open video")
                return
            
            self.update_status("Processing video...")
            
            while self.is_capturing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert OpenCV BGR frame to PIL RGB image for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Resize for display
                display_img = self.resize_for_display(pil_img, 400)
                
                # Prepare tensor for processing
                input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                # Process through generator
                with torch.no_grad():
                    output_tensor = self.generator(input_tensor)
                
                # Denormalize and convert to image
                output_tensor = self.denormalize(output_tensor.squeeze(0)).cpu()
                output_img = transforms.ToPILImage()(output_tensor)
                
                # Resize for display
                display_output = self.resize_for_display(output_img, 400)
                
                # Update UI (must be done in main thread)
                self.root.after(0, lambda: self.update_video_display(display_img, display_output))
                
                # Limit frame rate
                cv2.waitKey(30)
            
            cap.release()
            self.update_status("Video processing stopped")
            
        except Exception as e:
            self.update_status(f"Error processing video: {str(e)}")
        finally:
            # Re-enable play button, disable stop button in main thread
            self.root.after(0, lambda: self.reset_video_controls())
    
    def update_video_display(self, original_frame, dehazed_frame):
        # Convert PIL images to Tkinter PhotoImages
        original_tk = ImageTk.PhotoImage(original_frame)
        dehazed_tk = ImageTk.PhotoImage(dehazed_frame)
        
        # Update labels
        self.original_video_label.config(image=original_tk)
        self.original_video_label.image = original_tk  # Keep a reference
        
        self.dehazed_video_label.config(image=dehazed_tk)
        self.dehazed_video_label.image = dehazed_tk  # Keep a reference
    
    def reset_video_controls(self):
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def stop_video(self):
        self.is_capturing = False
        if self.video_thread:
            self.video_thread.join(1.0)  # Wait for thread to finish
        self.update_status("Video processing stopped")
        self.reset_video_controls()
    
    def save_dehazed_video(self):
        # This would involve recording the dehazed frames to a new video file
        self.update_status("Video saving not implemented in this demo")
    
    def start_webcam(self):
        # Disable start button, enable stop and screenshot buttons
        self.start_webcam_btn.config(state=tk.DISABLED)
        self.stop_webcam_btn.config(state=tk.NORMAL)
        self.screenshot_btn.config(state=tk.NORMAL)
        
        # Start webcam in a separate thread
        self.is_capturing = True
        self.video_thread = threading.Thread(target=self.process_webcam)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def process_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)  # Use default webcam
            if not self.cap.isOpened():
                self.update_status("Error: Could not open webcam")
                return
            
            self.update_status("Webcam started")
            
            while self.is_capturing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Convert OpenCV BGR frame to PIL RGB image for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Resize for display
                display_img = self.resize_for_display(pil_img, 400)
                
                # Save current frame for potential screenshot
                self.current_frame = pil_img.copy()
                
                # Prepare tensor for processing
                input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                # Process through generator
                with torch.no_grad():
                    output_tensor = self.generator(input_tensor)
                
                # Denormalize and convert to image
                output_tensor = self.denormalize(output_tensor.squeeze(0)).cpu()
                output_img = transforms.ToPILImage()(output_tensor)
                
                # Save current dehazed frame for potential screenshot
                self.current_dehazed = output_img.copy()
                
                # Resize for display
                display_output = self.resize_for_display(output_img, 400)
                
                # Update UI (must be done in main thread)
                self.root.after(0, lambda: self.update_webcam_display(display_img, display_output))
                
                # Limit frame rate
                cv2.waitKey(30)
            
            if self.cap:
                self.cap.release()
            self.update_status("Webcam stopped")
            
        except Exception as e:
            self.update_status(f"Error processing webcam: {str(e)}")
        finally:
            # Reset controls in main thread
            self.root.after(0, lambda: self.reset_webcam_controls())
    
    def update_webcam_display(self, original_frame, dehazed_frame):
        # Convert PIL images to Tkinter PhotoImages
        original_tk = ImageTk.PhotoImage(original_frame)
        dehazed_tk = ImageTk.PhotoImage(dehazed_frame)
        
        # Update labels
        self.webcam_label.config(image=original_tk)
        self.webcam_label.image = original_tk  # Keep a reference
        
        self.realtime_dehazed_label.config(image=dehazed_tk)
        self.realtime_dehazed_label.image = dehazed_tk  # Keep a reference
    
    def reset_webcam_controls(self):
        self.start_webcam_btn.config(state=tk.NORMAL)
        self.stop_webcam_btn.config(state=tk.DISABLED)
        self.screenshot_btn.config(state=tk.DISABLED)
    
    def stop_webcam(self):
        self.is_capturing = False
        self.update_status("Stopping webcam...")
    
    def stop_capture(self):
        # Stop any ongoing video capture
        self.is_capturing = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def take_screenshot(self):
        if not hasattr(self, 'current_frame') or not hasattr(self, 'current_dehazed'):
            self.update_status("No frames to capture")
            return
        
        try:
            # Create a screenshots directory if it doesn't exist
            if not os.path.exists("screenshots"):
                os.makedirs("screenshots")
            
            # Generate unique filenames based on timestamp
            import time
            timestamp = int(time.time())
            
            # Save both original and dehazed frames
            original_path = f"screenshots/original_{timestamp}.png"
            dehazed_path = f"screenshots/dehazed_{timestamp}.png"
            
            self.current_frame.save(original_path)
            self.current_dehazed.save(dehazed_path)
            
            self.update_status(f"Screenshots saved to screenshots folder")
        except Exception as e:
            self.update_status(f"Error saving screenshots: {str(e)}")
    
    def resize_for_display(self, pil_img, target_height):
        # Resize image to target height while maintaining aspect ratio
        w, h = pil_img.size
        ratio = target_height / h
        new_size = (int(w * ratio), target_height)
        return pil_img.resize(new_size, Image.LANCZOS)
    
    def denormalize(self, tensor):
        # Convert tensor from [-1, 1] to [0, 1]
        tensor = tensor * 0.5 + 0.5
        return tensor.clamp(0, 1)

# Define the Generator class
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.Tanh()
        )
        
    def forward(self, x):
        return self.main(x)

# Start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DehazeApp(root)
    root.mainloop()
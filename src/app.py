import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageOps
import os
import pandas as pd
import numpy as np
import threading
import tempfile
from facenet_pytorch import InceptionResnetV1

# Import backend functions
from config import IDENTIFICATION_THRESHOLD, VERIFICATION_THRESHOLD, TEST_PATH, GALLERY_PATH
from identification import identify_probe_open_set, process_probe_image
from verification import verify_claim
from sample_utils import load_gallery

class BioSysApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BioSys Face Recognition Prototype")
        self.root.geometry("1000x800")
        
        # Load backend resources
        self.status_var = tk.StringVar(value="Loading models and gallery...")
        self.gallery = None
        self.model = None
        self.classes_map = {}
        self.temp_image_path = None
        
        # Camera management
        self.cap = None
        self.camera_active = False

        # UI Setup
        self._setup_ui()
        
        # Start background loading
        threading.Thread(target=self._load_resources, daemon=True).start()
        
    def _load_resources(self):
        try:
            # Load Gallery
            self.gallery = load_gallery()
            
            # Load Model
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Load Names Mapping
            try:
                df = pd.read_csv("data/name_to_id.csv")
                self.classes_map = dict(zip(df.class_id, df.celebrity_name))
            except Exception as e:
                print(f"Could not load name mapping: {e}")
                # Fallback to class IDs if CSV missing
                if self.gallery:
                    unique_labels = np.unique(self.gallery['labels'])
                    for lbl in unique_labels:
                        self.classes_map[lbl] = f"Person {lbl}"
            
            # Update UI
            self.root.after(0, lambda: self.status_var.set("System Ready"))
            self.root.after(0, self._update_class_dropdown)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error loading resources: {str(e)}"))
            print(e)

    def _update_class_dropdown(self):
        sorted_names = [f"{k}: {v}" for k, v in sorted(self.classes_map.items())]
        self.class_combo['values'] = sorted_names
        if sorted_names:
            self.class_combo.current(0)

    def _setup_ui(self):
        # Main Layout: Left (Controls), Right (Preview & Results)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # --- Left Panel ---
        
        # Mode Selection
        mode_frame = ttk.LabelFrame(left_panel, text="Operation Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="identification")
        ttk.Radiobutton(mode_frame, text="Identification (1:N)", variable=self.mode_var, 
                       value="identification", command=self._on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Verification (1:1)", variable=self.mode_var, 
                       value="verification", command=self._on_mode_change).pack(anchor=tk.W)
        
        # Input Source
        source_frame = ttk.LabelFrame(left_panel, text="Input Source", padding="10")
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.source_var = tk.StringVar(value="webcam")
        ttk.Radiobutton(source_frame, text="Webcam", variable=self.source_var, 
                       value="webcam", command=self._on_source_change).pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="File Image", variable=self.source_var, 
                       value="file", command=self._on_source_change).pack(anchor=tk.W)
        
        # Action Buttons
        self.btn_frame = ttk.Frame(left_panel)
        self.btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_capture = ttk.Button(self.btn_frame, text="Start Camera", command=self._toggle_camera)
        self.btn_capture.pack(fill=tk.X, pady=2)
        
        self.btn_snap = ttk.Button(self.btn_frame, text="Take Snapshot", command=self._take_snapshot, state=tk.DISABLED)
        self.btn_snap.pack(fill=tk.X, pady=2)
        
        self.btn_browse = ttk.Button(self.btn_frame, text="Browse Probe...", command=self._browse_file)
        # Packed later based on mode
        
        # Verification Specific Controls
        self.verif_frame = ttk.LabelFrame(left_panel, text="Verification Details", padding="10")
        # Packed later based on mode
        
        ttk.Label(self.verif_frame, text="Claimed Identity:").pack(anchor=tk.W)
        self.class_combo = ttk.Combobox(self.verif_frame, state="readonly")
        self.class_combo.pack(fill=tk.X, pady=5)
        
        # Process Button
        self.btn_process = ttk.Button(left_panel, text="PROCESS RESULT", command=self._process_result, state=tk.DISABLED)
        self.btn_process.pack(fill=tk.X, pady=20)
        
        # --- Right Panel ---
        
        # Image Preview
        self.preview_label = ttk.Label(right_panel, text="No Image", background="black", foreground="white", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status Bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Results Display
        self.result_frame = ttk.LabelFrame(right_panel, text="Results", padding="10", height=200)
        self.result_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        self.result_text = tk.Text(self.result_frame, height=8, width=50)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Initial UI State
        self._on_source_change()
        self._on_mode_change()

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "verification":
            self.verif_frame.pack(fill=tk.X, pady=(0, 10), after=self.btn_frame)
        else:
            self.verif_frame.pack_forget()
            
    def _on_source_change(self):
        source = self.source_var.get()
        # Reset camera
        if self.camera_active:
            self._toggle_camera()
            
        if source == "webcam":
            self.btn_capture.pack(fill=tk.X, pady=2)
            self.btn_snap.pack(fill=tk.X, pady=2)
            self.btn_browse.pack_forget()
            self.preview_label.configure(text="Camera Off - Press Start Camera")
        else:
            self.btn_capture.pack_forget()
            self.btn_snap.pack_forget()
            self.btn_browse.pack(fill=tk.X, pady=2)
            self.preview_label.configure(text="No File Selected")

    def _toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            if self.cap:
                self.cap.release()
            self.btn_capture.configure(text="Start Camera")
            self.btn_snap.configure(state=tk.DISABLED)
            self.preview_label.configure(image='', text="Camera Stopped")
        else:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open webcam")
                    return
                self.camera_active = True
                self.btn_capture.configure(text="Stop Camera")
                self.btn_snap.configure(state=tk.NORMAL)
                self._update_feed()
            except Exception as e:
                messagebox.showerror("Error", f"Camera error: {e}")

    def _update_feed(self):
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB and then to ImageTk
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = frame_rgb # Keep for saving
                img = Image.fromarray(frame_rgb)
                
                # Resize for preview fitting
                display_w = self.preview_label.winfo_width()
                display_h = self.preview_label.winfo_height()
                if display_w > 1 and display_h > 1:
                    img.thumbnail((display_w, display_h))
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.preview_label.imgtk = imgtk
                self.preview_label.configure(image=imgtk, text="")
                
            self.root.after(20, self._update_feed)

    def _take_snapshot(self):
        if hasattr(self, 'current_frame'):
            # Stop camera feed to "freeze"
            self._toggle_camera()
            
            # Save frame to temporary file
            img = Image.fromarray(self.current_frame)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            self.temp_image_path = temp_file.name
            temp_file.close()
            img.save(self.temp_image_path)
            
            # Update preview with static image
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
            
            self.status_var.set("Snapshot taken. Ready to process.")
            self.btn_process.configure(state=tk.NORMAL)

    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=TEST_PATH,
            title="Select Probe Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            # Copy/Save as temp for consistency in processing
            try:
                img = Image.open(file_path)
                
                # Correct orientation from EXIF
                img = ImageOps.exif_transpose(img)
                
                img = img.convert('RGB') # Ensure RGB
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                self.temp_image_path = temp_file.name
                temp_file.close()
                img.save(self.temp_image_path)
                
                # Display
                display_w = self.preview_label.winfo_width()
                display_h = self.preview_label.winfo_height()
                if display_w > 1 and display_h > 1:
                    img_preview = img.copy()
                    img_preview.thumbnail((display_w, display_h))
                    imgtk = ImageTk.PhotoImage(image=img_preview)
                    self.preview_label.imgtk = imgtk
                    self.preview_label.configure(image=imgtk, text="")
                
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                self.btn_process.configure(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def _process_result(self):
        if not self.model or not self.gallery:
            messagebox.showwarning("Wait", "System is still loading resources...")
            return

        self.status_var.set("Processing...")
        self.root.update()

        try:
            mode = self.mode_var.get()
            
            if mode == "identification":
                self._run_identification()
            else:
                self._run_verification()
            
            # Clean up temporary file after processing
            self._cleanup_temp_file()
                
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            self.status_var.set("Error during processing")
            # Clean up even on error
            self._cleanup_temp_file()

    def _run_identification(self):
        # 1. Get embedding
        probe_emb = process_probe_image(self.temp_image_path, self.model)
        
        if probe_emb is None:
            self._display_result("Error: No face detected in the image.")
            return

        # 2. Identify
        result = identify_probe_open_set(
            probe_emb, self.gallery, 
            threshold=IDENTIFICATION_THRESHOLD,
            top_k=5
        )
        
        # 3. Format Output
        output = []
        output.append(f"--- IDENTIFICATION RESULT ---")
        output.append(f"Threshold: {IDENTIFICATION_THRESHOLD}")
        
        predicted_id = result['predicted_label']
        is_rejected = result['rejected']
        min_dist = result['min_distance']
        
        name = self.classes_map.get(predicted_id, f"Unknown ID {predicted_id}")
        
        if is_rejected:
            output.append(f"\nRESULT: UNKNOWN IDENTITY")
            output.append(f"(Best match: {name} at dist {min_dist:.4f} > Threshold)")
        else:
            output.append(f"\nRESULT: IDENTIFIED as {name}")
            output.append(f"Distance: {min_dist:.4f}")
        
        output.append(f"\nTop 5 Matches:")
        for idx in range(len(result['ranked_labels'])):
            lbl = result['ranked_labels'][idx]
            dist = result['distances'][idx]
            pname = self.classes_map.get(lbl, str(lbl))
            output.append(f" {idx+1}. {pname} (Dist: {dist:.4f})")
            
        self._display_result("\n".join(output))

    def _run_verification(self):
        # 1. Get Claim
        claim_str = self.class_combo.get()
        if not claim_str:
            messagebox.showwarning("Missing Input", "Please select a claimed identity.")
            return
            
        # Parse "ID: Name" format
        claimed_id = int(claim_str.split(":")[0])
        
        # 2. Verify
        # verify_claim takes a path, so we use self.temp_image_path
        res = verify_claim(
            self.temp_image_path, 
            claimed_id, 
            threshold=VERIFICATION_THRESHOLD, 
            gallery_path=GALLERY_PATH,
            model=self.model
        )
        
        # 3. Format Output
        output = []
        output.append(f"--- VERIFICATION RESULT ---")
        output.append(f"Claiming Identity: {self.classes_map.get(claimed_id, claimed_id)}")
        output.append(f"Threshold: {VERIFICATION_THRESHOLD}")
        
        if res.get('error'):
             output.append(f"\nError: {res['error']}")
        else:
            dist = res['distance']
            accepted = res['accepted']
            
            output.append(f"Computed Distance: {dist:.4f}")
            output.append(f"\nDECISION: {'ACCEPTED ✅' if accepted else 'REJECTED ❌'}")
            
        self._display_result("\n".join(output))

    def _cleanup_temp_file(self):
        """Delete temporary image file after processing."""
        if self.temp_image_path and os.path.exists(self.temp_image_path):
            try:
                os.remove(self.temp_image_path)
                self.temp_image_path = None
            except Exception as e:
                print(f"Could not delete temp file: {e}")

    def _display_result(self, text):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.status_var.set("Processing Complete")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        # Set icon if available, or basic theme
        style = ttk.Style()
        style.theme_use('clam')
        
        app = BioSysApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application crash: {e}")

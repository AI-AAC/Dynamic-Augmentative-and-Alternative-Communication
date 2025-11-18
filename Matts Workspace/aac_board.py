"""
AAC Board Generator with Image-to-Text and Text-to-Board workflow.

Features:
- Upload an image or type text description
- Generate AAC board using BLIP (image captioning) and sentence transformers (text matching)
- Adjustable specificity slider (10-30 items)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from PIL import Image, ImageTk
import requests
from io import BytesIO
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import os


class AACBoard:
    def __init__(self, csv_path):
        """
        Initialize the AAC board interface.
        
        Args:
            csv_path: Path to the arasaac_synset_mapping CSV file
        """
        self.csv_path = csv_path
        
        # Load CSV
        print("Loading CSV data...")
        self.df = pd.read_csv(csv_path)
        self.df = self.df.drop_duplicates(subset=['primary_keyword'])
        
        # Initialize models
        print("Loading models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # BLIP for image captioning
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.blip_loaded = True
        except Exception as e:
            print(f"Warning: Could not load BLIP model: {e}")
            self.blip_loaded = False
        
        # Sentence transformer for text matching
        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.sentence_model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model_loaded = False
        
        # Pre-compute keyword embeddings (store as numpy arrays for efficiency)
        if self.sentence_model_loaded:
            print("Pre-computing keyword embeddings...")
            keywords = self.df["primary_keyword"].tolist()
            # Compute embeddings in batches for efficiency
            embeddings = self.sentence_model.encode(keywords, show_progress_bar=True, convert_to_numpy=True)
            self.df["keyword_emb"] = [emb for emb in embeddings]
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("AAC Board Generator")
        self.root.geometry("1000x700")
        
        # Variables
        self.current_synset_ids = []
        self.synset_map = {}
        self.input_mode = tk.StringVar(value="text")
        self.specificity = tk.IntVar(value=20)
        self.current_image_path = None
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface."""
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Input mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Input Type", padding="10")
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(mode_frame, text="Text Description", variable=self.input_mode, 
                       value="text", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Upload Image", variable=self.input_mode, 
                       value="image", command=self.on_mode_change).pack(side=tk.LEFT, padx=5)
        
        # Text input
        self.text_frame = ttk.LabelFrame(control_frame, text="Text Description", padding="10")
        self.text_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        self.text_entry = tk.Text(self.text_frame, height=3, width=40)
        self.text_entry.pack(fill=tk.BOTH, expand=True)
        
        # Image upload button
        self.image_frame = ttk.LabelFrame(control_frame, text="Image Upload", padding="10")
        self.image_button = ttk.Button(self.image_frame, text="Select Image", 
                                       command=self.select_image)
        self.image_button.pack()
        self.image_label = ttk.Label(self.image_frame, text="No image selected")
        self.image_label.pack()
        
        # Specificity slider
        slider_frame = ttk.LabelFrame(control_frame, text="Specificity", padding="10")
        slider_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(slider_frame, text="Items:").pack()
        self.slider = ttk.Scale(slider_frame, from_=10, to=30, variable=self.specificity, 
                               orient=tk.HORIZONTAL, length=150)
        self.slider.pack()
        self.slider_label = ttk.Label(slider_frame, text="20")
        self.slider_label.pack()
        self.specificity.trace('w', lambda *args: self.slider_label.config(text=str(self.specificity.get())))
        
        # Generate button
        self.generate_button = ttk.Button(control_frame, text="Generate Board", 
                                         command=self.generate_board)
        self.generate_button.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Main frame for board
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel
        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Initial mode
        self.on_mode_change()
    
    def on_mode_change(self):
        """Handle input mode change."""
        if self.input_mode.get() == "text":
            self.text_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
            self.image_frame.pack_forget()
        else:
            self.text_frame.pack_forget()
            self.image_frame.pack(side=tk.LEFT, padx=5)
    
    def select_image(self):
        """Open file dialog to select image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.image_label.config(text=os.path.basename(file_path))
    
    def caption_image(self, image_path):
        """Generate caption from image using BLIP."""
        if not self.blip_loaded:
            raise Exception("BLIP model not loaded")
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=30, num_beams=5, repetition_penalty=1.15)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True).strip()
        return caption
    
    def text_to_synsets(self, text, top_n=20):
        """Convert text to ranked synset IDs using sentence transformers."""
        if not self.sentence_model_loaded:
            raise Exception("Sentence transformer model not loaded")
        
        # Encode input text
        text_emb = self.sentence_model.encode(text, convert_to_numpy=True)
        
        # Calculate similarity scores (cosine similarity)
        self.df["score"] = self.df["keyword_emb"].apply(
            lambda emb: np.dot(text_emb, emb) / (np.linalg.norm(text_emb) * np.linalg.norm(emb)))
        
        # Sort by score and get top N
        df_sorted = self.df.sort_values("score", ascending=False)
        top_results = df_sorted.head(top_n)
        
        # Extract unique synsets (prefer highest scoring)
        synset_ids = []
        seen_synsets = set()
        for _, row in top_results.iterrows():
            synset = row['synset']
            if synset not in seen_synsets:
                synset_ids.append(synset)
                seen_synsets.add(synset)
                if len(synset_ids) >= top_n:
                    break
        
        return synset_ids
    
    def generate_board(self):
        """Generate AAC board from input."""
        try:
            self.status_label.config(text="Generating...")
            self.root.update()
            
            # Get text input
            if self.input_mode.get() == "image":
                if not self.current_image_path:
                    messagebox.showwarning("No Image", "Please select an image first.")
                    return
                
                if not self.blip_loaded:
                    messagebox.showerror("Error", "BLIP model not available. Please install transformers.")
                    return
                
                # Generate caption from image
                text = self.caption_image(self.current_image_path)
                print(f"Generated caption: {text}")
            else:
                text = self.text_entry.get("1.0", tk.END).strip()
                if not text:
                    messagebox.showwarning("No Text", "Please enter a text description.")
                    return
            
            if not self.sentence_model_loaded:
                messagebox.showerror("Error", "Sentence transformer model not available.")
                return
            
            # Convert text to synset IDs
            top_n = int(self.specificity.get())
            synset_ids = self.text_to_synsets(text, top_n)
            
            if not synset_ids:
                messagebox.showinfo("No Results", "No matching symbols found.")
                return
            
            # Update board
            self.current_synset_ids = synset_ids
            self.update_board()
            
            self.status_label.config(text=f"Generated {len(synset_ids)} symbols")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate board: {str(e)}")
            self.status_label.config(text="Error")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_board(self):
        """Update the board display with current synset IDs."""
        # Clear existing buttons
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Build synset map
        self.synset_map = {}
        for synset in self.current_synset_ids:
            matches = self.df[self.df['synset'] == synset]
            if not matches.empty:
                row = matches.iloc[0]
                self.synset_map[synset] = {
                    'image_url': row['image_url'],
                    'keyword': row['primary_keyword'],
                    'pictogram_id': row['pictogram_id']
                }
        
        # Load and display buttons
        self.load_buttons(self.scrollable_frame)
    
    def load_image_from_url(self, url):
        """Load image from URL and resize for button."""
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content))
            img = img.resize((100, 100), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            # Return a placeholder
            img = Image.new('RGB', (100, 100), color='gray')
            return ImageTk.PhotoImage(img)
    
    def load_buttons(self, parent):
        """Load and display buttons for each synset."""
        row = 0
        col = 0
        cols_per_row = 4
        
        for synset_id in self.current_synset_ids:
            if synset_id not in self.synset_map:
                print(f"Warning: Synset {synset_id} not found in CSV")
                continue
            
            info = self.synset_map[synset_id]
            
            # Create button frame
            btn_frame = ttk.Frame(parent, padding="5")
            btn_frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Load image
            photo = self.load_image_from_url(info['image_url'])
            
            # Create button
            btn = tk.Button(
                btn_frame,
                image=photo,
                text=info['keyword'],
                compound=tk.TOP,
                width=120,
                height=130,
                command=lambda s=synset_id: self.on_button_click(s)
            )
            btn.image = photo  # Keep a reference
            btn.pack()
            
            # Update grid position
            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1
    
    def on_button_click(self, synset_id):
        """Handle button click."""
        info = self.synset_map.get(synset_id, {})
        keyword = info.get('keyword', synset_id)
        print(f"Clicked: {keyword} ({synset_id})")
        # You can extend this to handle speech synthesis, etc.
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()


if __name__ == "__main__":
    csv_path = "../Kaitlin's Workspace/arasaac_synset_mapping_20251106_130530.csv"
    
    board = AACBoard(csv_path)
    board.run()

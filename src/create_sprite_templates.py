import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw
import json
import os

class PixelAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Template Annotator")
        
        # Data structures
        self.original_image = None
        self.image_array = None
        self.pixel_categories = {}  # (x, y): category
        self.current_category = "skin"
        self.zoom_level = 20  # pixels per sprite pixel
        self.drawing = False
        self.filename = None
        
        # Categories and colors
        self.categories = {
            "skin": "#FFB6C1",
            "hair": "#DDA0DD",
            "top": "#87CEEB",
            "bottom": "#98FB98",
            "outline": "#FFD700"
        }
        
        # Undo stack
        self.undo_stack = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # File operations
        tk.Button(control_frame, text="Load Sprite", command=self.load_image).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Save Template", command=self.save_template).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Load Template", command=self.load_template).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=2)
        
        # Separator
        tk.Frame(control_frame, width=2, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Category selection
        tk.Label(control_frame, text="Category:").pack(side=tk.LEFT, padx=5)
        
        self.category_var = tk.StringVar(value="skin")
        for cat, color in self.categories.items():
            rb = tk.Radiobutton(control_frame, text=cat.capitalize(), variable=self.category_var, 
                               value=cat, bg=color, selectcolor=color, 
                               command=self.change_category)
            rb.pack(side=tk.LEFT, padx=2)
        
        # Eraser
        tk.Radiobutton(control_frame, text="Eraser", variable=self.category_var, 
                      value="eraser", command=self.change_category).pack(side=tk.LEFT, padx=2)
        
        # Main content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for sprite display
        self.canvas = tk.Canvas(content_frame, bg="white", cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Info panel
        info_frame = tk.Frame(content_frame, width=200)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        info_frame.pack_propagate(False)
        
        tk.Label(info_frame, text="Instructions:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        instructions = """
1. Load a sprite image
2. Select a category
3. Click or drag to paint pixels
4. Use Eraser to remove
5. Save when done

Categories:
- Skin/Hair/Top/Bottom: 
  Body parts (shading preserved)
- Outline: Fixed dark borders

Keyboard shortcuts:
1 = Skin
2 = Hair  
3 = Top
4 = Bottom
5 = Outline
E = Eraser
Ctrl+Z = Undo
        """
        tk.Label(info_frame, text=instructions, justify=tk.LEFT, font=("Arial", 9)).pack(anchor=tk.W)
        
        # Stats
        self.stats_label = tk.Label(info_frame, text="", justify=tk.LEFT, font=("Arial", 9))
        self.stats_label.pack(anchor=tk.W, pady=10)
        
        # Test colors frame
        tk.Label(info_frame, text="Test Colors:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.test_colors = {
            "skin": "#FFC0A8",
            "hair": "#8B4513", 
            "top": "#FF0000",
            "bottom": "#0000FF"
        }
        
        for cat in ["skin", "hair", "top", "bottom"]:
            frame = tk.Frame(info_frame)
            frame.pack(anchor=tk.W, pady=2)
            tk.Label(frame, text=f"{cat.capitalize()}:", width=8, anchor=tk.W).pack(side=tk.LEFT)
            btn = tk.Button(frame, bg=self.test_colors[cat], width=3, 
                          command=lambda c=cat: self.pick_test_color(c))
            btn.pack(side=tk.LEFT)
            
        tk.Button(info_frame, text="Preview with Test Colors", 
                 command=self.preview_with_colors).pack(pady=10, fill=tk.X)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Bind keyboard shortcuts
        self.root.bind("1", lambda e: self.set_category("skin"))
        self.root.bind("2", lambda e: self.set_category("hair"))
        self.root.bind("3", lambda e: self.set_category("top"))
        self.root.bind("4", lambda e: self.set_category("bottom"))
        self.root.bind("5", lambda e: self.set_category("outline"))
        self.root.bind("e", lambda e: self.set_category("eraser"))
        self.root.bind("<Control-z>", lambda e: self.undo())
        
    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Select Sprite Image",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            self.filename = os.path.splitext(os.path.basename(filepath))[0]
            self.original_image = Image.open(filepath).convert("RGBA")
            self.pixel_categories = {}
            self.undo_stack = []
            self.draw_canvas()
            self.update_stats()
            
    def draw_canvas(self):
        if self.original_image is None:
            return
            
        width, height = self.original_image.size
        canvas_width = width * self.zoom_level
        canvas_height = height * self.zoom_level
        
        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas.delete("all")
        
        # Draw sprite pixels
        for y in range(height):
            for x in range(width):
                pixel = self.original_image.getpixel((x, y))
                if pixel[3] > 0:  # If not fully transparent
                    x1 = x * self.zoom_level
                    y1 = y * self.zoom_level
                    x2 = x1 + self.zoom_level
                    y2 = y1 + self.zoom_level
                    
                    # Original pixel color
                    color = f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                    
        # Draw category overlays
        for (px, py), cat in self.pixel_categories.items():
            x1 = px * self.zoom_level
            y1 = py * self.zoom_level
            x2 = x1 + self.zoom_level
            y2 = y1 + self.zoom_level
            overlay_color = self.categories[cat]
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=overlay_color, 
                                        stipple="gray50", outline="black", width=1)
            
    def get_pixel_coords(self, event):
        if self.original_image is None:
            return None
        x = event.x // self.zoom_level
        y = event.y // self.zoom_level
        width, height = self.original_image.size
        if 0 <= x < width and 0 <= y < height:
            return (x, y)
        return None
        
    def on_mouse_down(self, event):
        self.drawing = True
        self.paint_pixel(event)
        
    def on_mouse_drag(self, event):
        if self.drawing:
            self.paint_pixel(event)
            
    def on_mouse_up(self, event):
        self.drawing = False
        
    def paint_pixel(self, event):
        coords = self.get_pixel_coords(event)
        if coords is None:
            return
            
        # Check if pixel exists (not transparent)
        pixel = self.original_image.getpixel(coords)
        if pixel[3] == 0:  # Fully transparent
            return
            
        # Save state for undo
        old_state = self.pixel_categories.copy()
        
        if self.current_category == "eraser":
            if coords in self.pixel_categories:
                del self.pixel_categories[coords]
                self.undo_stack.append(old_state)
        else:
            self.pixel_categories[coords] = self.current_category
            self.undo_stack.append(old_state)
            
        self.draw_canvas()
        self.update_stats()
        
    def change_category(self):
        self.current_category = self.category_var.get()
        
    def set_category(self, cat):
        self.category_var.set(cat)
        self.current_category = cat
        
    def undo(self):
        if self.undo_stack:
            self.pixel_categories = self.undo_stack.pop()
            self.draw_canvas()
            self.update_stats()
            
    def clear_all(self):
        if messagebox.askyesno("Clear All", "Remove all category assignments?"):
            self.undo_stack.append(self.pixel_categories.copy())
            self.pixel_categories = {}
            self.draw_canvas()
            self.update_stats()
            
    def update_stats(self):
        if self.original_image is None:
            self.stats_label.config(text="No image loaded")
            return
            
        counts = {"skin": 0, "hair": 0, "top": 0, "bottom": 0, "outline": 0}
        for cat in self.pixel_categories.values():
            counts[cat] += 1
            
        stats_text = "Pixel counts:\n"
        for cat, count in counts.items():
            stats_text += f"{cat.capitalize()}: {count}\n"
        stats_text += f"\nTotal: {sum(counts.values())}"
        self.stats_label.config(text=stats_text)
        
    def save_template(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
            
        if not self.pixel_categories:
            messagebox.showwarning("No Data", "Please assign some pixels first")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{self.filename}_template.json" if self.filename else "template.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            # Store original pixel colors along with categories
            template_data = {
                "width": self.original_image.size[0],
                "height": self.original_image.size[1],
                "pixels": []
            }
            
            for (x, y), cat in self.pixel_categories.items():
                pixel = self.original_image.getpixel((x, y))
                template_data["pixels"].append({
                    "x": x,
                    "y": y,
                    "category": cat,
                    "original_color": {
                        "r": pixel[0],
                        "g": pixel[1],
                        "b": pixel[2]
                    }
                })
            
            with open(filepath, 'w') as f:
                json.dump(template_data, f, indent=2)
                
            messagebox.showinfo("Saved", f"Template saved to {filepath}")
            
    def load_template(self):
        filepath = filedialog.askopenfilename(
            title="Select Template File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            with open(filepath, 'r') as f:
                template_data = json.load(f)
                
            self.pixel_categories = {
                (p["x"], p["y"]): p["category"]
                for p in template_data["pixels"]
            }
            
            self.draw_canvas()
            self.update_stats()
            messagebox.showinfo("Loaded", "Template loaded successfully")
            
    def pick_test_color(self, category):
        color = colorchooser.askcolor(initialcolor=self.test_colors[category])
        if color[1]:  # If user didn't cancel
            self.test_colors[category] = color[1]
            
    def preview_with_colors(self):
        if self.original_image is None or not self.pixel_categories:
            messagebox.showwarning("No Data", "Load an image and assign pixels first")
            return
            
        # Create new window with preview
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Color Preview")
        
        # Create preview image using the renderer logic
        preview_img = self.render_sprite(self.test_colors)
        
        # Scale up for viewing
        width, height = preview_img.size
        display_img = preview_img.resize((width * self.zoom_level, height * self.zoom_level), 
                                        Image.NEAREST)
        photo = ImageTk.PhotoImage(display_img)
        
        label = tk.Label(preview_window, image=photo)
        label.image = photo  # Keep reference
        label.pack(padx=10, pady=10)
        
        tk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=5)
        
    def render_sprite(self, color_input):
        """Render sprite with input colors using brightness-based shading"""
        width, height = self.original_image.size
        result_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        # First pass: fill in all original pixels (preserves unclassified areas)
        for y in range(height):
            for x in range(width):
                pixel = self.original_image.getpixel((x, y))
                if pixel[3] > 0:  # If not fully transparent
                    result_img.putpixel((x, y), pixel)
        
        # Second pass: override with categorized pixels
        for (x, y), cat in self.pixel_categories.items():
            original_pixel = self.original_image.getpixel((x, y))
            
            if cat == "outline":
                # Keep outline close to original darkness
                brightness = (original_pixel[0] + original_pixel[1] + original_pixel[2]) / (3 * 255)
                outline_val = int(brightness * 80)  # Dark but preserves some variation
                result_img.putpixel((x, y), (outline_val, outline_val, outline_val, 255))
            else:
                # Calculate brightness of original pixel (0.0 to 1.0)
                brightness = (original_pixel[0] + original_pixel[1] + original_pixel[2]) / (3 * 255)
                
                # Get target color
                target_hex = color_input[cat]
                target_r = int(target_hex[1:3], 16)
                target_g = int(target_hex[3:5], 16)
                target_b = int(target_hex[5:7], 16)
                
                # Apply brightness modifier
                # Use a curve to preserve shadows and highlights
                if brightness < 0.5:
                    # Darken for shadows
                    factor = brightness * 2
                else:
                    # Lighten for highlights
                    factor = 0.5 + (brightness - 0.5) * 1.5
                    
                new_r = int(min(255, max(0, target_r * factor)))
                new_g = int(min(255, max(0, target_g * factor)))
                new_b = int(min(255, max(0, target_b * factor)))
                
                result_img.putpixel((x, y), (new_r, new_g, new_b, 255))
        
        return result_img

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelAnnotator(root)
    root.mainloop()
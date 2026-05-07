import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model (object detection, not segmentation)
model = YOLO(Path(__file__).with_name('bestv26.pt'))
class_labels = ['immature', 'mature', 'rotten']
CONFIDENCE_THRESHOLD = 0.5
FONT_SIZE = 22

def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        process_and_display(img_path)

def analyze_rgb(image, boxes):
    np_image = np.array(image)
    avg_colors = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = np_image[y1:y2, x1:x2]
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        avg_colors.append(tuple(map(int, avg_color)))
    return avg_colors

def draw_boxes_with_info(image, boxes, classes, confs, avg_colors):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(image)

    # Dynamic font size
    scale_factor = image.size[1] / 500
    scaled_font_size = max(int(FONT_SIZE * scale_factor), 16)
    try:
        font = ImageFont.truetype("arial.ttf", scaled_font_size)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = class_labels[int(classes[i])]
        conf = confs[i]
        rgb = avg_colors[i]

        # Choose high-contrast color
        if label == "immature":
            color = (14, 165, 233)     # Yellow
        elif label == "mature":
            color = (34, 197, 94)      # Lime Green
        else:
            color = (239, 68, 68)      # Crimson Red

        fill_color = color + (60,)    # Semi-transparent fill
        outline_color = color + (255,)

        # Draw filled box with transparent fill
        draw_overlay.rectangle([x1, y1, x2, y2], fill=fill_color)

        # Draw thick border
        for offset in range(3):
            draw_main.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=outline_color)

        # Top: maize number + confidence
        top_text = f"Maize {i+1} ({conf:.2f})"
        draw_main.text((x1, y1), top_text, fill=outline_color, font=font)

        # Bottom: label + RGB
        bottom_text = f"{label} - R:{rgb[0]} G:{rgb[1]} B:{rgb[2]}"
        text_height = scaled_font_size + 4
        draw_main.text((x1, y2 - text_height), bottom_text, fill=outline_color, font=font)

    combined = Image.alpha_composite(image, overlay)
    return combined.convert("RGB")

def create_overlay_heatmap(image, boxes):
    np_image = np.array(image)
    heatmap = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = np_image[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        intensity = cv2.GaussianBlur(gray_roi, (15, 15), 0)
        heatmap[y1:y2, x1:x2] = intensity

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colormap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_TURBO)
    overlaid = cv2.addWeighted(np_image, 0.6, colormap, 0.4, 0)
    return overlaid

def process_and_display(path):
    results = model(path)[0]
    all_boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    mask = confs >= CONFIDENCE_THRESHOLD
    filtered_boxes = all_boxes[mask]
    filtered_confs = confs[mask]
    filtered_classes = classes[mask]

    if len(filtered_boxes) == 0:
        result_label.config(text="No maize detected.")
        return

    image = Image.open(path).convert("RGB")
    rgb_values = analyze_rgb(image, filtered_boxes)

    boxed_image = draw_boxes_with_info(image.copy(), filtered_boxes, filtered_classes, filtered_confs, rgb_values)
    boxed_image = boxed_image.resize((350, 350))
    tk_boxed = ImageTk.PhotoImage(boxed_image)
    canvas_input.config(image=tk_boxed)
    canvas_input.image = tk_boxed

    overlay = create_overlay_heatmap(image.copy(), filtered_boxes)
    overlay = Image.fromarray(cv2.resize(overlay, (350, 350)))
    tk_overlay = ImageTk.PhotoImage(overlay)
    canvas_output.config(image=tk_overlay)
    canvas_output.image = tk_overlay

    rgb_text = "\n".join([f"Maize {i+1}: R:{rgb_values[i][0]}, G:{rgb_values[i][1]}, B:{rgb_values[i][2]}" for i in range(len(rgb_values))])
    result_label.config(text=f"Detected maize: {len(rgb_values)}\n\n{rgb_text}")

def on_closing():
    root.destroy()

# GUI Setup
root = tk.Tk()
root.title("🌽 Colorimetric Maize Detector 🌽")
root.geometry("800x540")
root.configure(bg="#0f172a")
root.protocol("WM_DELETE_WINDOW", on_closing)

font_title = ("Helvetica", 18, "bold")
font_text = ("Helvetica", 14)
bg_color = "#0f172a"
fg_color = "#e2e8f0"
accent = "#38bdf8"

tk.Label(root, text="🌽 Colorimetric Maize Detector 🌽", font=font_title, bg=bg_color, fg=accent).pack(pady=10)

tk.Button(root, text="📤 Upload Image", command=upload_image, font=font_text,
          bg=accent, fg="white", height=2, width=20, bd=0, relief="flat").pack(pady=10)

frame_images = tk.Frame(root, bg=bg_color)
frame_images.pack(pady=10)

canvas_input = tk.Label(frame_images, bg=bg_color)
canvas_input.pack(side="left", padx=20)

canvas_output = tk.Label(frame_images, bg=bg_color)
canvas_output.pack(side="right", padx=20)

result_label = tk.Label(root, text="", font=font_text, bg=bg_color, fg=fg_color, justify="left")
result_label.pack(pady=10)

root.mainloop()

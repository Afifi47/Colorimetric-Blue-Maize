import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO


APP_TITLE = "Colorimetric Blue Maize Detector"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bestv26.pt"
CLASS_LABELS = ["immature", "mature", "rotten"]
LABEL_COLORS = {
    "immature": (14, 165, 233),
    "mature": (34, 197, 94),
    "rotten": (239, 68, 68),
}
PLACEHOLDER_SIZE = (620, 460)
MIN_ZOOM = 0.25
MAX_ZOOM = 2.50
ZOOM_STEP = 0.10
FONT_SIZE = 22


class BlueMaizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x780")
        self.root.minsize(820, 560)
        self.root.configure(bg="#07111f")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model = None
        self.selected_path = None
        self.annotated_image = None
        self.heatmap_image = None
        self.input_photo = None
        self.heatmap_photo = None
        self.is_busy = False
        self.threshold_job = None
        self.preview_canvases = {}
        self.preview_photos = {}
        self.display_images = {}

        self.confidence_var = tk.DoubleVar(value=0.50)
        self.zoom_var = tk.DoubleVar(value=1.00)
        self.zoom_label_var = tk.StringVar(value="Zoom: 100%")
        self.status_var = tk.StringVar(value="Ready. Choose an image to begin.")
        self.file_var = tk.StringVar(value="No image selected")
        self.count_vars = {
            "total": tk.StringVar(value="0"),
            "immature": tk.StringVar(value="0"),
            "mature": tk.StringVar(value="0"),
            "rotten": tk.StringVar(value="0"),
        }

        self.colors = {
            "bg": "#07111f",
            "panel": "#101c2e",
            "panel_alt": "#15243a",
            "panel_hot": "#1d3354",
            "text": "#f4fbff",
            "muted": "#a8bed0",
            "accent": "#00b8d9",
            "accent_dark": "#0288a8",
            "green": "#22c55e",
            "red": "#ef4444",
            "yellow": "#f59e0b",
            "purple": "#a855f7",
            "border": "#2e4b66",
        }

        self.build_ui()
        self.reset_image_panels()

    def build_ui(self):
        self.configure_styles()

        header = tk.Frame(self.root, bg=self.colors["bg"])
        header.pack(fill="x", padx=22, pady=(16, 10))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        title = tk.Label(
            header,
            text=APP_TITLE,
            font=("Segoe UI", 24, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["text"],
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = tk.Label(
            header,
            text="Detection, maturity class, RGB color analysis, heatmap, and export in one workspace",
            font=("Segoe UI", 11),
            bg=self.colors["bg"],
            fg=self.colors["muted"],
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        accent_row = tk.Frame(header, bg=self.colors["bg"])
        accent_row.grid(row=0, column=1, rowspan=2, sticky="e")
        for color in (self.colors["accent"], self.colors["green"], self.colors["yellow"], self.colors["red"]):
            tk.Label(accent_row, bg=color, width=4, height=2).pack(side="left", padx=3)

        main = tk.PanedWindow(
            self.root,
            orient="horizontal",
            bg=self.colors["bg"],
            sashwidth=8,
            sashrelief="flat",
            bd=0,
            showhandle=True,
        )
        main.pack(fill="both", expand=True, padx=22, pady=(0, 18))

        image_pane = tk.PanedWindow(
            main,
            orient="horizontal",
            bg=self.colors["bg"],
            sashwidth=8,
            sashrelief="flat",
            bd=0,
            showhandle=True,
        )

        self.left_panel = self.create_panel(image_pane)
        self.create_image_section(self.left_panel, "Detection Result", "input")

        self.middle_panel = self.create_panel(image_pane)
        self.create_image_section(self.middle_panel, "Color Heatmap", "heatmap")

        image_pane.add(self.left_panel, minsize=300, stretch="always")
        image_pane.add(self.middle_panel, minsize=300, stretch="always")

        self.side_panel = self.create_panel(main)
        self.create_control_section(self.side_panel)

        main.add(image_pane, minsize=560, stretch="always")
        main.add(self.side_panel, minsize=310, width=350, stretch="never")

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["panel_alt"],
            bordercolor=self.colors["border"],
            rowheight=28,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background=self.colors["border"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", self.colors["accent_dark"])])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=self.colors["panel_alt"],
            background=self.colors["accent"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )
        style.configure(
            "Horizontal.TScale",
            background=self.colors["panel"],
            troughcolor=self.colors["border"],
        )

    def create_panel(self, parent):
        panel = tk.Frame(
            parent,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        return panel

    def create_image_section(self, parent, title, panel_type):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        heading = tk.Frame(parent, bg=self.colors["panel"])
        heading.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 10))
        heading.columnconfigure(0, weight=1)

        title_label = tk.Label(
            heading,
            text=title,
            font=("Segoe UI", 14, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"],
        )
        title_label.grid(row=0, column=0, sticky="w")

        badge_text = "Boxes + RGB" if panel_type == "input" else "Intensity Overlay"
        badge_color = self.colors["accent"] if panel_type == "input" else self.colors["purple"]
        badge = tk.Label(
            heading,
            text=badge_text,
            font=("Segoe UI", 9, "bold"),
            bg=badge_color,
            fg="#ffffff",
            padx=10,
            pady=4,
        )
        badge.grid(row=0, column=1, sticky="e")

        canvas_frame = tk.Frame(
            parent,
            bg=self.colors["panel_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        canvas = tk.Canvas(
            canvas_frame,
            bg=self.colors["panel_alt"],
            highlightthickness=0,
            xscrollincrement=20,
            yscrollincrement=20,
        )
        x_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        y_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        canvas.bind("<MouseWheel>", lambda event, name=panel_type: self.on_canvas_mousewheel(event, name))
        canvas.bind("<Control-MouseWheel>", self.on_zoom_mousewheel)
        self.preview_canvases[panel_type] = canvas

    def create_control_section(self, parent):
        parent.configure(width=350)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        scroll_canvas = tk.Canvas(parent, bg=self.colors["panel"], highlightthickness=0, width=340)
        scroll_bar = tk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scroll_bar.set)
        scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scroll_bar.grid(row=0, column=1, sticky="ns")

        content = tk.Frame(scroll_canvas, bg=self.colors["panel"])
        content_id = scroll_canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
        )
        scroll_canvas.bind(
            "<Configure>",
            lambda event: scroll_canvas.itemconfigure(content_id, width=event.width),
        )
        scroll_canvas.bind("<MouseWheel>", lambda event: scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        parent = content
        parent.columnconfigure(0, weight=1)

        controls = tk.Frame(parent, bg=self.colors["panel"])
        controls.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        controls.columnconfigure(0, weight=1)

        self.upload_button = self.create_button(controls, "Upload Image", self.upload_image)
        self.upload_button.grid(row=0, column=0, sticky="ew")

        button_row = tk.Frame(controls, bg=self.colors["panel"])
        button_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        self.save_button = self.create_button(button_row, "Save Result", self.save_result, disabled=True)
        self.save_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.clear_button = self.create_button(button_row, "Clear", self.clear_results)
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        threshold_frame = tk.Frame(parent, bg=self.colors["panel"])
        threshold_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(4, 16))
        threshold_frame.columnconfigure(0, weight=1)

        self.threshold_label = tk.Label(
            threshold_frame,
            text="Confidence threshold: 0.50",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"],
        )
        self.threshold_label.grid(row=0, column=0, sticky="w")

        threshold = ttk.Scale(
            threshold_frame,
            from_=0.10,
            to=0.95,
            variable=self.confidence_var,
            command=self.on_threshold_change,
        )
        threshold.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        zoom_frame = tk.Frame(parent, bg=self.colors["panel"])
        zoom_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        zoom_frame.columnconfigure(1, weight=1)

        zoom_title = tk.Label(
            zoom_frame,
            textvariable=self.zoom_label_var,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"],
        )
        zoom_title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        zoom_out = self.create_button(zoom_frame, "-", self.zoom_out)
        zoom_out.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        zoom_slider = ttk.Scale(
            zoom_frame,
            from_=MIN_ZOOM,
            to=MAX_ZOOM,
            variable=self.zoom_var,
            command=self.on_zoom_change,
        )
        zoom_slider.grid(row=1, column=1, sticky="ew")

        zoom_in = self.create_button(zoom_frame, "+", self.zoom_in)
        zoom_in.grid(row=1, column=2, sticky="ew", padx=(6, 0))

        fit_button = self.create_button(zoom_frame, "Fit to View", self.fit_to_view)
        fit_button.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        file_label = tk.Label(
            parent,
            textvariable=self.file_var,
            font=("Segoe UI", 9),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            wraplength=250,
            justify="left",
        )
        file_label.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))

        stats = tk.Frame(parent, bg=self.colors["panel"])
        stats.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 16))
        stats.columnconfigure((0, 1), weight=1)

        self.create_stat(stats, "Detected", self.count_vars["total"], 0, 0, self.colors["purple"])
        self.create_stat(stats, "Immature", self.count_vars["immature"], 0, 1, self.colors["accent"])
        self.create_stat(stats, "Mature", self.count_vars["mature"], 1, 0, self.colors["green"])
        self.create_stat(stats, "Rotten", self.count_vars["rotten"], 1, 1, self.colors["red"])

        table_title = tk.Label(
            parent,
            text="Analysis",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"],
        )
        table_title.grid(row=5, column=0, sticky="w", padx=18, pady=(0, 8))

        columns = ("maize", "class", "conf", "rgb")
        self.result_table = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        self.result_table.heading("maize", text="#")
        self.result_table.heading("class", text="Class")
        self.result_table.heading("conf", text="Conf")
        self.result_table.heading("rgb", text="RGB")
        self.result_table.column("maize", width=36, anchor="center", stretch=False)
        self.result_table.column("class", width=76, anchor="center", stretch=False)
        self.result_table.column("conf", width=54, anchor="center", stretch=False)
        self.result_table.column("rgb", width=122, anchor="center", stretch=True)
        self.result_table.grid(row=6, column=0, sticky="nsew", padx=18)
        parent.rowconfigure(6, weight=1)

        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.grid(row=7, column=0, sticky="ew", padx=18, pady=(14, 8))

        status = tk.Label(
            parent,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            wraplength=250,
            justify="left",
        )
        status.grid(row=8, column=0, sticky="ew", padx=18, pady=(0, 18))

    def create_button(self, parent, text, command, disabled=False):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            activebackground=self.colors["accent_dark"],
            fg="#ffffff",
            activeforeground="#ffffff",
            disabledforeground="#8fa2ad",
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            cursor="hand2",
        )
        if disabled:
            button.configure(state="disabled", bg="#334652")
        return button

    def create_stat(self, parent, label, variable, row, column, accent):
        card = tk.Frame(
            parent,
            bg=self.colors["panel_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        card.grid(row=row, column=column, sticky="ew", padx=4, pady=4)

        tk.Frame(card, bg=accent, height=4).pack(fill="x")

        value_label = tk.Label(
            card,
            textvariable=variable,
            font=("Segoe UI", 18, "bold"),
            bg=self.colors["panel_alt"],
            fg=self.colors["text"],
        )
        value_label.pack(pady=(8, 0))

        name_label = tk.Label(
            card,
            text=label,
            font=("Segoe UI", 9),
            bg=self.colors["panel_alt"],
            fg=self.colors["muted"],
        )
        name_label.pack(pady=(0, 8))

    def on_threshold_change(self, _value):
        threshold = self.confidence_var.get()
        self.threshold_label.configure(text=f"Confidence threshold: {threshold:.2f}")
        if not self.selected_path or self.is_busy:
            return

        if self.threshold_job is not None:
            self.root.after_cancel(self.threshold_job)
        self.threshold_job = self.root.after(450, self.reprocess_selected_image)

    def reprocess_selected_image(self):
        self.threshold_job = None
        if self.selected_path and not self.is_busy:
            self.start_processing(self.selected_path)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Choose a maize image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.selected_path = Path(path)
        self.file_var.set(str(self.selected_path))
        self.start_processing(self.selected_path)

    def start_processing(self, path):
        self.set_busy(True)
        self.status_var.set("Analyzing image...")
        thread = threading.Thread(target=self.process_image_worker, args=(Path(path),), daemon=True)
        thread.start()

    def load_model(self):
        if self.model is not None:
            return self.model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        return self.model

    def process_image_worker(self, path):
        try:
            model = self.load_model()
            original = Image.open(path).convert("RGB")
            results = model(str(path), verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            threshold = self.confidence_var.get()
            mask = confs >= threshold
            filtered_boxes = boxes[mask]
            filtered_confs = confs[mask]
            filtered_classes = classes[mask]

            detections = []
            if len(filtered_boxes) > 0:
                rgb_values = analyze_rgb(original, filtered_boxes)
                detections = build_detections(filtered_boxes, filtered_classes, filtered_confs, rgb_values)
                annotated = draw_boxes_with_info(original.copy(), detections)
                heatmap = create_overlay_heatmap(original.copy(), filtered_boxes)
                heatmap = Image.fromarray(heatmap)
            else:
                annotated = original.copy()
                heatmap = create_empty_heatmap(original.copy())

            self.root.after(0, self.show_results, annotated, heatmap, detections)
        except Exception as exc:
            self.root.after(0, self.show_error, exc)

    def show_results(self, annotated, heatmap, detections):
        self.annotated_image = annotated
        self.heatmap_image = heatmap
        self.set_preview_image("input", annotated)
        self.set_preview_image("heatmap", heatmap)
        self.update_table(detections)
        self.update_counts(detections)

        if detections:
            self.status_var.set(f"Analysis complete. {len(detections)} maize object(s) detected.")
            self.save_button.configure(state="normal", bg=self.colors["accent"])
        else:
            self.status_var.set("Analysis complete. No maize detected above the selected threshold.")
            self.save_button.configure(state="disabled", bg="#334652")

        self.set_busy(False)

    def show_error(self, exc):
        self.set_busy(False)
        self.status_var.set("Could not analyze the selected image.")
        messagebox.showerror("Analysis error", str(exc))

    def set_preview_image(self, panel_type, image):
        self.display_images[panel_type] = image
        canvas = self.preview_canvases[panel_type]
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom_var.get()))
        width = max(1, int(image.width * zoom))
        height = max(1, int(image.height * zoom))

        max_side = 5200
        if width > max_side or height > max_side:
            ratio = min(max_side / width, max_side / height)
            width = max(1, int(width * ratio))
            height = max(1, int(height * ratio))

        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas_width = max(canvas.winfo_width(), 1)
        canvas_height = max(canvas.winfo_height(), 1)
        x = max((canvas_width - width) // 2, 0)
        y = max((canvas_height - height) // 2, 0)
        canvas.create_image(x, y, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, max(width, canvas_width), max(height, canvas_height)))

        self.preview_photos[panel_type] = photo
        if panel_type == "input":
            self.input_photo = photo
        else:
            self.heatmap_photo = photo

    def refresh_previews(self):
        for panel_type, image in self.display_images.items():
            self.set_preview_image(panel_type, image)

    def on_zoom_change(self, _value=None):
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom_var.get()))
        self.zoom_var.set(zoom)
        self.zoom_label_var.set(f"Zoom: {int(zoom * 100)}%")
        self.refresh_previews()

    def zoom_in(self):
        self.zoom_var.set(min(MAX_ZOOM, self.zoom_var.get() + ZOOM_STEP))
        self.on_zoom_change()

    def zoom_out(self):
        self.zoom_var.set(max(MIN_ZOOM, self.zoom_var.get() - ZOOM_STEP))
        self.on_zoom_change()

    def fit_to_view(self):
        image = self.display_images.get("input")
        canvas = self.preview_canvases.get("input")
        if image is None or canvas is None:
            return

        canvas_width = max(canvas.winfo_width() - 18, 1)
        canvas_height = max(canvas.winfo_height() - 18, 1)
        zoom = min(canvas_width / image.width, canvas_height / image.height)
        self.zoom_var.set(max(MIN_ZOOM, min(MAX_ZOOM, zoom)))
        self.on_zoom_change()

    def on_zoom_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        return "break"

    def on_canvas_mousewheel(self, event, panel_type):
        canvas = self.preview_canvases[panel_type]
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def update_table(self, detections):
        for row in self.result_table.get_children():
            self.result_table.delete(row)

        for detection in detections:
            rgb = detection["rgb"]
            self.result_table.insert(
                "",
                "end",
                values=(
                    detection["index"],
                    detection["label"],
                    f"{detection['confidence']:.2f}",
                    f"{rgb[0]}, {rgb[1]}, {rgb[2]}",
                ),
            )

    def update_counts(self, detections):
        counts = {"immature": 0, "mature": 0, "rotten": 0}
        for detection in detections:
            counts[detection["label"]] = counts.get(detection["label"], 0) + 1

        self.count_vars["total"].set(str(len(detections)))
        for label in ("immature", "mature", "rotten"):
            self.count_vars[label].set(str(counts[label]))

    def clear_results(self):
        self.selected_path = None
        self.annotated_image = None
        self.heatmap_image = None
        self.file_var.set("No image selected")
        self.status_var.set("Ready. Choose an image to begin.")
        self.save_button.configure(state="disabled", bg="#334652")
        self.update_table([])
        self.update_counts([])
        self.reset_image_panels()

    def reset_image_panels(self):
        self.set_preview_image("input", create_placeholder("Upload an image"))
        self.set_preview_image("heatmap", create_placeholder("Heatmap appears here"))

    def save_result(self):
        if self.annotated_image is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save annotated result",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg")],
        )
        if not path:
            return

        self.annotated_image.save(path)
        self.status_var.set(f"Saved result to {path}")

    def set_busy(self, busy):
        self.is_busy = busy
        if busy:
            self.progress.start(12)
            self.upload_button.configure(state="disabled", bg="#334652")
            self.clear_button.configure(state="disabled", bg="#334652")
        else:
            self.progress.stop()
            self.upload_button.configure(state="normal", bg=self.colors["accent"])
            self.clear_button.configure(state="normal", bg=self.colors["accent"])

    def on_closing(self):
        self.root.destroy()


def analyze_rgb(image, boxes):
    np_image = np.array(image)
    avg_colors = []

    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, np_image.shape)
        roi = np_image[y1:y2, x1:x2]
        if roi.size == 0:
            avg_colors.append((0, 0, 0))
            continue

        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        avg_colors.append(tuple(map(int, avg_color)))

    return avg_colors


def build_detections(boxes, classes, confs, avg_colors):
    detections = []
    for index, box in enumerate(boxes, start=1):
        class_index = int(classes[index - 1])
        label = CLASS_LABELS[class_index] if 0 <= class_index < len(CLASS_LABELS) else f"class {class_index}"
        detections.append(
            {
                "index": index,
                "box": box,
                "label": label,
                "confidence": float(confs[index - 1]),
                "rgb": avg_colors[index - 1],
            }
        )
    return detections


def draw_boxes_with_info(image, detections):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(image)

    scale_factor = image.size[1] / 500
    scaled_font_size = max(int(FONT_SIZE * scale_factor), 16)
    font = load_font(scaled_font_size)

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        label = detection["label"]
        conf = detection["confidence"]
        rgb = detection["rgb"]
        color = LABEL_COLORS.get(label, (245, 158, 11))

        draw_overlay.rectangle([x1, y1, x2, y2], fill=color + (55,))
        for offset in range(3):
            draw_main.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color + (255,),
            )

        top_text = f"Maize {detection['index']} ({conf:.2f})"
        bottom_text = f"{label} - R:{rgb[0]} G:{rgb[1]} B:{rgb[2]}"
        draw_label(draw_main, (x1, y1), top_text, font, color)
        draw_label(draw_main, (x1, max(y1, y2 - scaled_font_size - 10)), bottom_text, font, color)

    return Image.alpha_composite(image, overlay).convert("RGB")


def draw_label(draw, position, text, font, color):
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    padding = 5
    background = [
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    ]
    draw.rectangle(background, fill=(16, 24, 32, 220))
    draw.text((x, y), text, fill=color + (255,), font=font)


def create_overlay_heatmap(image, boxes):
    np_image = np.array(image)
    heatmap = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, np_image.shape)
        roi = np_image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        intensity = cv2.GaussianBlur(gray_roi, (15, 15), 0)
        heatmap[y1:y2, x1:x2] = intensity

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colormap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_TURBO)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(np_image, 0.58, colormap, 0.42, 0)
    return overlaid


def clip_box(box, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def create_empty_heatmap(image):
    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    colormap = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(np_image, 0.70, colormap, 0.30, 0))


def create_placeholder(text):
    image = Image.new("RGB", PLACEHOLDER_SIZE, "#15243a")
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (PLACEHOLDER_SIZE[0] - (bbox[2] - bbox[0])) // 2
    y = (PLACEHOLDER_SIZE[1] - (bbox[3] - bbox[1])) // 2
    draw.rectangle([1, 1, PLACEHOLDER_SIZE[0] - 2, PLACEHOLDER_SIZE[1] - 2], outline="#2e4b66", width=2)
    draw.rectangle([20, 20, PLACEHOLDER_SIZE[0] - 22, 28], fill="#00b8d9")
    draw.rectangle([20, 34, PLACEHOLDER_SIZE[0] - 180, 42], fill="#22c55e")
    draw.rectangle([20, 48, PLACEHOLDER_SIZE[0] - 300, 56], fill="#f59e0b")
    draw.text((x, y), text, fill="#9fb3c1", font=font)
    return image


def load_font(size):
    for font_name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    root = tk.Tk()
    app = BlueMaizeApp(root)
    root.mainloop()

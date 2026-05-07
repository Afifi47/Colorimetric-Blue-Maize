import threading
import tkinter as tk
from math import cos, radians, sin
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
        self.analysis_rows = []
        self.current_confidence_meter = 0
        self.current_rgb_meter = (0, 0, 0)

        self.confidence_var = tk.DoubleVar(value=0.50)
        self.zoom_var = tk.DoubleVar(value=1.00)
        self.zoom_label_var = tk.StringVar(value="Zoom: 100%")
        self.status_var = tk.StringVar(value="Ready. Choose an image to begin.")
        self.file_var = tk.StringVar(value="No image selected")
        self.model_status_var = tk.StringVar(value="Model ready" if MODEL_PATH.exists() else "Model missing")
        self.image_info_var = tk.StringVar(value="No image loaded")
        self.avg_confidence_var = tk.StringVar(value="--")
        self.dominant_rgb_var = tk.StringVar(value="R: --  G: --  B: --")
        self.maturity_summary_var = tk.StringVar(value="Waiting for analysis")
        self.count_vars = {
            "total": tk.StringVar(value="0"),
            "immature": tk.StringVar(value="0"),
            "mature": tk.StringVar(value="0"),
            "rotten": tk.StringVar(value="0"),
        }
        self.distribution_vars = {
            "immature": tk.StringVar(value="0%"),
            "mature": tk.StringVar(value="0%"),
            "rotten": tk.StringVar(value="0%"),
        }

        self.colors = {
            "bg": "#060817",
            "panel": "#0d1326",
            "panel_alt": "#121a33",
            "panel_hot": "#172044",
            "text": "#f7fbff",
            "muted": "#91a9c7",
            "accent": "#00e5ff",
            "accent_dark": "#029db4",
            "green": "#39ff88",
            "red": "#ff3864",
            "yellow": "#ffd166",
            "purple": "#b967ff",
            "pink": "#ff4ecd",
            "border": "#31446b",
            "grid": "#1b2b54",
        }

        self.root.configure(bg=self.colors["bg"])
        self.build_ui()
        self.reset_image_panels()

    def build_ui(self):
        self.configure_styles()

        header = tk.Frame(
            self.root,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        header.pack(fill="x", padx=22, pady=(16, 10))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=3)
        header.columnconfigure(2, weight=1)

        left_accent = tk.Frame(header, bg=self.colors["panel"])
        left_accent.grid(row=0, column=0, rowspan=2, sticky="w", padx=18, pady=12)
        for color in (self.colors["accent"], self.colors["purple"], self.colors["pink"]):
            tk.Label(left_accent, bg=color, width=5, height=1).pack(anchor="w", pady=3)

        title = tk.Label(
            header,
            text="COLORIMETRIC BLUE MAIZE",
            font=("OCR A Extended", 24, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["accent"],
        )
        title.grid(row=0, column=1, sticky="ew", pady=(12, 0))

        subtitle = tk.Label(
            header,
            text="DETECTION DASHBOARD  |  RGB ANALYTICS  |  HEATMAP VISUALIZATION",
            font=("Consolas", 10, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
        )
        subtitle.grid(row=1, column=1, sticky="ew", pady=(4, 12))

        accent_row = tk.Frame(header, bg=self.colors["panel"])
        accent_row.grid(row=0, column=2, rowspan=2, sticky="e", padx=18)
        for color in (self.colors["accent"], self.colors["green"], self.colors["yellow"], self.colors["red"], self.colors["pink"]):
            tk.Label(accent_row, bg=color, width=2, height=2).pack(side="left", padx=3)

        self.create_dashboard()

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

    def create_dashboard(self):
        dashboard = tk.Frame(self.root, bg=self.colors["bg"])
        dashboard.pack(fill="x", padx=22, pady=(0, 14))
        dashboard.rowconfigure(0, minsize=250, weight=1)
        for column in range(4):
            dashboard.columnconfigure(column, weight=1, uniform="dashboard")

        self.model_card = self.create_metric_card(
            dashboard,
            "System",
            self.model_status_var,
            "YOLOv8 weights",
            self.colors["accent"],
            0,
        )
        self.image_card = self.create_metric_card(
            dashboard,
            "Image",
            self.image_info_var,
            "Current input",
            self.colors["purple"],
            1,
        )
        self.confidence_card = self.create_metric_card(
            dashboard,
            "Confidence",
            self.avg_confidence_var,
            "Average detection",
            self.colors["green"],
            2,
        )
        self.confidence_meter = tk.Canvas(
            self.confidence_card,
            height=142,
            bg=self.colors["panel_alt"],
            highlightthickness=0,
        )
        self.confidence_meter.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 12))
        self.confidence_meter.bind("<Configure>", lambda _event: self.draw_confidence_meter(self.current_confidence_meter))

        color_card = self.create_metric_card(
            dashboard,
            "Dominant RGB",
            self.dominant_rgb_var,
            "Average detected color",
            self.colors["yellow"],
            3,
        )
        self.rgb_swatch = tk.Label(
            color_card,
            bg="#334652",
            width=9,
            height=4,
            highlightbackground=self.colors["text"],
            highlightthickness=1,
        )
        self.rgb_swatch.grid(row=1, column=1, rowspan=3, sticky="ne", padx=(10, 14), pady=(14, 0))
        self.rgb_meter = tk.Canvas(
            color_card,
            height=142,
            bg=self.colors["panel_alt"],
            highlightthickness=0,
        )
        self.rgb_meter.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 12))
        self.rgb_meter.bind("<Configure>", lambda _event: self.draw_rgb_meter(self.current_rgb_meter))
        self.draw_confidence_meter(0)
        self.draw_rgb_meter((0, 0, 0))

    def create_metric_card(self, parent, title, value_var, subtitle, accent, column):
        card = tk.Frame(
            parent,
            bg=self.colors["panel_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            height=250,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        card.grid_propagate(False)
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=0)
        card.rowconfigure(4, weight=1)

        tk.Frame(card, bg=accent, height=5).grid(row=0, column=0, columnspan=2, sticky="ew")

        title_label = tk.Label(
            card,
            text=title.upper(),
            font=("Consolas", 8, "bold"),
            bg=self.colors["panel_alt"],
            fg=accent,
        )
        title_label.grid(row=1, column=0, sticky="w", padx=14, pady=(12, 0))

        value_label = tk.Label(
            card,
            textvariable=value_var,
            font=("Consolas", 13, "bold"),
            bg=self.colors["panel_alt"],
            fg=self.colors["text"],
            anchor="w",
            justify="left",
            wraplength=220,
        )
        value_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(2, 0))

        subtitle_label = tk.Label(
            card,
            text=subtitle,
            font=("Consolas", 9),
            bg=self.colors["panel_alt"],
            fg=self.colors["muted"],
        )
        subtitle_label.grid(row=3, column=0, sticky="w", padx=14, pady=(2, 12))
        return card

    def draw_confidence_meter(self, value):
        if not hasattr(self, "confidence_meter"):
            return

        self.current_confidence_meter = value
        meter = self.confidence_meter
        meter.delete("all")
        width = max(meter.winfo_width(), 220)
        height = 142
        value = max(0, min(1, value))
        cx = width / 2
        cy = height - 24
        radius = min(width * 0.40, 96)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        arc_width = 18

        meter.create_arc(bbox, start=0, extent=180, style="arc", outline=self.colors["grid"], width=arc_width)
        meter.create_arc(
            bbox,
            start=180,
            extent=-180 * value,
            style="arc",
            outline=self.colors["green"] if value >= 0.5 else self.colors["yellow"],
            width=arc_width,
        )

        for marker, text in ((0, "0"), (0.5, "50"), (1, "100")):
            angle = 180 - (180 * marker)
            x1, y1 = point_on_circle(cx, cy, radius - 12, angle)
            x2, y2 = point_on_circle(cx, cy, radius + 5, angle)
            meter.create_line(x1, y1, x2, y2, fill=self.colors["border"], width=2)

            if marker == 0.5:
                xt, yt = point_on_circle(cx, cy, radius - 30, angle)
                anchor = "center"
            elif marker == 0:
                xt, yt = point_on_circle(cx, cy, radius + 12, angle)
                anchor = "e"
            else:
                xt, yt = point_on_circle(cx, cy, radius + 12, angle)
                anchor = "w"

            meter.create_text(xt, yt, text=text, anchor=anchor, fill=self.colors["muted"], font=("Consolas", 9, "bold"))

        needle_angle = 180 - (180 * value)
        nx, ny = point_on_circle(cx, cy, radius - 24, needle_angle)
        meter.create_line(cx, cy, nx, ny, fill=self.colors["text"], width=4)
        meter.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, fill=self.colors["border"], outline=self.colors["text"], width=2)
        meter.create_text(cx, cy - 30, text=f"{int(value * 100)}%", fill=self.colors["text"], font=("Consolas", 16, "bold"))

    def draw_rgb_meter(self, rgb):
        if not hasattr(self, "rgb_meter"):
            return

        self.current_rgb_meter = rgb
        meter = self.rgb_meter
        meter.delete("all")
        width = max(meter.winfo_width(), 220)
        labels = (
            ("R", rgb[0], self.colors["red"]),
            ("G", rgb[1], self.colors["green"]),
            ("B", rgb[2], self.colors["accent"]),
        )
        gauge_width = width / 3
        radius = min(gauge_width * 0.38, 48)

        for index, (label, value, color) in enumerate(labels):
            value = max(0, min(255, value))
            fraction = value / 255
            cx = (gauge_width * index) + (gauge_width / 2)
            cy = 104
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)

            meter.create_arc(bbox, start=0, extent=180, style="arc", outline=self.colors["grid"], width=11)
            meter.create_arc(bbox, start=180, extent=-180 * fraction, style="arc", outline=color, width=11)

            needle_angle = 180 - (180 * fraction)
            nx, ny = point_on_circle(cx, cy, radius - 13, needle_angle)
            meter.create_line(cx, cy, nx, ny, fill=self.colors["text"], width=2)
            meter.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=self.colors["border"], outline=color, width=2)
            meter.create_text(cx, 16, text=label, fill=color, font=("Consolas", 11, "bold"))
            meter.create_text(cx, 42, text=f"{value:03d}", fill=self.colors["text"], font=("Consolas", 13, "bold"))
            meter.create_text(cx - radius, cy + 12, text="0", fill=self.colors["muted"], font=("Consolas", 8, "bold"))
            meter.create_text(cx + radius, cy + 12, text="255", fill=self.colors["muted"], font=("Consolas", 8, "bold"))

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
            font=("OCR A Extended", 13, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["accent"],
        )
        title_label.grid(row=0, column=0, sticky="w")

        badge_text = "Boxes + RGB" if panel_type == "input" else "Intensity Overlay"
        badge_color = self.colors["accent"] if panel_type == "input" else self.colors["purple"]
        badge = tk.Label(
            heading,
            text=badge_text,
            font=("Consolas", 9, "bold"),
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

        self.create_section_title(parent, "Workflow", "Upload, analyze, export")

        controls = tk.Frame(parent, bg=self.colors["panel"])
        controls.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 12))
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

        self.create_section_title(parent, "Tuning", "Adjust sensitivity and inspection scale", row=2)

        threshold_frame = tk.Frame(parent, bg=self.colors["panel"])
        threshold_frame.grid(row=3, column=0, sticky="ew", padx=18, pady=(4, 16))
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
        zoom_frame.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 16))
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
        file_label.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 14))

        self.create_section_title(parent, "Dashboard", "Live detection summary", row=6)

        stats = tk.Frame(parent, bg=self.colors["panel"])
        stats.grid(row=7, column=0, sticky="ew", padx=18, pady=(8, 16))
        stats.columnconfigure((0, 1), weight=1)

        self.create_stat(stats, "Detected", self.count_vars["total"], 0, 0, self.colors["purple"])
        self.create_stat(stats, "Immature", self.count_vars["immature"], 0, 1, self.colors["accent"])
        self.create_stat(stats, "Mature", self.count_vars["mature"], 1, 0, self.colors["green"])
        self.create_stat(stats, "Rotten", self.count_vars["rotten"], 1, 1, self.colors["red"])

        self.summary_label = tk.Label(
            parent,
            textvariable=self.maturity_summary_var,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["panel_hot"],
            fg=self.colors["text"],
            padx=12,
            pady=10,
            wraplength=280,
            justify="left",
        )
        self.summary_label.grid(row=8, column=0, sticky="ew", padx=18, pady=(0, 14))

        distribution = tk.Frame(parent, bg=self.colors["panel"])
        distribution.grid(row=9, column=0, sticky="ew", padx=18, pady=(0, 16))
        distribution.columnconfigure(1, weight=1)
        self.distribution_bars = {}
        self.create_distribution_row(distribution, "immature", self.colors["accent"], 0)
        self.create_distribution_row(distribution, "mature", self.colors["green"], 1)
        self.create_distribution_row(distribution, "rotten", self.colors["red"], 2)

        table_title = tk.Label(
            parent,
            text="Analysis",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"],
        )
        table_title.grid(row=10, column=0, sticky="w", padx=18, pady=(0, 8))

        self.analysis_canvas = tk.Canvas(
            parent,
            bg=self.colors["panel_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            height=260,
        )
        self.analysis_scrollbar = tk.Scrollbar(parent, orient="vertical", command=self.analysis_canvas.yview)
        self.analysis_canvas.configure(yscrollcommand=self.analysis_scrollbar.set)
        self.analysis_canvas.grid(row=11, column=0, sticky="nsew", padx=(18, 0))
        self.analysis_scrollbar.grid(row=11, column=1, sticky="ns", padx=(0, 18))

        self.analysis_frame = tk.Frame(self.analysis_canvas, bg=self.colors["panel_alt"])
        self.analysis_window = self.analysis_canvas.create_window((0, 0), window=self.analysis_frame, anchor="nw")
        self.analysis_frame.bind(
            "<Configure>",
            lambda _event: self.analysis_canvas.configure(scrollregion=self.analysis_canvas.bbox("all")),
        )
        self.analysis_canvas.bind(
            "<Configure>",
            lambda event: self.analysis_canvas.itemconfigure(self.analysis_window, width=event.width),
        )
        self.analysis_canvas.bind(
            "<MouseWheel>",
            lambda event: self.analysis_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"),
        )
        self.create_analysis_header()
        parent.rowconfigure(11, weight=1)

        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.grid(row=12, column=0, sticky="ew", padx=18, pady=(14, 8))

        status = tk.Label(
            parent,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            wraplength=250,
            justify="left",
        )
        status.grid(row=13, column=0, sticky="ew", padx=18, pady=(0, 18))

    def create_section_title(self, parent, title, subtitle, row=0):
        section = tk.Frame(parent, bg=self.colors["panel"])
        section.grid(row=row, column=0, sticky="ew", padx=18, pady=(18, 0))
        section.columnconfigure(0, weight=1)

        tk.Label(
            section,
            text=title,
            font=("OCR A Extended", 11, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["accent"],
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            section,
            text=subtitle,
            font=("Consolas", 9),
            bg=self.colors["panel"],
            fg=self.colors["muted"],
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

    def create_distribution_row(self, parent, label, color, row):
        tk.Label(
            parent,
            text=label.capitalize(),
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"],
            width=9,
            anchor="w",
        ).grid(row=row, column=0, sticky="w", pady=4)

        canvas = tk.Canvas(parent, height=16, bg=self.colors["panel_alt"], highlightthickness=0)
        canvas.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        canvas.create_rectangle(0, 0, 0, 16, fill=color, outline="")
        self.distribution_bars[label] = canvas

        tk.Label(
            parent,
            textvariable=self.distribution_vars[label],
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["panel"],
            fg=color,
            width=5,
            anchor="e",
        ).grid(row=row, column=2, sticky="e", pady=4)

    def create_analysis_header(self):
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        header = tk.Frame(self.analysis_frame, bg=self.colors["grid"])
        header.pack(fill="x", padx=8, pady=(8, 4))
        header.columnconfigure(0, weight=0)
        header.columnconfigure(1, weight=1)
        header.columnconfigure(2, weight=0)
        header.columnconfigure(3, weight=0)

        labels = [
            ("ID", 0, 4),
            ("MATURITY CLASS", 1, 14),
            ("CONF", 2, 7),
            ("RGB SIGNATURE", 3, 18),
        ]
        for text, column, width in labels:
            tk.Label(
                header,
                text=text,
                font=("Consolas", 9, "bold"),
                bg=self.colors["grid"],
                fg=self.colors["accent"],
                width=width,
                anchor="w" if column in (1, 3) else "center",
                padx=8,
                pady=6,
            ).grid(row=0, column=column, sticky="ew")

    def create_analysis_row(self, detection):
        rgb = detection["rgb"]
        label = detection["label"]
        accent = rgb_to_hex(rgb)
        class_color = {
            "immature": self.colors["accent"],
            "mature": self.colors["green"],
            "rotten": self.colors["red"],
        }.get(label, self.colors["yellow"])

        row = tk.Frame(
            self.analysis_frame,
            bg=self.colors["panel"],
            highlightbackground=class_color,
            highlightthickness=1,
        )
        row.pack(fill="x", padx=8, pady=4)
        row.columnconfigure(1, weight=1)

        id_box = tk.Label(
            row,
            text=f"{detection['index']:02d}",
            font=("Consolas", 11, "bold"),
            bg=self.colors["panel_hot"],
            fg=self.colors["text"],
            width=4,
            pady=8,
        )
        id_box.grid(row=0, column=0, sticky="nsw")

        class_frame = tk.Frame(row, bg=self.colors["panel"])
        class_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=7)
        class_frame.columnconfigure(1, weight=1)

        tk.Label(class_frame, bg=class_color, width=2, height=1).grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Label(
            class_frame,
            text=label.upper(),
            font=("OCR A Extended", 10, "bold"),
            bg=self.colors["panel"],
            fg=class_color,
            anchor="w",
        ).grid(row=0, column=1, sticky="ew")

        tk.Label(
            row,
            text=f"{detection['confidence']:.2f}",
            font=("Consolas", 11, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["yellow"],
            width=7,
            pady=8,
        ).grid(row=0, column=2, sticky="e")

        rgb_frame = tk.Frame(row, bg=self.colors["panel"])
        rgb_frame.grid(row=0, column=3, sticky="e", padx=(8, 10), pady=7)

        swatch = tk.Frame(
            rgb_frame,
            bg=accent,
            width=26,
            height=22,
            highlightbackground=self.colors["text"],
            highlightthickness=1,
        )
        swatch.grid(row=0, column=0, sticky="e", padx=(0, 8))
        swatch.grid_propagate(False)

        tk.Label(
            rgb_frame,
            text=f"{rgb[0]:03d} {rgb[1]:03d} {rgb[2]:03d}",
            font=("Consolas", 10, "bold"),
            bg=self.colors["panel"],
            fg=accent,
            width=13,
            anchor="e",
        ).grid(row=0, column=1, sticky="e")

    def create_button(self, parent, text, command, disabled=False):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Consolas", 10, "bold"),
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
            font=("OCR A Extended", 18, "bold"),
            bg=self.colors["panel_alt"],
            fg=self.colors["text"],
        )
        value_label.pack(pady=(8, 0))

        name_label = tk.Label(
            card,
            text=label,
            font=("Consolas", 9),
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
        self.update_dashboard(detections, annotated)
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

    def update_dashboard(self, detections, image):
        self.model_status_var.set("Model loaded")
        image_name = self.selected_path.name if self.selected_path else "Loaded image"
        self.image_info_var.set(f"{image_name}\n{image.width} x {image.height} px")

        if not detections:
            self.avg_confidence_var.set("--")
            self.dominant_rgb_var.set("R: --  G: --  B: --")
            self.rgb_swatch.configure(bg="#334652")
            self.draw_confidence_meter(0)
            self.draw_rgb_meter((0, 0, 0))
            self.maturity_summary_var.set("No maize detected above the selected threshold.")
            return

        avg_conf = sum(detection["confidence"] for detection in detections) / len(detections)
        self.avg_confidence_var.set(f"{avg_conf:.2f}")
        self.draw_confidence_meter(avg_conf)

        avg_rgb = tuple(
            int(sum(detection["rgb"][channel] for detection in detections) / len(detections))
            for channel in range(3)
        )
        self.dominant_rgb_var.set(f"R: {avg_rgb[0]}  G: {avg_rgb[1]}  B: {avg_rgb[2]}")
        self.rgb_swatch.configure(bg=rgb_to_hex(avg_rgb))
        self.draw_rgb_meter(avg_rgb)

        counts = count_labels(detections)
        dominant_label = max(counts, key=counts.get)
        self.maturity_summary_var.set(
            f"Most detected class: {dominant_label.capitalize()} "
            f"({counts[dominant_label]} of {len(detections)} objects)"
        )

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
        self.create_analysis_header()

        for detection in detections:
            self.create_analysis_row(detection)

        if not detections:
            empty = tk.Label(
                self.analysis_frame,
                text="NO DETECTION DATA",
                font=("Consolas", 10, "bold"),
                bg=self.colors["panel_alt"],
                fg=self.colors["muted"],
                pady=18,
            )
            empty.pack(fill="x", padx=8, pady=6)

    def update_counts(self, detections):
        counts = count_labels(detections)

        self.count_vars["total"].set(str(len(detections)))
        for label in ("immature", "mature", "rotten"):
            self.count_vars[label].set(str(counts[label]))
        self.update_distribution(counts, len(detections))

    def update_distribution(self, counts, total):
        for label, canvas in self.distribution_bars.items():
            percent = 0 if total == 0 else counts[label] / total
            self.distribution_vars[label].set(f"{int(percent * 100)}%")
            canvas.delete("all")
            width = max(canvas.winfo_width(), 180)
            canvas.create_rectangle(0, 0, width, 16, fill=self.colors["panel_alt"], outline="")
            fill_width = int(width * percent)
            color = {
                "immature": self.colors["accent"],
                "mature": self.colors["green"],
                "rotten": self.colors["red"],
            }[label]
            canvas.create_rectangle(0, 0, fill_width, 16, fill=color, outline="")

    def clear_results(self):
        self.selected_path = None
        self.annotated_image = None
        self.heatmap_image = None
        self.file_var.set("No image selected")
        self.image_info_var.set("No image loaded")
        self.avg_confidence_var.set("--")
        self.dominant_rgb_var.set("R: --  G: --  B: --")
        self.maturity_summary_var.set("Waiting for analysis")
        self.rgb_swatch.configure(bg="#334652")
        self.draw_confidence_meter(0)
        self.draw_rgb_meter((0, 0, 0))
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


def count_labels(detections):
    counts = {"immature": 0, "mature": 0, "rotten": 0}
    for detection in detections:
        label = detection["label"]
        if label in counts:
            counts[label] += 1
    return counts


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def point_on_circle(cx, cy, radius, angle_degrees):
    angle = radians(angle_degrees)
    return cx + radius * cos(angle), cy - radius * sin(angle)


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

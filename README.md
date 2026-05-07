# Colorimetric Blue Maize Detector

This project is a Tkinter desktop application for detecting blue maize maturity using a YOLOv8 model. It displays the detected maize objects, class labels, confidence values, average RGB values, and a heatmap visualization.

## Features

- Upload maize images from your computer.
- Detect maize with the trained YOLO model.
- Classify detected maize as `immature`, `mature`, or `rotten`.
- Show average RGB values for each detected maize object.
- Display an annotated detection image and a color heatmap.
- Review dashboard cards for model status, image size, average confidence, and dominant RGB.
- Read semicircle gauge meters for average confidence and RGB channel strength.
- View colored maturity distribution bars for immature, mature, and rotten detections.
- Inspect each analysis row with a class indicator and RGB color swatch.
- Use the cyberpunk-style dashboard layout for clearer visual scanning.
- Adjust the confidence threshold.
- Resize the interface with draggable panels.
- Zoom in, zoom out, fit images to the view, and scroll large images.
- Save the annotated detection result.

## Project Structure

```text
Colorimetric-Blue-Maize/
  BLUE_MAIZE/
    CodeBlueMaize.py
    bestv26.pt
  requirements.txt
  README.md
```

The app expects the trained model file to be here:

```text
BLUE_MAIZE/bestv26.pt
```

## Requirements

- Python 3.10 or newer is recommended.
- A working webcam is not required because the app uses uploaded image files.
- The trained YOLO model file must exist at `BLUE_MAIZE/bestv26.pt`.

## Setup on Windows

Open PowerShell in the project folder:

```powershell
cd C:\Users\afifi\Desktop\FYP\Colorimetric-Blue-Maize
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

## Run the Application

From the project folder, run:

```powershell
python BLUE_MAIZE\CodeBlueMaize.py
```

The application window should open. Click `Upload Image`, choose a maize image, and wait for the analysis to finish.

## How to Use

1. Click `Upload Image`.
2. Select a `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` image.
3. Review the detection result and heatmap.
4. Check the top dashboard cards for image size, average confidence, dominant RGB, and gauge meters.
5. Review the maturity distribution bars and analysis table.
6. Adjust the confidence threshold if needed.
7. Use `+`, `-`, the zoom slider, or `Fit to View` to adjust the image size.
8. Use the scrollbars to inspect large images after zooming in.
9. Drag the panel dividers to resize the detection, heatmap, and control areas.
10. Click `Save Result` to save the annotated image.
11. Click `Clear` to reset the screen.

## Check for Syntax Errors

Use this command:

```powershell
python -m py_compile BLUE_MAIZE\CodeBlueMaize.py
```

If the command returns without an error, the Python file syntax is valid.

## Troubleshooting

If Python says `No module named ultralytics`, install the dependencies again:

```powershell
pip install -r requirements.txt
```

If the app says the model file is missing, make sure this file exists:

```text
BLUE_MAIZE/bestv26.pt
```

If OpenCV fails to import, reinstall it:

```powershell
pip install opencv-python
```

If you see a Windows path `unicodeescape` error, use forward slashes or `Path` in Python paths. This project already uses `Path(__file__)` so the model is loaded relative to `CodeBlueMaize.py`.

## Push Changes to GitHub

To push all changed files:

```powershell
git status
git add .
git commit -m "Update blue maize app"
git push
```

To push one file only:

```powershell
git add BLUE_MAIZE\CodeBlueMaize.py
git commit -m "Update app interface"
git push
```

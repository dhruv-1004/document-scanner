# Document Scanner v2

A Python + OpenCV document scanner that automatically detects a document in an image, corrects its perspective, and produces a clean scanned version.

## Features

- Automatic document boundary detection
- Multiple detection strategies for difficult images
  - Canny edge detection with multiple thresholds
  - Adaptive thresholding
  - HSV saturation analysis
  - LAB lightness analysis
- Perspective correction using a four-point transform
- EXIF image-rotation correction
- Adaptive thresholding for a clean black-and-white scan
- Optional colour output
- Webcam scanning mode
- Debug mode for inspecting detection strategies
- Fallback to the original frame when no confident document boundary is found

## Tech Stack

- Python
- OpenCV
- NumPy
- Pillow (optional, for EXIF rotation correction)

## Project Structure

```text
document-scanner/
├── scanner.py
├── all_compare.jpg
├── paper_scanned_v2.jpeg
├── price_scanned_v2.jpeg
└── p3_scanned_v2.jpeg
```

## Installation

Install the required Python packages:

```bash
pip install opencv-python numpy pillow
```

## Usage

### Scan an image

```bash
python scanner.py --image path/to/image.jpg
```

The processed image will be saved beside the input image with `_scanned_v2` added to its filename.

### Keep the scanned result in colour

```bash
python scanner.py --image path/to/image.jpg --color
```

### Enable debug output

```bash
python scanner.py --image path/to/image.jpg --debug
```

Debug mode saves the intermediate masks produced by the different document-detection strategies.

### Use the webcam

```bash
python scanner.py --webcam
```

Press **S** to capture and scan the current frame, or **Q** to quit.

## How It Works

1. The input image is loaded and its EXIF orientation is corrected when Pillow is available.
2. The image is resized for efficient document detection.
3. Several image-processing strategies are applied to identify possible document boundaries.
4. Candidate contours are evaluated using area coverage and rectangularity.
5. The best document quadrilateral is selected.
6. A perspective transformation straightens the document.
7. Adaptive thresholding converts the result into a clean scanned document.
8. If no reliable document is detected, the original image is used as a fallback.

## Example

The repository includes sample input/output images that can be used to compare scanning results and evaluate the detection quality.

## Future Improvements

- Real-time document boundary stabilization for webcam mode
- Automatic brightness and contrast correction
- Shadow and glare removal
- OCR integration
- PDF export and multi-page scanning
- Mobile/web interface

## License

This project is provided for educational and personal use.

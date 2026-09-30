<h1 align="center">ImgsProc: Image Processing from Scratch</h1>

<p align="center">
  Classic digital image processing algorithms implemented pixel by pixel in Python, served by a Flask API
  and consumed by an Android app. Built for the Digital Image Processing course at the
  Federal University of Rondônia (UNIR).
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white" alt="Python 3">
  <img src="https://img.shields.io/badge/NumPy-arrays-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/OpenCV-I%2FO%20only-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Android-Java-3DDC84?logo=android&logoColor=white" alt="Android">
</p>

## About

The goal of this project was to understand image processing by writing the algorithms ourselves instead of calling
ready made library functions. Every technique walks over the image with explicit loops and applies the math from the
course directly to each pixel. OpenCV is only used to read, show and save images, to convert them to grayscale and,
in the histogram functions, to compute the histogram.

The project has three parts:

1. **Algorithm scripts**: one Python file per technique, used to study and test each one on its own
2. **Flask server** (`server.py`): gathers the algorithms behind a single HTTP endpoint
3. **Android app** (`Customizaesdeimagens/`): lets the user pick a technique and a photo from the gallery and send
   it to the server

## Implemented techniques

| Category | Techniques |
| --- | --- |
| Intensity transformations | Negative, Thresholding, Logarithmic transformation, Power (gamma) transformation, Contrast stretching |
| Histogram | Histogram expansion, Histogram equalization |
| Spatial filtering | Average filter, Median filter, Convolution with a custom kernel |
| Sharpening and edges | High boost filtering, Laplacian sharpening, Sobel edge detection |
| Local contrast | Adaptive contrast control using the mean and standard deviation of a 4, diagonal or 8 pixel neighborhood |
| Geometric transformations | Rotation (with neighbor interpolation to fill gaps), Rebound (transpose or 90° turn), Shear, Zoom in, Zoom out |
| Image blending | Uniform cross dissolve, Non uniform cross dissolve |

## How it works

```
Android app → picks a technique and a photo
            → POST /process_image (multipart: image + position)
Flask server → decodes the image with OpenCV
             → runs the algorithm selected by position
             → returns the processed JPEG
```

The `position` field is the index of the technique in the app's list:

| # | Technique | # | Technique |
| --- | --- | --- | --- |
| 0 | Average filter | 10 | Power transformation (`c=1`, `y=2`) |
| 1 | High boost (`k=2`) | 11 | Rebound (transpose) |
| 2 | Contrast control (`c=1`, 4 neighbors) | 12 | Rotation (45°) |
| 3 | Contrast stretching (0 to 255) | 13 | Sobel edge detection (`t=50`) |
| 4 | Histogram expansion | 14 | Thresholding (`k=128`) |
| 5 | Logarithmic transformation (`c=1`) | 15 | Uniform cross dissolve |
| 6 | Median filter | 16 | Zoom in (2x) |
| 7 | Shear (`sx=0.5`, `sy=0.5`) | 17 | Zoom out (0.5x) |
| 8 | Negative | 18 | Laplacian sharpening |
| 9 | Non uniform cross dissolve | | |

## Project structure

| Path | Description |
| --- | --- |
| `negative.py`, `thresholding.py`, `rotation.py`, ... | One script per technique (21 files) |
| `QA.py`, `QB.py` | Modules that gather all the functions in a single file |
| `server.py` | Flask API that applies the selected technique to an uploaded image |
| `Customizaesdeimagens/` | Android Studio project ("Customizações de imagens") |

### Android app

Written in Java, targeting Android 7.0 (API 24) up to Android 14 (API 34).

- **Start screen** (`AcitivityInicio`): app logo and a button to begin
- **Technique list** (`MainActivity`): a `RecyclerView` with the 19 available techniques
- **Processing screen** (`MainActivity2`): pick a photo from the gallery and send it to the server

The app talks to `http://10.0.2.2:5000`, the address the Android emulator uses to reach the host machine.

## Getting started

### Python scripts and server

```bash
git clone https://github.com/Wyllgner/ImgsProc-Project.git
cd ImgsProc-Project
pip install numpy opencv-python flask pillow
```

Each technique script defines its function so it can be imported and tested on any image:

```python
import cv2
from thresholding import thresholding

img = cv2.imread("your_image.png")
cv2.imshow("Result", thresholding(img, 128))
cv2.waitKey(0)
```

To start the API, create the upload folder and run the server on port 5000:

```bash
mkdir uploads
python server.py
```

### Android app

1. Open the `Customizaesdeimagens` folder in Android Studio
2. Let Gradle sync the project
3. Run it on an Android emulator while `server.py` is running on the same computer

## Project status

This is an academic project from early 2024 and it was left as it was at the end of the course. Anyone who wants to
run it should know that:

- `server.py` does not start as is: line 68 ends with `;` instead of `:` (`elif position == 18;`), and the same
  branch passes an undefined `f` to `laplacian_sharpening` instead of `original_image`
- Techniques that need a second image or extra input (both cross dissolves) receive `None` from the server and
  fail when called through the API
- The app sends the image and logs the server response, but does not display the processed image yet
- The server address is fixed to the emulator's `10.0.2.2`; a physical phone needs the computer's local IP instead
- `negative.py` and `average_filter.py` still load test images from a local Windows path when imported or run;
  change the path before using them
- Rebound only works on square images, since it swaps rows and columns in place
- Most algorithms convert the input to grayscale first, so the output is always a single channel image
- Pure Python loops are slow on large photos; small images (256 or 512 pixels wide) give the best experience

## Authors

- **Wyllgner França** ([@Wyllgner](https://github.com/Wyllgner))
- **João** ([@majinaru](https://github.com/majinaru))

Federal University of Rondônia (UNIR), 2024.

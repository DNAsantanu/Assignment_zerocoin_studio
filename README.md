# Assignment_zerocoin_studio

This project implements a two-stage GPU-accelerated image processing pipeline using OpenCL via PyOpenCL:

Per-channel 3×3 Gaussian Blur and Logarithmic Tone Mapping on Luminance

The tool reads an input image (PNG/BMP), applies both processing stages on the GPU, and outputs an enhanced image with adjusted luminance while preserving color ratios and transparency.

1d_averaging.ipynb and image_data_handling.ipynb are the two jupyter notebook where the first two part of the assignment Environment Setup & OpenCL Fundamentals and Image Data Handling respectively done.
## File Structure

```bash
.
├── host_code.py          # Python script (PyOpenCL) that drives the GPU pipeline
├── kernels.cl            # OpenCL kernel source (Gaussian blur + tone mapping)
├── input_images          # Sample input image (RGBA)
├── output_images         # Final processed images are stored there
├── assignment.pdf        # Assignment description 
├── report.tex            # One-page LaTeX report describing the project
├── requirements.txt
└── README.md             # Project overview and instructions

```

## Clone this repo and 1st create and enviroment using pip or conda and run the requirements.txt to install the required libraries or manually install using the under-given cmd.

## Requirements
Python 3.8+

PyOpenCL

NumPy

Pillow (PIL)

OpenCL runtime (supports Apple M1 or any OpenCL-compatible GPU)

```bash
pip install pyopencl numpy pillow
```

## How to Run
* Place the input image in the Input_images folder.

Run the script:
```bash
python host_code.py
```
The output image will be saved in output_images folder.

## Features
* GPU-based 3×3 Gaussian blur (applied per channel)

* Logarithmic tone mapping with configurable max_luminance

* Handles edge cases with coordinate clamping

* Preserves original alpha channel

* Fully parallelized kernel execution per pixel

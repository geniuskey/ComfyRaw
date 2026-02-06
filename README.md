# ComfyRaw

CPU-based OpenCV Node Editor for Image/Video Processing

ComfyRaw is a visual node-based editor for image and video processing, built on the ComfyUI infrastructure but completely stripped of generative AI features. It provides a pure OpenCV/NumPy processing pipeline that runs entirely on CPU.

## Features

- **Node-based visual workflow** - Drag and drop nodes to create processing pipelines
- **CPU-only processing** - No GPU required, runs on any machine
- **40+ OpenCV nodes** - Filters, edge detection, morphology, color adjustments, and more
- **Video support** - Load, process, and save video files
- **Batch processing** - Process multiple images at once
- **Real-time preview** - See results as you build your workflow

## Requirements

- Python 3.10+
- OpenCV 4.8+
- NumPy 1.25+

## Installation

### From Source

```bash
git clone https://github.com/geniuskey/ComfyRaw.git
cd ComfyRaw
pip install -r requirements.txt
```

## Quick Start

```bash
# Start the server
python main.py

# Open in browser
# http://127.0.0.1:8188
```

## Available Nodes

### Basic Image Operations
| Node | Description |
|------|-------------|
| LoadImage | Load image from file |
| SaveImage | Save image to file |
| PreviewImage | Preview image in UI |
| ImageScale | Resize image to specific dimensions |
| ImageScaleBy | Scale image by factor |
| ImageFlip | Flip horizontally/vertically |
| ImageRotate | Rotate 90/180/270 degrees |
| ImageCrop | Crop to region |
| ImageInvert | Invert colors |
| ImageBatch | Combine images into batch |
| EmptyImage | Create solid color image |

### Filters
| Node | Description |
|------|-------------|
| GaussianBlur | Gaussian blur filter |
| MedianBlur | Median blur filter |
| BilateralFilter | Edge-preserving blur |
| BoxBlur | Box blur filter |
| Sharpen | Sharpen image |
| UnsharpMask | Unsharp mask sharpening |

### Edge Detection
| Node | Description |
|------|-------------|
| CannyEdge | Canny edge detection |
| SobelEdge | Sobel edge detection |
| LaplacianEdge | Laplacian edge detection |
| FindContours | Find contours in image |
| DrawContours | Draw contours on image |

### Morphology
| Node | Description |
|------|-------------|
| Erode | Morphological erosion |
| Dilate | Morphological dilation |
| MorphOpen | Morphological opening |
| MorphClose | Morphological closing |
| MorphGradient | Morphological gradient |

### Color Processing
| Node | Description |
|------|-------------|
| ColorConvert | Convert color spaces (RGB, HSV, LAB, Gray) |
| AdjustBrightness | Adjust brightness |
| AdjustContrast | Adjust contrast |
| AdjustHueSaturation | Adjust hue and saturation |
| HistogramEqualize | Histogram equalization |
| ColorBalance | Adjust RGB balance |

### Threshold
| Node | Description |
|------|-------------|
| Threshold | Binary threshold |
| AdaptiveThreshold | Adaptive threshold |

### Composite
| Node | Description |
|------|-------------|
| Blend | Blend two images |
| AlphaComposite | Alpha compositing with mask |
| MaskApply | Apply mask to image |
| ChannelSplit | Split RGB channels |
| ChannelMerge | Merge channels to RGB |

### Drawing
| Node | Description |
|------|-------------|
| DrawText | Draw text on image |
| DrawRectangle | Draw rectangle |
| DrawCircle | Draw circle |
| DrawLine | Draw line |

### Analysis
| Node | Description |
|------|-------------|
| ImageInfo | Get image dimensions and stats |
| Histogram | Generate histogram visualization |

## Project Structure

```
ComfyRaw/
├── main.py                 # Entry point
├── server.py               # Web server
├── nodes.py                # Basic image nodes
├── execution.py            # Node execution engine
├── comfy/                  # Core modules
│   ├── model_management.py # Memory management
│   ├── utils.py            # Image utilities
│   └── ...
├── comfy_cv/               # OpenCV processing module
│   ├── image.py            # ImageProcessor class
│   ├── video.py            # VideoProcessor class
│   ├── memory.py           # MemoryManager class
│   └── types.py            # Type definitions
├── comfy_extras/
│   └── nodes_opencv.py     # OpenCV nodes
└── tests-unit/             # Unit tests
```

## Development

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific module tests
python -m pytest tests-unit/comfy_cv/ -v

# Run with coverage
python -m pytest tests-unit/ --cov=comfy_cv --cov-report=html
```

### Adding Custom Nodes

Create a Python file in `custom_nodes/` directory:

```python
import numpy as np

class MyCustomNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, image, value):
        result = np.clip(image * value, 0, 1)
        return (result.astype(np.float32),)

NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode,
}
```

## Data Types

| Type | Format | Description |
|------|--------|-------------|
| IMAGE | `np.ndarray` (B, H, W, C) float32 0-1 | Batch of RGB images |
| MASK | `np.ndarray` (B, H, W) float32 0-1 | Batch of grayscale masks |
| INT | Integer | Integer value |
| FLOAT | Float | Float value |
| STRING | Text | Text string |

## Keyboard Shortcuts

| Keybind | Action |
|---------|--------|
| Ctrl + Enter | Queue workflow |
| Ctrl + Z / Y | Undo / Redo |
| Ctrl + S | Save workflow |
| Ctrl + O | Load workflow |
| Ctrl + A | Select all nodes |
| Delete | Delete selected nodes |
| Space + Drag | Pan canvas |
| Ctrl + C / V | Copy / Paste nodes |
| Double-Click | Open node search |

## API

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/prompt` | POST | Execute workflow |
| `/history` | GET | Get execution history |
| `/system_stats` | GET | Get system memory info |
| `/view` | GET | View output images |

### WebSocket

Connect to `ws://127.0.0.1:8188/ws` for real-time updates.

## Configuration

```bash
python main.py --help

Options:
  --listen HOST        Listen address (default: 127.0.0.1)
  --port PORT          Listen port (default: 8188)
  --input-directory    Input directory path
  --output-directory   Output directory path
```

## License

GPL-3.0 License - See [LICENSE](LICENSE) file

## Credits

Based on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous.
Modified by removing all generative AI features and adding pure OpenCV processing capabilities.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python run_tests.py`
5. Submit a pull request

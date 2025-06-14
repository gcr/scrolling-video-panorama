# Scrolling Video Panorama

Convert screen recordings of scrolling content into panoramic images. This tool analyzes video frames to detect scrolling motion and stitches them together into a single long panoramic image.

Requires a CLEAN screen recording where you are only scrolling vertically!

Untested when you scroll down and up at the same time!

Best for web pages, long documents, long group chat messages, etc.

## Usage

```bash
$ uv run main.py --help

 Usage: main.py [OPTIONS]

   Options
      --video-path    TEXT     [default: ScreenRecording_06-11-2025 13-53-32_1.MP4]                                  │
      --output-path   TEXT     [default: output.png]
      --num-buckets   INTEGER  Number of buckets for pixel median [default: 20]
      --crop-pixels   INTEGER  Height of header and footer to crop [default: 512]
      --help                   Show this message and exit.
```


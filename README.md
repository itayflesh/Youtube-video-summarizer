# YouTube Video Summarizer

## Description

https://drive.google.com/file/d/1pBSwhTjMXOAw_HiWYPA8ssE6ajrHMHXZ/view?usp=share_link

This project is a Python-based YouTube video summarizer that automatically creates a visual and textual summary of a YouTube video based on a user-provided subject. The program searches YouTube for a video matching the subject, downloads it, extracts key frames, performs text recognition on these frames, and creates an animated GIF summary along with extracted text.

## Installation

To run this project, you need to have Python installed on your system. Follow these steps to set up the project:

1. Clone this repository to your local machine.
2. Install the required Python packages by running:

```
pip install numpy pytube opencv-python scenedetect Pillow easyocr imagehash IPython
```

3. Ensure you have FFmpeg installed on your system, as it's required for video processing.

## How It Works

1. The user inputs a subject for the video search.
2. The program searches YouTube for videos on that subject and downloads the first video under 10 minutes.
3. It then processes the video to detect major scenes and attractive frames.
4. Key frames are extracted and saved as images.
5. Text is extracted from these images using OCR (Optical Character Recognition).
6. A watermark is added to each image.
7. An animated GIF is created from the extracted frames.
8. The program displays the GIF and prints the extracted text from all frames.

## Usage

Run the script using Python:

```
python youtubesummarizer.py
```

When prompted, enter a subject for the video you want to summarize.

## Technologies Used

- Python
- pytube: For YouTube video search and download
- OpenCV (cv2): For video and image processing
- scenedetect: For detecting scene changes in the video
- PIL (Python Imaging Library): For image handling and GIF creation
- easyocr: For text extraction from images
- imagehash: For comparing image similarity
- numpy: For numerical operations
- IPython: For displaying images (optional, used in some versions)


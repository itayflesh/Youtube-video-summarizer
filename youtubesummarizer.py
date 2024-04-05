import random
from pytube import Search, YouTube
import certifi
import ssl
from pytube.exceptions import AgeRestrictedError
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect import VideoManager
import cv2
from scenedetect import detect, ContentDetector, ThresholdDetector

ssl._create_default_https_context = ssl._create_unverified_context

from scenedetect.detectors import ContentDetector
import cv2

def detect_major_frames(video_path, subject):
    
    threshold=50.0
    min_scene_len=30
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Initialize a ContentDetector with adjustable parameters
    detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)

    # Initialize variables to store major scenes and attractive frames
    major_scenes = []
    attractive_frames = []

    # Open the video file
    if cap.isOpened():
        # Create a frame skip counter
        frame_skip = 0
        frame_num = 0
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame every 10 frames
            frame_skip += 1
            frame_num += 1
            if frame_skip % 10 == 0:
                # Detect content changes in the frame
                scene_list = detector.process_frame(frame_num, frame)

                # If a scene change is detected, save the frame as a major scene
                if scene_list:
                    major_scenes.append(frame)

                # Check for attractiveness of the frame (e.g., based on brightness or colorfulness)
                # For demonstration purposes, let's consider frames with high brightness as attractive
                if is_attractive_frame(frame):
                    attractive_frames.append(frame)

        # Release the VideoCapture object
        cap.release()

        # Save key frames for major scenes
        for i, frame in enumerate(major_scenes[:3]):
            key_frame_path = f'{subject}_major_scene_{i+1}.jpg'
            cv2.imwrite(key_frame_path, frame)
            print(f'Saved key frame for major scene {i+1} of {subject}')

        # Save attractive frames
        for i, frame in enumerate(attractive_frames[:3]):
            key_frame_path = f'{subject}_attractive_frame_{i+1}.jpg'
            cv2.imwrite(key_frame_path, frame)
            print(f'Saved attractive frame {i+1} of {subject}_{i+1}')
    else:
        print(f"Error: Unable to open video file {video_path}")

def is_attractive_frame(frame, brightness_threshold=200):
    # Convert frame to grayscale for simplicity
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the average brightness of the frame
    average_brightness = cv2.mean(gray_frame)[0]

    # Check if the frame is attractive based on brightness threshold
    return average_brightness > brightness_threshold


def calculate_attractiveness(frame_img):
    # Placeholder function to calculate attractiveness score
    # You can implement your own algorithm to evaluate visual appeal
    # Here, we simply return a random score between 0 and 1 as an example
    return random.uniform(0, 1)


def search_and_download(subject, max_downloads=1):
    # Search for videos on YouTube
    search_results = Search(subject).results

    downloaded_count = 0

    for video in search_results:
        if video.length >= 120:
            continue

        print(f"Downloading video {downloaded_count + 1}: {video.title}")
        print(f"URL: {video.watch_url}")
        print(f"Duration: {video.length} seconds")

        try:
            # Download the video
            youtube = YouTube(video.watch_url)
            video_stream = youtube.streams.get_highest_resolution()
            video_path = f"{subject}_{downloaded_count + 1}.mp4"
            video_stream.download(filename=video_path)
            print(f"Video {downloaded_count + 1} downloaded successfully!")
            
            # Detect major frames in the downloaded video
            detect_major_frames(video_path, subject)

            downloaded_count += 1
            if downloaded_count == max_downloads:
                break
        except AgeRestrictedError:
            print(f"Video {downloaded_count + 1} is age-restricted and cannot be downloaded. Skipping...")
        except Exception as e:
            print(f"An error occurred while downloading video {downloaded_count + 1}: {str(e)}")

    if downloaded_count == 0:
        print("No videos found less than 2 minutes.")

# Ask the user for a subject
subject = input("Enter a subject to search on YouTube: ")

# Search YouTube and download up to 5 videos
search_and_download(subject)

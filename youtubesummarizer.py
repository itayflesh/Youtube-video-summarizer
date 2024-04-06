import numpy as np
from pytube import Search, YouTube
import certifi
import ssl
from pytube.exceptions import AgeRestrictedError
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect import VideoManager
import cv2
from scenedetect import detect, ContentDetector, ThresholdDetector
import imagehash
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

def detect_major_frames(video_path, subject):
    threshold = 50.0
    min_scene_len = 30
    colorfulness_threshold = 25
    phash_threshold = 10
    black_threshold = 20

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Initialize a ContentDetector with adjustable parameters
    detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)

    # Initialize variables to store major scenes, attractive frames, and their pHashes
    major_scenes = {}
    attractive_frames = {}
    phash_dict = {}

    # Open the video file
    if cap.isOpened():
        # Create a frame skip counter
        frame_skip = 0
        frame_num = 0
        current_scene = 0
        scene_start_frame = 0

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

                # If a scene change is detected, save the middle frame of the previous scene
                if scene_list:
                    if current_scene > 0:
                        scene_end_frame = frame_num - 1
                        scene_middle_frame = (scene_start_frame + scene_end_frame) // 2
                        cap.set(cv2.CAP_PROP_POS_FRAMES, scene_middle_frame)
                        _, middle_frame = cap.read()
                        if not is_black_frame(middle_frame, black_threshold):
                            phash = imagehash.phash(Image.fromarray(middle_frame))
                            if current_scene not in phash_dict or not is_similar_phash(phash, phash_dict[current_scene], phash_threshold):
                                major_scenes[current_scene] = middle_frame
                                phash_dict[current_scene] = phash
                    current_scene += 1
                    scene_start_frame = frame_num

                # Check for attractiveness of the frame
                if is_attractive_frame(frame, colorfulness_threshold):
                    phash = imagehash.phash(Image.fromarray(frame))
                    if current_scene not in attractive_frames or not is_similar_phash(phash, phash_dict[current_scene], phash_threshold):
                        attractive_frames[current_scene] = frame
                        phash_dict[current_scene] = phash

        # Release the VideoCapture object
        cap.release()

        # Save key frames for major scenes (middle frames) and attractive frames from each scene
        for scene, frame in major_scenes.items():
            if scene not in attractive_frames:
                key_frame_path = f'{subject}_major_scene_{scene}.jpg'
                cv2.imwrite(key_frame_path, frame)
                print(f'Saved key frame for major scene {scene} of {subject}')

        for scene, frame in attractive_frames.items():
            key_frame_path = f'{subject}_attractive_frame_{scene}.jpg'
            cv2.imwrite(key_frame_path, frame)
            print(f'Saved attractive frame for scene {scene} of {subject}')
    else:
        print(f"Error: Unable to open video file {video_path}")

def is_attractive_frame(frame, colorfulness_threshold):
    colorfulness = calculate_colorfulness(frame)
    return colorfulness > colorfulness_threshold

def calculate_colorfulness(frame):
    (B, G, R) = cv2.split(frame.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    colorfulness = stdRoot + (0.3 * meanRoot)
    return colorfulness

def is_similar_phash(phash1, phash2, threshold):
    return phash1 - phash2 <= threshold

def is_black_frame(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold

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

import os
import yt_dlp
import cv2
import json
import math
from PIL import Image
import pytesseract
from moviepy.editor import VideoFileClip
import whisper

# Constants
VIDEO_FILE = "video.mp4"
AUDIO_FILE = "audio.mp3"
IMAGE_FRAMES_DIR = "image_frames"
JSON_FILE = "data.json"
MAX_FRAMES_SHORT = 30  # Max frames for videos ≤ 1 min
MAX_FRAMES_LONG = 300  # Max frames for videos > 1 min


# 1️⃣ Download Instagram Reel
def download_instagram_reel(url, output_filename=VIDEO_FILE):
    """Download Instagram Reel and overwrite the existing file."""
    if os.path.exists(output_filename):
        os.remove(output_filename)  # Remove existing file

    options = {
        'outtmpl': output_filename,  # Fixed filename ensures overwrite
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.extract_info(url, download=True)
        print(f"Downloaded: {output_filename}")


# 2️⃣ Optimized Frame Skip Calculation
def determine_frame_skip(video_path):
    """Calculate optimal frame skipping interval based on video length."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return 100  # Default

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps  # Video duration in seconds

    print(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f} sec")

    if duration <= 60:  # Short video (≤ 1 min)
        frame_skip = max(1, total_frames // MAX_FRAMES_SHORT)
    else:  # Long videos (> 1 min)
        frame_skip = max(50, total_frames // MAX_FRAMES_LONG)

    cap.release()
    return frame_skip


# 3️⃣ Extract Optimized Frames
def extract_frames_from_video(video_path):
    """Extract frames from a video at an optimized interval."""
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return False

    frame_skip = determine_frame_skip(video_path)
    os.makedirs(IMAGE_FRAMES_DIR, exist_ok=True)

    # Clear previous frames
    for file in os.listdir(IMAGE_FRAMES_DIR):
        os.remove(os.path.join(IMAGE_FRAMES_DIR, file))

    cap = cv2.VideoCapture(video_path)
    index = 0
    frame_count = 0
    max_frames = MAX_FRAMES_SHORT if determine_frame_skip(video_path) <= 60 else MAX_FRAMES_LONG

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        if index % frame_skip == 0:
            frame_path = os.path.join(IMAGE_FRAMES_DIR, f"frame{index}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            print(f"Extracted: {frame_path}")

        index += 1

    cap.release()
    print(f"Frame extraction complete. Extracted {frame_count} frames.")
    return True


# 4️⃣ Extract Text from Frames
def extract_text_from_frames():
    """Extract text from frames using Tesseract OCR and store in JSON."""
    extracted_texts = []

    for filename in sorted(os.listdir(IMAGE_FRAMES_DIR)):
        image_path = os.path.join(IMAGE_FRAMES_DIR, filename)
        text = pytesseract.image_to_string(Image.open(image_path)).strip()
        if text:
            extracted_texts.append(text)

    text_output = "\n".join(extracted_texts)
    update_json_file({"extracted_text": text_output})

    print(f"Extracted text saved to {JSON_FILE}")


# 5️⃣ Extract & Transcribe Audio
def extract_audio_from_video(video_path, audio_path):
    """Extract audio and save as MP3. If no audio, skip transcription."""
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return False

    video_clip = VideoFileClip(video_path)

    if video_clip.audio is None:
        print("No audio detected in video.")
        update_json_file({"transcription": ""})  # Save empty transcription
        video_clip.close()
        return False

    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec="mp3", verbose=False)

    audio_clip.close()
    video_clip.close()
    del video_clip  # Force cleanup

    print("Audio extraction successful!")
    return True


def transcribe_audio_with_whisper(audio_path):
    """Transcribe audio using Whisper and save results to JSON."""
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found!")
        return

    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path, language="en", verbose=False)
    update_json_file({"transcription": result["text"]})

    print("Transcription saved to JSON.")


# 6️⃣ Update JSON File
def update_json_file(new_data):
    """Update JSON file while ensuring only 'transcription' and 'extracted_text' exist."""
    allowed_keys = {"transcription", "extracted_text"}

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data.update({k: v for k, v in new_data.items() if k in allowed_keys})

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Updated JSON file.")


# 7️⃣ Run the Full Pipeline
if __name__ == "__main__":
    reel_url = input("Enter Instagram Reel URL: ")
    download_instagram_reel(reel_url)

    if extract_frames_from_video(VIDEO_FILE):
        extract_text_from_frames()

    if extract_audio_from_video(VIDEO_FILE, AUDIO_FILE):
        transcribe_audio_with_whisper(AUDIO_FILE)

    print("Processing completed!")

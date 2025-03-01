import os
import yt_dlp
import cv2
import json
from PIL import Image
import pytesseract
from moviepy.editor import VideoFileClip
import whisper

# Constants
VIDEO_FILE = "video.mp4"
AUDIO_FILE = "audio.mp3"
IMAGE_FRAMES_DIR = "image_frames"
JSON_FILE = "data.json"
MAX_FRAMES_SHORT = 15
MAX_FRAMES_LONG = 50

# 1️⃣ Download Instagram Reel
def download_instagram_reel(url, output_filename=VIDEO_FILE):
    """Download Instagram Reel and overwrite the existing file."""
    if os.path.exists(output_filename):
        os.remove(output_filename)

    options = {
        'outtmpl': output_filename,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.extract_info(url, download=True)
        print(f"Downloaded: {output_filename}")

# 2️⃣ Extract Frames
def extract_frames_from_video(video_path):
    """Extract frames from a video at an optimized interval."""
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return False

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    frame_skip = max(1, total_frames // MAX_FRAMES_SHORT if duration <= 60 else total_frames // MAX_FRAMES_LONG)

    os.makedirs(IMAGE_FRAMES_DIR, exist_ok=True)
    for file in os.listdir(IMAGE_FRAMES_DIR):
        os.remove(os.path.join(IMAGE_FRAMES_DIR, file))

    index, frame_count = 0, 0
    max_frames = MAX_FRAMES_SHORT if duration <= 60 else MAX_FRAMES_LONG

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
    return True

# 3️⃣ Extract Text
def extract_text_from_frames():
    """Extract text from frames using Tesseract OCR."""
    extracted_texts = []
    
    for filename in sorted(os.listdir(IMAGE_FRAMES_DIR)):
        image_path = os.path.join(IMAGE_FRAMES_DIR, filename)
        text = pytesseract.image_to_string(Image.open(image_path)).strip()
        if text:
            extracted_texts.append(text)

    update_json_file({"extracted_text": "\n".join(extracted_texts)})

# 4️⃣ Extract & Transcribe Audio
def extract_audio_from_video(video_path, audio_path):
    """Extract audio and save as MP3."""
    if not os.path.exists(video_path):
        return False

    video_clip = VideoFileClip(video_path)
    if video_clip.audio is None:
        update_json_file({"transcription": ""})
        return False

    video_clip.audio.write_audiofile(audio_path, codec="mp3", verbose=False)
    return True

def transcribe_audio_with_whisper(audio_path):
    """Transcribe audio using Whisper AI."""
    if not os.path.exists(audio_path):
        return

    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path, language="en", verbose=False)
    update_json_file({"transcription": result["text"]})

# 5️⃣ Update JSON File
def update_json_file(new_data):
    """Update JSON file with extracted text or transcription."""
    allowed_keys = {"transcription", "extracted_text"}

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data.update({k: v for k, v in new_data.items() if k in allowed_keys})

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

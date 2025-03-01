import os
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


def determine_frame_skip(video_path):
    """Determine optimal frame skipping interval based on video length."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return 100  # Default

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000  # Convert to seconds

    cap.release()

    if duration < 10:  # Short video (<10s) → Skip fewer frames
        return max(1, frame_count // 10)
    elif duration < 60:  # Medium video (<60s) → Moderate skipping
        return max(10, frame_count // 50)
    else:  # Long video (>60s) → Higher frame skipping
        return max(50, frame_count // 200)


def extract_frames_from_video(video_path):
    """Extract frames from a video at an optimal interval."""
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return False

    frame_skip = determine_frame_skip(video_path)

    os.makedirs(IMAGE_FRAMES_DIR, exist_ok=True)

    # Clear previous frames
    for file in os.listdir(IMAGE_FRAMES_DIR):
        os.remove(os.path.join(IMAGE_FRAMES_DIR, file))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open {video_path}.")
        return False

    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if index % frame_skip == 0:
            frame_path = os.path.join(IMAGE_FRAMES_DIR, f"frame{index}.png")
            cv2.imwrite(frame_path, frame)
            print(f"Extracted: {frame_path}")

        index += 1

    cap.release()
    print("Frame extraction complete.")
    return True


def extract_text_from_frames():
    """Extract text from frames using Tesseract OCR and store in JSON."""
    extracted_texts = []

    for filename in sorted(os.listdir(IMAGE_FRAMES_DIR)):
        image_path = os.path.join(IMAGE_FRAMES_DIR, filename)
        text = pytesseract.image_to_string(Image.open(image_path)).strip()
        if text:
            extracted_texts.append(text)

    text_output = "\n".join(extracted_texts)

    # Save extracted text to JSON
    update_json_file({"extracted_text": text_output})

    print(f"Extracted text saved to {JSON_FILE}")


def extract_audio_from_video(video_path, audio_path):
    """Extract audio from a video and save it as an MP3 file."""
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return False

    video_clip = VideoFileClip(video_path)

    if video_clip.audio is None:
        print("No audio detected in video.")
        update_json_file({"transcription": ""})  # Save empty transcription
        video_clip.close()  # Close video to avoid subprocess issues
        return False

    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, codec="mp3", verbose=False)

    # Proper cleanup
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

    # Save transcription to JSON
    update_json_file({"transcription": result["text"]})

    print("Transcription saved to JSON.")


def update_json_file(new_data):
    """Update JSON file while ensuring only 'transcription' and 'extracted_text' exist."""
    allowed_keys = {"transcription", "extracted_text"}

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Only keep allowed keys
    data.update({k: v for k, v in new_data.items() if k in allowed_keys})

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Updated JSON file.")


if __name__ == "__main__":
    if extract_frames_from_video(VIDEO_FILE):
        extract_text_from_frames()

    if extract_audio_from_video(VIDEO_FILE, AUDIO_FILE):
        transcribe_audio_with_whisper(AUDIO_FILE)

    print("Processing completed!")

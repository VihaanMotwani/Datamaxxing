import yt_dlp
import os

def download_instagram_reel(url, output_filename="video.mp4"):
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
        info = ydl.extract_info(url, download=True)
        print(f"Downloaded: {output_filename}")

# Example usage
reel_url = input("Enter Instagram Reel URL: ")
download_instagram_reel(reel_url)

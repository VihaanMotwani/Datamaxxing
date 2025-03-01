import yt_dlp

def download_instagram_reel(url, output_folder="downloads"):
    """Download Instagram Reel as an .mp4 file."""
    options = {
        'outtmpl': f'{output_folder}/%(title)s.%(ext)s',
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        print(f"Downloaded: {info['title']}.mp4")

# Example usage
reel_url = input("Enter Instagram Reel URL: ")
download_instagram_reel(reel_url)
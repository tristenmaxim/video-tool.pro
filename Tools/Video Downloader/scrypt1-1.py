#YES TIKTOK!!!
#python "/Users/maximtristen/Desktop/Tools/Video Downloader/scrypt1-1.py" https://youtu.be/oK2q7SEbgms

import re
import os
import argparse
import subprocess
from tqdm import tqdm
import requests
import sys
import json

# For TikTok videos
try:
    from TikTokApi import TikTokApi
    TIKTOK_API_AVAILABLE = True
except ImportError:
    TIKTOK_API_AVAILABLE = False


def is_youtube_url(url):
    """Check if the URL is a valid YouTube URL."""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|shorts/|.+\?v=)?([^&=%\?]{11})'
    )
    youtube_pattern = re.compile(youtube_regex)
    return bool(youtube_pattern.match(url))


def is_tiktok_url(url):
    """Check if the URL is a valid TikTok URL."""
    # Updated regex to include vt.tiktok.com and other short URL formats
    tiktok_regex = r'(https?://)?(www\.|vm\.|vt\.)?tiktok\.com/(@[\w.-]+/video/\d+|@[\w.-]+|\w+)'
    tiktok_pattern = re.compile(tiktok_regex)
    
    # Direct check for common TikTok URL patterns
    is_tiktok = bool(tiktok_pattern.match(url))
    
    # Additional check for shortened URLs that might not match the pattern
    if not is_tiktok and ('tiktok.com' in url or 'vt.tiktok' in url):
        return True
        
    return is_tiktok


def check_yt_dlp_installed():
    """Check if yt-dlp is installed and install it if not."""
    try:
        # Check if yt-dlp is installed
        subprocess.run(["yt-dlp", "--version"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("yt-dlp is not installed. Attempting to install it...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
                          check=True)
            print("yt-dlp has been successfully installed.")
            return True
        except subprocess.SubprocessError:
            print("Failed to install yt-dlp. Please install it manually with: pip install yt-dlp")
            return False


def download_with_yt_dlp(url, output_path="./downloads"):
    """Download a video using yt-dlp."""
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Get video info first
        print(f"Getting information for: {url}")
        cmd_info = ["yt-dlp", "--dump-json", "--no-playlist", url]
        
        try:
            result = subprocess.run(cmd_info, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)
            
            print(f"Downloading: {video_info.get('title', 'Unknown title')}")
            print(f"Channel: {video_info.get('channel', 'Unknown channel')}")
            print(f"Duration: {video_info.get('duration_string', 'Unknown')}")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Could not retrieve video info: {e}")
            print("Continuing with download anyway...")
            video_info = None
        
        # Format the output template
        if is_youtube_url(url):
            output_template = os.path.join(output_path, "%(title)s-%(id)s.%(ext)s")
        else:
            output_template = os.path.join(output_path, "%(uploader)s-%(title)s-%(id)s.%(ext)s")
        
        # Set up the yt-dlp command
        cmd = [
            "yt-dlp",
            "-f", "best",  # Download best quality
            "--no-playlist",  # Don't download playlists
            "--progress",  # Show progress
            "-o", output_template,  # Output file template
            url
        ]
        
        # Run the command
        print("Starting download...")
        subprocess.run(cmd, check=True)
        
        print(f"\nVideo downloaded successfully to the directory: {output_path}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video with yt-dlp: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def download_tiktok_video_api(url, output_path="./downloads"):
    """Download a TikTok video using TikTokApi."""
    if not TIKTOK_API_AVAILABLE:
        print("TikTokApi is not installed. Falling back to yt-dlp.")
        return download_with_yt_dlp(url, output_path)
    
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Initialize TikTokApi
        api = TikTokApi()
        
        # Extract video ID from URL
        video_id = extract_tiktok_id(url)
        
        if not video_id:
            print("Could not extract TikTok video ID from URL. Falling back to yt-dlp.")
            return download_with_yt_dlp(url, output_path)
        
        # Get video info
        video_info = api.video(id=video_id)
        
        # Get video download URL
        video_url = video_info['itemInfo']['itemStruct']['video']['downloadAddr']
        
        # Get video details
        author = video_info['itemInfo']['itemStruct']['author']['uniqueId']
        desc = video_info['itemInfo']['itemStruct']['desc']
        
        print(f"Downloading TikTok video from: {author}")
        print(f"Description: {desc[:50]}..." if len(desc) > 50 else f"Description: {desc}")
        
        # Download the video using requests
        filename = f"{author}_{video_id}.mp4"
        filepath = os.path.join(output_path, filename)
        
        # Download with progress bar
        response = requests.get(video_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        print(f"\nVideo downloaded successfully to: {filepath}")
        return True
    
    except Exception as e:
        print(f"Error downloading TikTok video with API: {e}")
        print("Falling back to yt-dlp...")
        return download_with_yt_dlp(url, output_path)


def extract_tiktok_id(url):
    """Extract TikTok video ID from URL."""
    # Try to find the video ID using regex
    patterns = [
        r'tiktok\.com/.*?/video/(\d+)',
        r'tiktok\.com/v/(\d+)',
        r'vm\.tiktok\.com/(\w+)',
        r'vt\.tiktok\.com/(\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # If ID can't be extracted from URL, make a request to follow redirects
    try:
        print("Following URL redirects to find TikTok video ID...")
        response = requests.head(url, allow_redirects=True)
        final_url = response.url
        print(f"Redirected to: {final_url}")
        
        for pattern in patterns:
            match = re.search(pattern, final_url)
            if match:
                return match.group(1)
        
        # If we still couldn't find it, look for any digits that might be the ID
        digit_match = re.search(r'/(\d{18,19})', final_url)
        if digit_match:
            return digit_match.group(1)
    except Exception as e:
        print(f"Error following redirects: {e}")
    
    return None


def download_video(url, output_path="./downloads"):
    """Download a video from either YouTube or TikTok based on the URL."""
    # Check if yt-dlp is installed
    if not check_yt_dlp_installed():
        print("Cannot proceed without yt-dlp. Please install it manually.")
        return False
    
    if is_youtube_url(url):
        print("Detected YouTube URL.")
        return download_with_yt_dlp(url, output_path)
    elif is_tiktok_url(url):
        print("Detected TikTok URL.")
        if TIKTOK_API_AVAILABLE:
            return download_tiktok_video_api(url, output_path)
        else:
            print("TikTokApi not available. Using yt-dlp for TikTok download.")
            return download_with_yt_dlp(url, output_path)
    else:
        # If URL contains "tiktok" at all, try downloading it as TikTok
        if "tiktok" in url.lower():
            print("URL appears to be TikTok but not in standard format. Attempting download anyway.")
            return download_with_yt_dlp(url, output_path)
        
        print("Unsupported URL. Please provide a valid YouTube or TikTok URL.")
        print("If you believe this URL should work, you can force yt-dlp to try downloading it anyway.")
        response = input("Would you like to try downloading with yt-dlp anyway? (y/n): ").lower()
        if response.startswith('y'):
            return download_with_yt_dlp(url, output_path)
        return False


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download videos from YouTube or TikTok using yt-dlp.')
    parser.add_argument('url', help='URL of the video to download')
    parser.add_argument('--output', '-o', default='./downloads', 
                        help='Output directory for downloaded videos')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force download even if URL is not recognized')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Force download if requested
    if args.force:
        print("Forced download mode. Attempting to download URL directly with yt-dlp.")
        download_with_yt_dlp(args.url, args.output)
    else:
        # Download the video
        download_video(args.url, args.output)


if __name__ == "__main__":
    main()
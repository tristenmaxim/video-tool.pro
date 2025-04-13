import cv2
import numpy as np
import os
import subprocess
import tempfile
import time

def add_noise(frame, intensity=5):
    """Add slight noise to the frame"""
    noise = np.random.randint(-intensity, intensity, frame.shape, dtype='int16')
    frame = np.clip(frame.astype('int16') + noise, 0, 255).astype('uint8')
    return frame

def process_video(input_path, output_path, sample_rate=10):
    """Modify video to make it unique while keeping it visually unchanged
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        sample_rate: Only process 1 in every N frames for speed (higher = faster)
    """
    print(f"Processing {input_path}...")
    start_time = time.time()
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return
        
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        
        # Open the video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{input_path}'. Check if the file is valid.")
            return
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer with h264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264') if cv2.__version__ >= '4.0.0' else cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process some frames for better performance
            if frame_count % sample_rate == 0:
                frame = add_noise(frame, intensity=3)  # Lower intensity for subtlety
                
            out.write(frame)
            frame_count += 1
            
            # Show progress
            if frame_count % 100 == 0:
                print(f"Processing: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
        
        cap.release()
        out.release()
        
        print("Applying final encoding with audio...")
        
        # Use FFmpeg directly to copy audio from original to the processed video
        try:
            cmd = [
                'ffmpeg',
                '-i', temp_video,  # Processed video (no audio)
                '-i', input_path,  # Original video (with audio)
                '-c:v', 'copy',    # Copy video stream without re-encoding
                '-c:a', 'aac',     # Use AAC codec for audio
                '-map', '0:v:0',   # Use video from first input
                '-map', '1:a:0',   # Use audio from second input
                '-shortest',       # Finish encoding when shortest stream ends
                output_path,
                '-y'               # Overwrite output file if it exists
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f'New unique video saved as {output_path}')
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            
        except FileNotFoundError:
            print("Error: FFmpeg executable not found. Please install FFmpeg on your system.")
            print("For macOS: brew install ffmpeg")
            print("For Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("For Windows: Download from https://ffmpeg.org/download.html")
        except Exception as e:
            print(f"Error processing video: {e}")

# Example usage with your specific path
input_path = '/Users/maximtristen/Desktop/Unicalization Tool/Creatives/Geo - EN/21.03.2025/5.mp4'
output_path = '/Users/maximtristen/Desktop/Unicalization Tool/Creatives/Geo - EN/21.03.2025/5_unique.mp4'
process_video(input_path, output_path, sample_rate=5)  # Process 1 in every 5 frames for speed

# Uncomment and modify this line to use absolute paths:
# process_video('/full/path/to/input.mp4', '/full/path/to/output_unique.mp4')

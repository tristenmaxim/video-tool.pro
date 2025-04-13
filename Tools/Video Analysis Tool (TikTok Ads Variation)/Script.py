import os
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import tempfile
from pathlib import Path
import subprocess
import json
import time
import re
from openai import OpenAI
from datetime import datetime
import hashlib
from collections import Counter

print(">>> Debug: Starting TikTok Ad Analyzer...")

class TikTokAdAnalyzer:
    """An advanced tool for analyzing TikTok ads using OpenAI's vision models."""
    
    def __init__(self, api_key=None, verbose=False):
        """
        Initialize the TikTok Ad Analyzer.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, uses OPENAI_API_KEY env variable.
            verbose (bool): Whether to print detailed logs.
        """
        try:
            self.client = OpenAI(api_key=api_key)
            print(">>> Debug: Successfully initialized OpenAI client")
        except Exception as e:
            print(f">>> Debug: Error initializing OpenAI client: {str(e)}")
            raise
            
        self.verbose = verbose
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # List of models to try
        self.vision_models = ["gpt-4o", "gpt-4-vision-preview"]
        
        # Track processing metrics
        self.metrics = {
            "frames_extracted": 0,
            "text_regions_detected": 0,
            "ui_elements_detected": 0,
            "processing_time": 0
        }
        
        if self.verbose:
            print(f"Created temporary directory at {self.temp_dir}")
    
    def log(self, message):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] {message}")
    
    def extract_tiktok_frames(self, video_path, frame_interval=0.5, max_frames=30):
        """
        Extract frames from TikTok video at regular time intervals.
        
        Args:
            video_path (str): Path to the video file.
            frame_interval (float): Time interval between frames in seconds.
            max_frames (int): Maximum number of frames to extract.
            
        Returns:
            list: List of extracted frame paths.
        """
        self.log(f"Extracting frames from {video_path} at {frame_interval}s intervals")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            self.log(f"Error: Video file not found at {video_path}")
            return []
            
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log(f"Error: Could not open video file {video_path}")
                return []
                
            # Get video metadata
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            self.log(f"TikTok video has {frame_count} frames, {fps:.2f} fps, duration: {duration:.2f} seconds")
            
            # Calculate frames to extract based on interval
            frames = []
            frame_positions = []
            
            # Calculate frame positions at specified intervals
            current_time = 0
            while current_time < duration and len(frame_positions) < max_frames:
                frame_pos = int(current_time * fps)
                frame_positions.append(frame_pos)
                current_time += frame_interval
            
            # Extract frames at calculated positions
            for i, frame_pos in enumerate(frame_positions):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = cap.read()
                    if ret:
                        frame_path = self._save_frame(frame, i)
                        frames.append(frame_path)
                    else:
                        self.log(f"Failed to extract frame at position {frame_pos}")
                except Exception as e:
                    self.log(f"Error processing frame {i}: {str(e)}")
            
            cap.release()
            self.metrics["frames_extracted"] = len(frames)
            
            self.log(f"Successfully extracted {len(frames)} frames at {frame_interval}s intervals")
            return frames
            
        except Exception as e:
            self.log(f"Error extracting frames: {str(e)}")
            if 'cap' in locals():
                cap.release()
            return []
    
    def _save_frame(self, frame, index):
        """Save a frame to a temporary file."""
        try:
            frame_path = self.temp_dir / f"{index:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            return frame_path
        except Exception as e:
            self.log(f"Error saving frame {index}: {str(e)}")
            # Create a blank frame as fallback
            blank_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)  # TikTok portrait orientation
            frame_path = self.temp_dir / f"{index:04d}.jpg"
            cv2.imwrite(str(frame_path), blank_frame)
            return frame_path
    
    def extract_audio(self, video_path):
        """
        Extract audio from video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: Path to the extracted audio file.
        """
        self.log(f"Extracting audio from {video_path}")
        audio_path = self.temp_dir / "audio.mp3"
        
        # Check if video file exists
        if not os.path.exists(video_path):
            self.log(f"Error: Video file not found at {video_path}")
            return None
        
        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg", "-i", video_path,
            "-q:a", "0", "-map", "a", str(audio_path),
            "-y"  # Overwrite output file if it exists
        ]
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.log("Audio extraction successful")
            return audio_path if audio_path.exists() else None
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            self.log(f"Error extracting audio: {stderr}")
            
            # Check if the video has no audio stream
            if "Stream specifier 'a' in filtergraph" in stderr or "Invalid data found" in stderr:
                self.log("This video may not have an audio stream")
            return None
        except Exception as e:
            self.log(f"Unexpected error extracting audio: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using OpenAI's Whisper model.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            dict: The transcription result.
        """
        if not audio_path or not Path(audio_path).exists():
            self.log("No audio file found for transcription")
            return {"text": ""}
            
        self.log(f"Transcribing audio from {audio_path}")
        
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            self.log("Transcription successful")
            return transcription
        except Exception as e:
            self.log(f"Error transcribing audio: {str(e)}")
            return {"text": ""}
    
    def create_grid_image(self, frame_paths, grid_size=(5, 6), add_timestamps=True):
        """
        Create a grid image from multiple frames, optimized for TikTok portrait videos.
        
        Args:
            frame_paths (list): List of paths to frames.
            grid_size (tuple): Grid dimensions (rows, cols).
            add_timestamps (bool): Whether to add timestamps to frames.
            
        Returns:
            PIL.Image: The grid image.
        """
        self.log(f"Creating grid image with {len(frame_paths)} frames in a {grid_size} grid")
        
        if not frame_paths:
            self.log("Error: No frames available to create grid")
            # Create a small blank image as fallback (TikTok portrait orientation)
            return Image.new('RGB', (1080, 1920), color='black')
            
        # Resize the number of frames to match the grid size
        max_frames = grid_size[0] * grid_size[1]
        frame_paths = frame_paths[:max_frames]
        
        try:
            # Load the first image to get dimensions
            sample_img = Image.open(frame_paths[0])
            orig_width, orig_height = sample_img.size
            
            # Calculate thumbnail size to fit in grid
            rows, cols = grid_size
            thumb_width = 1080 // cols
            thumb_height = 1920 // rows
            
            # Maintain aspect ratio for thumbnails
            if orig_width / orig_height > thumb_width / thumb_height:
                # Width constrained
                new_width = thumb_width
                new_height = int(orig_height * (thumb_width / orig_width))
            else:
                # Height constrained
                new_height = thumb_height
                new_width = int(orig_width * (thumb_height / orig_height))
            
            # Calculate grid size
            grid_width = thumb_width * cols
            grid_height = thumb_height * rows
            
            # Create a new blank image
            grid_img = Image.new('RGB', (grid_width, grid_height), color=(20, 20, 20))
            
            # Try to load a font for timestamps
            try:
                font = ImageFont.truetype("Arial", 12)
            except IOError:
                font = ImageFont.load_default()
            
            # Place images in the grid
            for idx, frame_path in enumerate(frame_paths):
                try:
                    # Calculate position in grid
                    row = idx // cols
                    col = idx % cols
                    x = col * thumb_width + (thumb_width - new_width) // 2
                    y = row * thumb_height + (thumb_height - new_height) // 2
                    
                    # Open, resize and paste image
                    img = Image.open(frame_path)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    grid_img.paste(img, (x, y))
                    
                    # Add timestamp or frame number
                    if add_timestamps:
                        draw = ImageDraw.Draw(grid_img)
                        time_sec = idx * 0.5  # Assuming 0.5s intervals
                        mins = int(time_sec // 60)
                        secs = time_sec % 60
                        text = f"{mins:02d}:{secs:05.2f}"
                        draw.rectangle([x, y, x+new_width, y+20], fill=(0, 0, 0, 180))
                        draw.text((x+5, y+3), text, fill="white", font=font)
                except Exception as e:
                    self.log(f"Error processing frame {idx}: {str(e)}")
                    # Just continue with other frames
            
            return grid_img
        except Exception as e:
            self.log(f"Error creating grid image: {str(e)}")
            # Create a small blank image as fallback
            return Image.new('RGB', (1080, 1920), color='black')
    
    def detect_text_in_frames(self, frame_paths):
        """
        Detect and analyze text content in TikTok ad frames.
        
        Args:
            frame_paths (list): List of paths to frames.
            
        Returns:
            dict: Analysis of text content.
        """
        self.log("Analyzing text content in frames")
        
        text_analysis = {
            "frames_with_text": 0,
            "total_text_regions": 0,
            "text_density_over_time": [],
            "text_regions": {}
        }
        
        if not frame_paths:
            return text_analysis
        
        try:
            for i, frame_path in enumerate(frame_paths):
                # Read image
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect text regions
                regions = self._detect_text_regions(gray)
                
                # Store results
                time_point = i * 0.5  # Assuming 0.5s intervals
                text_analysis["text_regions"][f"frame_{i}"] = len(regions)
                text_analysis["text_density_over_time"].append({
                    "time": time_point,
                    "regions": len(regions)
                })
                
                if len(regions) > 0:
                    text_analysis["frames_with_text"] += 1
                    text_analysis["total_text_regions"] += len(regions)
            
            return text_analysis
            
        except Exception as e:
            self.log(f"Error in text analysis: {str(e)}")
            return text_analysis
    
    def _detect_text_regions(self, gray_img):
        """
        Detect regions that likely contain text.
        
        Args:
            gray_img: Grayscale OpenCV image.
            
        Returns:
            list: List of (x, y, width, height) tuples for detected text regions.
        """
        regions = []
        
        try:
            # Apply thresholding
            _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for potential text regions
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by typical text dimensions and aspect ratio
                if 10 < w < gray_img.shape[1]//2 and 5 < h < gray_img.shape[0]//15:
                    aspect = w / float(h)
                    if 1.5 < aspect < 20:  # Text is usually wider than tall
                        regions.append((x, y, w, h))
            
            # Group nearby regions that might form lines of text
            grouped_regions = []
            for region in regions:
                x, y, w, h = region
                
                # Check if this region should be merged with any existing group
                merged = False
                for i, (gx, gy, gw, gh) in enumerate(grouped_regions):
                    # If vertically aligned and close horizontally
                    if abs(y - gy) < h and (x <= gx + gw + 10):
                        # Merge by extending the group
                        new_x = min(x, gx)
                        new_y = min(y, gy)
                        new_w = max(x + w, gx + gw) - new_x
                        new_h = max(y + h, gy + gh) - new_y
                        grouped_regions[i] = (new_x, new_y, new_w, new_h)
                        merged = True
                        break
                
                if not merged:
                    grouped_regions.append(region)
            
            self.metrics["text_regions_detected"] += len(grouped_regions)
            return grouped_regions
            
        except Exception as e:
            self.log(f"Error detecting text regions: {str(e)}")
            return []
    
    def encode_image(self, image_path=None, pil_image=None):
        """
        Encode an image to base64.
        
        Args:
            image_path (str, optional): Path to the image file.
            pil_image (PIL.Image, optional): PIL Image object.
            
        Returns:
            str: Base64 encoded image.
        """
        try:
            if pil_image:
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            elif image_path:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                return None
        except Exception as e:
            self.log(f"Error encoding image: {str(e)}")
            return None
    
    def analyze_with_vision(self, grid_image, transcription=""):
        """
        Analyze TikTok ad using OpenAI's vision model.
        
        Args:
            grid_image (PIL.Image): Grid image of video frames.
            transcription (str): Transcription of audio.
            
        Returns:
            dict: The OpenAI response or error information.
        """
        if grid_image is None:
            self.log("Error: No grid image available for analysis")
            return {"error": "No grid image available"}
            
        encoded_grid = self.encode_image(pil_image=grid_image)
        if not encoded_grid:
            self.log("Error: Failed to encode grid image")
            return {"error": "Failed to encode grid image"}
        
        # Craft a TikTok ad analysis prompt
        prompt = (
            "This is a frame-by-frame sequence of a TikTok advertisement captured at 0.5 second intervals. "
            "Please analyze this advertisement and provide the following insights:\n\n"
            "1) What product, service, or app is being promoted?\n"
            "2) What is the structure and flow of the ad? How does it engage viewers?\n"
            "3) What key messages, text overlays, or calls-to-action appear throughout the ad?\n"
            "4) What visual techniques are used (transitions, effects, text animations)?\n"
            "5) Who appears to be the target audience?\n"
            "6) What marketing strategies or psychological techniques are employed?\n\n"
            "Please provide a comprehensive analysis that would help me understand the effectiveness "
            "of this TikTok ad and the techniques used."
        )
        
        if transcription:
            prompt += f"\n\nHere is the transcribed audio from the ad which may provide additional context: \"{transcription}\""
        
        model = "gpt-4o"
        self.log(f"Sending TikTok ad analysis request to OpenAI API using {model}...")
        
        try:
            # Prepare messages for the API
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a TikTok marketing specialist and ad analyst with expertise in social media "
                        "advertising patterns, visual marketing techniques, and user engagement strategies. "
                        "Provide detailed, insightful analysis of TikTok advertisements with a focus on structure, "
                        "messaging, visual techniques, and marketing strategies."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_grid}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1500
            )
            
            self.log(f"Received response from OpenAI API using {model}")
            return response
                
        except Exception as e:
            self.log(f"Error analyzing with {model}: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_tiktok_ad(self, video_path, frame_interval=0.5, grid_size=(5, 6), transcribe=True, save_grid=True):
        """
        Complete pipeline to analyze a TikTok advertisement.
        
        Args:
            video_path (str): Path to the video file.
            frame_interval (float): Time interval between frames in seconds.
            grid_size (tuple): Grid dimensions for frame layout (rows, cols).
            transcribe (bool): Whether to transcribe audio.
            save_grid (bool): Whether to save the grid image.
            
        Returns:
            dict: Results of the analysis.
        """
        results = {
            "video_path": video_path,
            "frame_count": 0,
            "transcription": "",
            "analysis": "",
            "grid_image_path": None,
            "text_analysis": {},
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Verify the video file exists
            if not os.path.exists(video_path):
                self.log(f"Error: Video file not found at {video_path}")
                results["error"] = f"Video file not found at {video_path}"
                return results
            
            # 1. Extract frames at regular intervals
            self.log(f"Starting frame extraction from {video_path}")
            max_frames = grid_size[0] * grid_size[1]
            frame_paths = self.extract_tiktok_frames(
                video_path, 
                frame_interval=frame_interval,
                max_frames=max_frames
            )
            
            if not frame_paths:
                self.log("Error: Failed to extract frames from video")
                results["error"] = "Failed to extract frames from video"
                results["processing_time"] = time.time() - start_time
                return results
                
            results["frame_count"] = len(frame_paths)
            self.log(f"Successfully extracted {len(frame_paths)} frames")
            
            # 2. Transcribe audio if requested
            transcription = ""
            if transcribe:
                self.log("Starting audio extraction and transcription")
                audio_path = self.extract_audio(video_path)
                if audio_path:
                    transcription_result = self.transcribe_audio(audio_path)
                    if hasattr(transcription_result, 'text'):
                        transcription = transcription_result.text
                    elif isinstance(transcription_result, dict) and 'text' in transcription_result:
                        transcription = transcription_result['text']
                    
                    if transcription:
                        self.log(f"Transcription successful: {transcription[:50]}...")
                    else:
                        self.log("Transcription was empty")
                else:
                    self.log("No audio extracted or video has no audio track")
                    
                results["transcription"] = transcription
            
            # 3. Detect text in frames
            text_analysis = self.detect_text_in_frames(frame_paths)
            results["text_analysis"] = text_analysis
            self.log(f"Detected text in {text_analysis['frames_with_text']} frames with {text_analysis['total_text_regions']} text regions")
            
            # 4. Create grid image
            self.log("Creating grid image from extracted frames")
            grid_image = self.create_grid_image(frame_paths, grid_size=grid_size)
            
            if grid_image:
                try:
                    # Get video filename for naming the grid image
                    video_name = os.path.basename(video_path)
                    video_name_no_ext = os.path.splitext(video_name)[0]
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # Save in temp dir for processing
                    temp_grid_path = self.temp_dir / "tiktok_grid.jpg"
                    grid_image.save(temp_grid_path)
                    
                    # If save_grid is true, also save to current directory for keeping
                    if save_grid:
                        permanent_grid_path = Path(f"{video_name_no_ext}_grid_{timestamp}.jpg")
                        grid_image.save(permanent_grid_path)
                        self.log(f"Permanent grid image saved to {permanent_grid_path}")
                        results["permanent_grid_path"] = str(permanent_grid_path)
                    
                    results["grid_image_path"] = str(temp_grid_path)
                    self.log(f"Grid image saved to {temp_grid_path}")
                except Exception as e:
                    self.log(f"Error saving grid image: {str(e)}")
            else:
                self.log("Warning: Failed to create grid image")
            
            # 5. Analyze with vision model
            self.log("Starting TikTok ad analysis")
            vision_response = self.analyze_with_vision(
                grid_image,
                transcription
            )
            
            if isinstance(vision_response, dict) and "error" in vision_response:
                self.log(f"Error in vision analysis: {vision_response['error']}")
                results["error"] = vision_response["error"]
            else:
                try:
                    if hasattr(vision_response, 'choices') and vision_response.choices:
                        analysis_text = vision_response.choices[0].message.content
                    elif isinstance(vision_response, dict) and 'choices' in vision_response:
                        analysis_text = vision_response['choices'][0]['message']['content']
                    else:
                        self.log(f"Warning: Unexpected vision response format: {type(vision_response)}")
                        analysis_text = ""
                        
                    results["analysis"] = analysis_text
                    self.log("Analysis completed successfully")
                except Exception as e:
                    self.log(f"Error extracting analysis text: {str(e)}")
                    results["error"] = f"Error extracting analysis text: {str(e)}"
            
            # Calculate processing time
            results["processing_time"] = time.time() - start_time
            self.log(f"TikTok ad analysis completed in {results['processing_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.log(f"Unhandled exception in analyze_tiktok_ad: {str(e)}")
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
            return results
    
    def cleanup(self):
        """Remove temporary files."""
        self.log("Cleaning up temporary files")
        try:
            for file_path in self.temp_dir.glob("*"):
                try:
                    file_path.unlink()
                except Exception as e:
                    self.log(f"Error removing file {file_path}: {str(e)}")
            
            try:
                self.temp_dir.rmdir()
                self.log(f"Removed temporary directory {self.temp_dir}")
            except Exception as e:
                self.log(f"Error removing directory {self.temp_dir}: {str(e)}")
        except Exception as e:
            self.log(f"Error during cleanup: {str(e)}")


def save_results_to_txt(results, output_path=None):
    """
    Save analysis results to a text file.
    
    Args:
        results (dict): Analysis results dictionary.
        output_path (str, optional): Path to save the results. If None, uses the video filename.
        
    Returns:
        str: Path to the saved text file.
    """
    if output_path is None:
        # Create a default filename based on the video path
        video_name = os.path.basename(results['video_path'])
        video_name_no_ext = os.path.splitext(video_name)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{video_name_no_ext}_analysis_{timestamp}.txt"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("===== TikTok Ad Analysis Results =====\n\n")
            
            # Write basic info
            f.write(f"Video: {results['video_path']}\n")
            f.write(f"Processed {results['frame_count']} frames in {results['processing_time']:.2f} seconds\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write transcription if available
            if results.get("transcription"):
                f.write("----- Transcription -----\n")
                f.write(results["transcription"])
                f.write("\n\n")
            
            # Write text analysis
            f.write("----- Text Analysis -----\n")
            f.write(f"Frames with text: {results['text_analysis'].get('frames_with_text', 0)}\n")
            f.write(f"Total text regions: {results['text_analysis'].get('total_text_regions', 0)}\n")
            
            # Write text density timeline if available
            if results['text_analysis'].get('text_density_over_time'):
                f.write("\nText density timeline:\n")
                for point in results['text_analysis']['text_density_over_time']:
                    f.write(f"Time {point['time']:.1f}s: {point['regions']} text regions\n")
            f.write("\n")
            
            # Write main analysis
            f.write("----- Ad Analysis -----\n")
            f.write(results.get("analysis", "No analysis available"))
            f.write("\n\n")
            
            # Write grid image path
            if results.get("grid_image_path"):
                f.write(f"Grid image saved at: {results['grid_image_path']}\n")
            
            # Write any errors
            if "error" in results:
                f.write(f"\nERROR: {results['error']}\n")
                
        print(f"Results saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving results to file: {str(e)}")
        return None

def main():
    # Replace these values with your actual video path and API key
    video_path = "/Users/maximtristen/Desktop/downloads/🔥 REPEAT AFTER ME ｜ Anyone Can Make Money Online With Right Pocket Option Strategy-oK2q7SEbgms.mp4"
    api_key = "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"    
    output_txt_path = None  # Set to None for auto-naming or specify a path

    print(f"Starting TikTok ad analysis for: {video_path}")
    analyzer = TikTokAdAnalyzer(api_key=api_key, verbose=True)

    # Analyze the TikTok ad
    results = analyzer.analyze_tiktok_ad(
        video_path,
        frame_interval=0.5,  # Extract frames every 0.5 seconds
        grid_size=(5, 6),    # 5×6 grid for shorter TikTok videos
        transcribe=True
    )

    # Print results to console
    print("\n===== TikTok Ad Analysis Results =====")
    print(f"Video: {results['video_path']}")
    print(f"Processed {results['frame_count']} frames in {results['processing_time']:.2f} seconds")
            
    if results.get("transcription"):
        print("\n----- Transcription -----")
        print(results["transcription"])
            
    print("\n----- Text Analysis -----")
    print(f"Frames with text: {results['text_analysis'].get('frames_with_text', 0)}")
    print(f"Total text regions: {results['text_analysis'].get('total_text_regions', 0)}")
            
    print("\n----- Ad Analysis -----")
    print(results.get("analysis", "No analysis available"))
            
    if "error" in results:
        print(f"\nError: {results['error']}")
    
    # Save results to text file
    txt_path = save_results_to_txt(results, output_txt_path)
    if txt_path:
        print(f"Full analysis results saved to: {txt_path}")

    # Cleanup temp files
    analyzer.cleanup()

if __name__ == "__main__":
    main()
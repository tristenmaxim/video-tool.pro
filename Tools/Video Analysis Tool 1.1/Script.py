# Only Transcribing!!!
#!/usr/bin/env python3
# Enhanced Lecture Transcriber with Audio Chunking Support
#python "/Users/maximtristen/Desktop/Video Analysis Tool 1.1/Script.py" '/Users/maximtristen/Desktop/downloads/Разбор резюмешек｜Кейс №1-P3yjSdyFIAc.mp4' --openai-key YOUR_OPENAI_API_KEY --verbose

import os
import numpy as np
import tempfile
from pathlib import Path
import json
import time
import re
from openai import OpenAI
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
import subprocess
from pyannote.audio import Pipeline
from pyannote.core import Segment

# Download necessary NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EnhancedLectureTranscriber:
    """Advanced tool for transcribing lectures with speaker diarization and punctuation."""
    
    def __init__(self, api_key=None, hf_token=None, verbose=False):
        """
        Initialize the Enhanced Lecture Transcriber.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, uses OPENAI_API_KEY env variable.
            hf_token (str, optional): HuggingFace API token for speaker diarization.
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
        self.hf_token = hf_token
        
        # Initialize speaker diarization if token is provided
        self.diarization_pipeline = None
        if self.hf_token:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=self.hf_token
                )
                print(">>> Debug: Successfully initialized speaker diarization pipeline")
            except Exception as e:
                print(f">>> Debug: Error initializing speaker diarization: {str(e)}")
                print(">>> Speaker diarization will be disabled")
        
        # Track processing metrics
        self.metrics = {
            "audio_duration": 0,
            "processing_time": 0,
            "speakers_detected": 0,
            "chunks_processed": 0
        }
        
        # Set Whisper API size limit in bytes (25MB)
        self.max_file_size = 25 * 1024 * 1024
        
        if self.verbose:
            print(f"Created temporary directory at {self.temp_dir}")
    
    def log(self, message):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] {message}")
    
    def extract_audio(self, video_path):
        """
        Extract audio from video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: Path to the extracted audio file.
        """
        self.log(f"Extracting audio from {video_path}")
        audio_path = self.temp_dir / "audio.wav"
        
        # Check if video file exists
        if not os.path.exists(video_path):
            self.log(f"Error: Video file not found at {video_path}")
            return None
        
        # Use ffmpeg to extract audio with high quality for better transcription
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"  # Mono 16kHz WAV, overwrite if exists
        ]
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.log("Audio extraction successful")
            
            # Get audio duration
            duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                           "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)]
            duration_result = subprocess.run(duration_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            duration = float(duration_result.stdout.decode().strip())
            self.metrics["audio_duration"] = duration
            self.log(f"Audio duration: {duration:.2f} seconds")
            
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
    
    def check_file_size(self, file_path):
        """
        Check if the file size exceeds OpenAI's API limit.
        
        Args:
            file_path (Path or str): Path to the file to check.
            
        Returns:
            bool: True if file size is below the limit, False otherwise.
        """
        file_size = os.path.getsize(file_path)
        self.log(f"File size: {file_size / (1024 * 1024):.2f} MB")
        return file_size <= self.max_file_size
    
    def split_audio(self, audio_path, chunk_duration=600):
        """
        Split audio file into smaller chunks that meet API size limits.
        
        Args:
            audio_path (Path or str): Path to the audio file.
            chunk_duration (int): Maximum duration of each chunk in seconds.
                                 Default is 10 minutes (600 seconds).
            
        Returns:
            list: List of paths to audio chunks.
        """
        self.log(f"Checking if audio needs to be split")
        
        # If file is small enough, return the original path
        if self.check_file_size(audio_path):
            self.log("Audio file is below size limit, no splitting needed")
            return [audio_path]
        
        self.log(f"Audio file exceeds size limit. Splitting into chunks of {chunk_duration} seconds")
        
        # Get total duration
        duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                       "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)]
        duration_result = subprocess.run(duration_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        total_duration = float(duration_result.stdout.decode().strip())
        
        # Calculate number of chunks needed
        num_chunks = int(np.ceil(total_duration / chunk_duration))
        self.log(f"Splitting {total_duration:.2f} seconds into {num_chunks} chunks")
        
        chunk_paths = []
        
        # Create each chunk with ffmpeg
        for i in range(num_chunks):
            start_time = i * chunk_duration
            
            # Chunk file path
            chunk_path = self.temp_dir / f"chunk_{i:03d}.wav"
            chunk_paths.append(chunk_path)
            
            # FFmpeg command to extract chunk
            cmd = [
                "ffmpeg", "-i", str(audio_path),
                "-ss", str(start_time),  # Start time
                "-t", str(chunk_duration),  # Duration
                "-c:a", "pcm_s16le",  # Audio codec
                "-ar", "16000",  # Sample rate
                "-ac", "1",  # Mono
                str(chunk_path),
                "-y"  # Overwrite if exists
            ]
            
            self.log(f"Creating chunk {i+1}/{num_chunks} starting at {self.format_timestamp(start_time)}")
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Verify the chunk size
                chunk_size = os.path.getsize(chunk_path)
                if chunk_size > self.max_file_size:
                    # If still too large, try with shorter duration (recursive halving approach)
                    self.log(f"Chunk {i+1} still too large ({chunk_size / (1024 * 1024):.2f} MB). Further splitting required.")
                    os.remove(chunk_path)  # Remove the oversized chunk
                    
                    # Recalculate chunk duration for the entire split operation
                    return self.split_audio(audio_path, chunk_duration // 2)
                
                self.log(f"Chunk {i+1} created: {chunk_size / (1024 * 1024):.2f} MB")
            except Exception as e:
                self.log(f"Error creating chunk {i+1}: {str(e)}")
                # Continue with other chunks if possible
        
        self.metrics["chunks_processed"] = len(chunk_paths)
        return chunk_paths
    
    def merge_transcriptions(self, chunk_transcriptions):
        """
        Merge transcriptions from multiple chunks into a single transcription.
        
        Args:
            chunk_transcriptions (list): List of transcription dictionaries or OpenAI response objects.
            
        Returns:
            dict: Merged transcription.
        """
        if not chunk_transcriptions:
            self.log("No transcription chunks to merge")
            return {"text": "", "segments": []}
        
        self.log(f"Merging {len(chunk_transcriptions)} transcription chunks")
        
        merged = {
            "text": "",
            "segments": []
        }
        
        # Track the last segment ID
        last_segment_id = 0
        last_end_time = 0
        
        # Concatenate text and adjust segments
        for i, chunk in enumerate(chunk_transcriptions):
            try:
                self.log(f"Processing chunk {i+1}/{len(chunk_transcriptions)} for merge")
                
                # Convert OpenAI response object to dict if needed
                chunk_data = chunk
                
                # Extract text and segments, handling different response formats
                chunk_text = ""
                chunk_segments = []
                
                # Try to access text attribute (OpenAI response object)
                if hasattr(chunk_data, 'text'):
                    chunk_text = chunk_data.text
                    self.log(f"Found text in chunk {i+1} (length: {len(chunk_text)})")
                elif isinstance(chunk_data, dict) and "text" in chunk_data:
                    chunk_text = chunk_data.get("text", "")
                    self.log(f"Found text in chunk dict {i+1} (length: {len(chunk_text)})")
                    
                # Try to access segments
                if hasattr(chunk_data, 'segments'):
                    chunk_segments = chunk_data.segments
                    self.log(f"Found {len(chunk_segments)} segments in chunk {i+1}")
                elif isinstance(chunk_data, dict) and "segments" in chunk_data:
                    chunk_segments = chunk_data.get("segments", [])
                    self.log(f"Found {len(chunk_segments)} segments in chunk dict {i+1}")
                
                # If no segments, try to create a single segment from the whole text
                if not chunk_segments and chunk_text:
                    self.log(f"No segments found in chunk {i+1}, creating a single segment from text")
                    chunk_segments = [{
                        "id": 0,
                        "start": 0,
                        "end": 10,  # Arbitrary duration
                        "text": chunk_text
                    }]
                
                # Add the text with a space
                if merged["text"] and chunk_text:
                    merged["text"] += " "
                merged["text"] += chunk_text
                
                # Debug information about segments
                if hasattr(chunk_segments, '__iter__'):
                    self.log(f"Chunk {i+1} has {len(list(chunk_segments))} iterable segments")
                else:
                    self.log(f"Chunk {i+1} has non-iterable segments of type: {type(chunk_segments)}")
                
                # Adjust segment timestamps based on chunk position
                chunk_start_time = last_end_time
                
                # Process all segments in this chunk
                for segment in chunk_segments:
                    try:
                        # Handle different segment formats
                        segment_dict = segment
                        if not isinstance(segment, dict):
                            # Try to convert OpenAI segment object to dict
                            segment_dict = {
                                "id": getattr(segment, "id", i),
                                "start": getattr(segment, "start", 0),
                                "end": getattr(segment, "end", 10),
                                "text": getattr(segment, "text", "")
                            }
                        
                        # Create a new segment with adjusted values
                        new_segment = {
                            "id": last_segment_id,
                            "start": chunk_start_time + segment_dict.get("start", 0),
                            "end": chunk_start_time + segment_dict.get("end", 10),
                            "text": segment_dict.get("text", "")
                        }
                        
                        # Increment segment ID
                        last_segment_id += 1
                        
                        # Add to merged segments
                        merged["segments"].append(new_segment)
                    except Exception as e:
                        self.log(f"Error processing segment in chunk {i+1}: {str(e)}")
                        continue
                
                # Update the last end time if we added segments
                if merged["segments"]:
                    last_segment = merged["segments"][-1]
                    last_end_time = last_segment["end"]
                    self.log(f"Updated last_end_time to {last_end_time}")
                    
            except Exception as e:
                self.log(f"Error merging chunk {i+1}: {str(e)}")
                # Continue with other chunks
        
        self.log(f"Merge complete. Output has {len(merged['segments'])} segments and text length {len(merged['text'])}")
        return merged
    
    def transcribe_with_openai(self, audio_path, speaker_segments=None):
        """
        Transcribe audio using OpenAI's Whisper model.
        If speaker_segments is provided, segments the transcription by speaker.
        Handles large files by splitting into chunks.
        
        Args:
            audio_path (str): Path to the audio file.
            speaker_segments (list, optional): List of speaker segments from diarization.
            
        Returns:
            dict: The transcription result with timestamps and speakers.
        """
        if not audio_path or not Path(audio_path).exists():
            self.log("No audio file found for transcription")
            return {"text": ""}
            
        self.log(f"Transcribing audio from {audio_path}")
        
        try:
            # Split audio into chunks if necessary
            audio_chunks = self.split_audio(audio_path)
            
            # If we have multiple chunks
            if len(audio_chunks) > 1:
                self.log(f"Transcribing {len(audio_chunks)} audio chunks sequentially")
                
                # Process each chunk
                chunk_transcriptions = []
                for i, chunk_path in enumerate(audio_chunks):
                    self.log(f"Transcribing chunk {i+1}/{len(audio_chunks)}")
                    
                    try:
                        # Process chunk
                        with open(chunk_path, "rb") as audio_file:
                            chunk_response = self.client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                response_format="verbose_json",
                                timestamp_granularities=["segment"]
                            )
                        
                        # Convert OpenAI response to a standard dictionary we can work with
                        chunk_transcription = {
                            "text": chunk_response.text if hasattr(chunk_response, 'text') else "",
                            "segments": []
                        }
                        
                        # Process segments
                        if hasattr(chunk_response, 'segments'):
                            for segment in chunk_response.segments:
                                segment_data = {
                                    "id": segment.id if hasattr(segment, 'id') else 0,
                                    "start": segment.start if hasattr(segment, 'start') else 0,
                                    "end": segment.end if hasattr(segment, 'end') else 0,
                                    "text": segment.text if hasattr(segment, 'text') else ""
                                }
                                chunk_transcription["segments"].append(segment_data)
                        
                        # Store the result
                        chunk_transcriptions.append(chunk_transcription)
                        self.log(f"Successfully transcribed chunk {i+1} with {len(chunk_transcription['segments'])} segments")
                    except Exception as e:
                        self.log(f"Error transcribing chunk {i+1}: {str(e)}")
                        # Continue with other chunks
                
                # Merge the transcription chunks
                self.log("Merging transcriptions from all chunks")
                merged_transcription = self.merge_transcriptions(chunk_transcriptions)
                
                if not merged_transcription or not merged_transcription.get("text"):
                    self.log("Merge resulted in empty transcription")
                    if chunk_transcriptions:
                        # If merge failed but we have individual transcriptions, concatenate texts as fallback
                        fallback_text = " ".join(chunk["text"] for chunk in chunk_transcriptions if chunk.get("text"))
                        self.log(f"Created fallback transcription from concatenated texts: {len(fallback_text)} characters")
                        merged_transcription = {
                            "text": fallback_text,
                            "segments": []
                        }
                
                # Add speaker information if available
                if speaker_segments and merged_transcription.get("segments"):
                    self.assign_speakers_to_segments(merged_transcription, speaker_segments)
                
                return merged_transcription
            else:
                # Single chunk processing (original path)
                self.log("Processing single audio file")
                with open(audio_path, "rb") as audio_file:
                    transcription_response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )
                
                # Get the segments with timestamps
                result = {
                    "text": transcription_response.text if hasattr(transcription_response, 'text') else "",
                    "segments": []
                }
                
                # Process segments
                if hasattr(transcription_response, 'segments'):
                    for segment in transcription_response.segments:
                        segment_data = {
                            "id": segment.id if hasattr(segment, 'id') else 0,
                            "start": segment.start if hasattr(segment, 'start') else 0,
                            "end": segment.end if hasattr(segment, 'end') else 0,
                            "text": segment.text if hasattr(segment, 'text') else ""
                        }
                        
                        result["segments"].append(segment_data)
                
                # Add speaker information if available
                if speaker_segments and result.get("segments"):
                    self.assign_speakers_to_segments(result, speaker_segments)
                
                self.log(f"Transcription successful: {len(result['segments'])} segments")
                return result
                
        except Exception as e:
            self.log(f"Error transcribing audio: {str(e)}")
            return {"text": "", "segments": []}
    
    def perform_speaker_diarization(self, audio_path):
        """
        Perform speaker diarization to identify different speakers.
        
        Args:
            audio_path (Path): Path to the audio file.
            
        Returns:
            dict: Speaker diarization results with timestamps.
        """
        if not self.diarization_pipeline:
            self.log("Speaker diarization not available - HF token not provided")
            return None
            
        self.log(f"Performing speaker diarization on {audio_path}")
        
        try:
            # Run the diarization pipeline
            diarization = self.diarization_pipeline(audio_path)
            
            # Process the output into usable speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            
            # Count unique speakers
            unique_speakers = set(segment["speaker"] for segment in speaker_segments)
            self.metrics["speakers_detected"] = len(unique_speakers)
            
            self.log(f"Detected {len(unique_speakers)} unique speakers with {len(speaker_segments)} segments")
            return speaker_segments
        except Exception as e:
            self.log(f"Error in speaker diarization: {str(e)}")
            return None
    
    def assign_speakers_to_segments(self, transcription, speaker_segments):
        """
        Assign speakers to transcription segments.
        
        Args:
            transcription (dict): Transcription with segments.
            speaker_segments (list): Speaker diarization segments.
            
        Returns:
            None: Modifies transcription in place.
        """
        for segment in transcription["segments"]:
            # Find the speaker who speaks the most during this segment
            speakers_during_segment = {}
            segment_start, segment_end = segment["start"], segment["end"]
            
            for speaker_segment in speaker_segments:
                s_start, s_end = speaker_segment["start"], speaker_segment["end"]
                
                # Check for overlap
                if s_end > segment_start and s_start < segment_end:
                    # Calculate overlap duration
                    overlap_start = max(segment_start, s_start)
                    overlap_end = min(segment_end, s_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    # Add to speaker's total time
                    speaker = speaker_segment["speaker"]
                    if speaker in speakers_during_segment:
                        speakers_during_segment[speaker] += overlap_duration
                    else:
                        speakers_during_segment[speaker] = overlap_duration
            
            # Assign the dominant speaker
            if speakers_during_segment:
                dominant_speaker = max(speakers_during_segment, key=speakers_during_segment.get)
                segment["speaker"] = dominant_speaker
    
    def add_punctuation(self, text):
        """
        Add missing punctuation to text using NLP techniques.
        
        Args:
            text (str): Text to process.
            
        Returns:
            str: Text with added punctuation.
        """
        if not text:
            return text
            
        self.log("Adding punctuation to transcript")
        
        try:
            # Pattern for detecting existing sentence endings
            sentence_end_pattern = r'(?<=[.!?])\s+'
            
            # Capitalize the first letter of the text if it's lowercase
            if text and text[0].islower():
                text = text[0].upper() + text[1:]
            
            # Basic preprocessing - normalize spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 1. Add periods where there are clear sentence boundaries
            # Look for lowercase word followed by capitalized word
            text = re.sub(r'(?<=[a-z])\s+(?=[A-Z])', '. ', text)
            
            # 2. Use NLTK to find potential sentence boundaries and add periods
            # This is more conservative to avoid overriding existing punctuation
            potential_sentences = sent_tokenize(text)
            
            # Only modify if NLTK found more than one sentence
            if len(potential_sentences) > 1:
                # Reconstruct text with proper sentence endings
                processed_text = ""
                for sentence in potential_sentences:
                    # Check if the sentence already ends with punctuation
                    if not sentence.strip().endswith(('.', '!', '?')):
                        sentence += '.'
                    processed_text += sentence + " "
                text = processed_text.strip()
            
            # 3. Add question marks based on question words
            # Find sentences that start with question words but don't end with ?
            question_pattern = r'(?<=[.!?]\s+|\A)(Why|What|Who|Where|When|How|Is|Are|Can|Could|Would|Will|Do|Does|Did)(?i)[^.!?]*(?<!\?)\s+(?=[A-Z]|$)'
            text = re.sub(question_pattern, lambda m: m.group(0).strip() + '? ', text)
            
            # 4. Add commas for natural pauses
            # After introductory phrases
            intro_phrases = ["However", "Therefore", "Meanwhile", "Nevertheless", 
                           "Moreover", "Furthermore", "In addition", "As a result", 
                           "For example", "In fact", "On the other hand"]
            for phrase in intro_phrases:
                text = re.sub(fr'\b{phrase}\b(?!,)', f'{phrase},', text)
            
            # Add commas before conjunctions joining independent clauses
            conj_pattern = r'(?<=[a-z])\s+(and|but|or|so|for|nor|yet)\s+(?=[A-Z])'
            text = re.sub(conj_pattern, lambda m: f', {m.group(1)} ', text)
            
            # 5. Add missing periods at the end of text if needed
            if text and not text.strip().endswith(('.', '!', '?')):
                text += '.'
            
            return text
        
        except Exception as e:
            self.log(f"Error adding punctuation: {str(e)}")
            return text  # Return original text if something goes wrong
    
    def enhance_transcription(self, segments):
        """
        Enhance the transcription by adding punctuation to each segment.
        
        Args:
            segments (list): List of transcription segments.
            
        Returns:
            list: Enhanced segments with punctuation.
        """
        enhanced_segments = []
        
        for segment in segments:
            segment_copy = segment.copy()
            segment_copy["text"] = self.add_punctuation(segment["text"])
            enhanced_segments.append(segment_copy)
        
        return enhanced_segments
    
    def format_transcript(self, transcription, with_speakers=True, with_timestamps=True):
        """
        Format the transcription for readability.
        
        Args:
            transcription (dict): Transcription results with segments.
            with_speakers (bool): Whether to include speaker labels.
            with_timestamps (bool): Whether to include timestamps.
            
        Returns:
            str: Formatted transcript text.
        """
        if not transcription or "segments" not in transcription or not transcription["segments"]:
            return ""
        
        formatted_lines = []
        current_speaker = None
        
        for segment in transcription["segments"]:
            line = ""
            
            # Add timestamp if requested
            if with_timestamps:
                start_time = self.format_timestamp(segment["start"])
                line += f"[{start_time}] "
            
            # Add speaker if available and requested
            if with_speakers and "speaker" in segment:
                speaker = segment["speaker"]
                # Only add speaker label if different from the previous one
                if speaker != current_speaker:
                    line += f"{speaker}: "
                    current_speaker = speaker
            
            # Add the text
            line += segment["text"]
            
            formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def to_srt_timestamp(self, seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def transcribe_lecture(self, video_path, add_punctuation=True, with_diarization=True, output_format="txt", chunk_duration=600):
        """
        Complete pipeline to transcribe a lecture.
        
        Args:
            video_path (str): Path to the video/audio file.
            add_punctuation (bool): Whether to add punctuation to the transcript.
            with_diarization (bool): Whether to perform speaker diarization.
            output_format (str): Format for the output ("txt", "json", or "srt").
            chunk_duration (int): Maximum duration in seconds for each audio chunk.
            
        Returns:
            dict: Results of the transcription with metadata.
        """
        results = {
            "source_path": video_path,
            "transcript": "",
            "enhanced_transcript": "",
            "formatted_transcript": "",
            "speakers_detected": 0,
            "processing_time": 0,
            "chunks_processed": 0,
            "output_file": None
        }
        
        start_time = time.time()
        
        try:
            # Verify the file exists
            if not os.path.exists(video_path):
                self.log(f"Error: File not found at {video_path}")
                results["error"] = f"File not found at {video_path}"
                return results
            
            # 1. Extract audio (if it's a video file)
            self.log(f"Starting audio extraction from {video_path}")
            audio_path = None
            
            if video_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                audio_path = self.extract_audio(video_path)
                if not audio_path:
                    self.log("Error: Failed to extract audio from video")
                    results["error"] = "Failed to extract audio from video"
                    results["processing_time"] = time.time() - start_time
                    return results
            else:
                # Assume it's already an audio file
                audio_path = video_path
                
            self.log(f"Audio ready for processing: {audio_path}")
            
            # 2. Perform speaker diarization if requested and available
            speaker_segments = None
            if with_diarization and self.diarization_pipeline:
                self.log("Starting speaker diarization")
                speaker_segments = self.perform_speaker_diarization(audio_path)
                if speaker_segments:
                    results["speakers_detected"] = self.metrics["speakers_detected"]
                else:
                    self.log("Speaker diarization produced no results")
            
            # 3. Transcribe audio with OpenAI (handles chunking internally)
            self.log("Starting transcription with OpenAI")
            transcription = self.transcribe_with_openai(audio_path, speaker_segments)
            
            if not transcription or not transcription.get("text"):
                self.log("Error: Transcription failed or returned empty result")
                results["error"] = "Transcription failed or returned empty result"
                results["processing_time"] = time.time() - start_time
                return results
                
            results["transcript"] = transcription["text"]
            results["chunks_processed"] = self.metrics["chunks_processed"]
            
            # 4. Enhance transcription with punctuation if requested
            if add_punctuation and transcription.get("segments"):
                self.log("Enhancing transcription with punctuation")
                enhanced_segments = self.enhance_transcription(transcription["segments"])
                transcription["segments"] = enhanced_segments
                
                # Regenerate full text from enhanced segments
                enhanced_text = " ".join([segment["text"] for segment in enhanced_segments])
                results["enhanced_transcript"] = enhanced_text
            
            # 5. Format transcript for output
            self.log("Formatting transcript with speakers and timestamps")
            has_speakers = any("speaker" in segment for segment in transcription.get("segments", []))
            formatted_transcript = self.format_transcript(
                transcription, 
                with_speakers=has_speakers, 
                with_timestamps=True
            )
            results["formatted_transcript"] = formatted_transcript
            
            # 6. Save to file based on requested format
            self.log(f"Saving transcript in {output_format.upper()} format")
            file_path = self.save_transcript(
                transcription,
                output_format=output_format,
                base_name=os.path.splitext(os.path.basename(video_path))[0]
            )
            
            if file_path:
                results["output_file"] = str(file_path)
                self.log(f"Saved transcript to {file_path}")
            
            # Calculate processing time
            results["processing_time"] = time.time() - start_time
            self.log(f"Transcription completed in {results['processing_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.log(f"Unhandled exception in transcribe_lecture: {str(e)}")
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
            return results
    
    def save_transcript(self, transcription, output_format="txt", base_name="transcript"):
        """
        Save transcript to file in various formats.
        
        Args:
            transcription (dict): Transcription data with segments.
            output_format (str): Format to save ("txt", "json", or "srt").
            base_name (str): Base name for the output file.
            
        Returns:
            Path: Path to the saved file.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            if output_format.lower() == "txt":
                # Save as plain text with formatting
                output_path = Path(f"{base_name}_transcript_{timestamp}.txt")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("===== Lecture Transcript =====\n\n")
                    f.write(f"Source: {base_name}\n")
                    f.write(f"Transcription date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Duration: {self.metrics.get('audio_duration', 0):.2f} seconds\n")
                    
                    if self.metrics.get("chunks_processed", 0) > 1:
                        f.write(f"Audio processed in {self.metrics['chunks_processed']} chunks\n")
                    
                    if self.metrics.get("speakers_detected", 0) > 1:
                        f.write(f"Speakers detected: {self.metrics['speakers_detected']}\n")
                    
                    f.write("\n----- Transcript -----\n\n")
                    
                    # Write formatted transcript
                    has_speakers = any("speaker" in segment for segment in transcription.get("segments", []))
                    formatted_text = self.format_transcript(
                        transcription, 
                        with_speakers=has_speakers, 
                        with_timestamps=True
                    )
                    f.write(formatted_text)
                
            elif output_format.lower() == "json":
                # Save as JSON with all metadata
                output_path = Path(f"{base_name}_transcript_{timestamp}.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": {
                            "source": base_name,
                            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "duration": self.metrics.get('audio_duration', 0),
                            "chunks_processed": self.metrics.get("chunks_processed", 0),
                            "speakers_detected": self.metrics.get("speakers_detected", 0)
                        },
                        "transcript": transcription
                    }, f, indent=2)
                    
            elif output_format.lower() == "srt":
                # Save as SubRip subtitle format
                output_path = Path(f"{base_name}_transcript_{timestamp}.srt")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    for i, segment in enumerate(transcription.get("segments", []), 1):
                        # Convert timestamps to SRT format (HH:MM:SS,mmm)
                        start_time = self.to_srt_timestamp(segment["start"])
                        end_time = self.to_srt_timestamp(segment["end"])
                        
                        # Format the text (with speaker if available)
                        text = segment["text"]
                        if "speaker" in segment:
                            text = f"{segment['speaker']}: {text}"
                        
                        # Write SRT entry
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{text}\n\n")
            
            else:
                self.log(f"Unsupported output format: {output_format}")
                return None
                
            return output_path
            
        except Exception as e:
            self.log(f"Error saving transcript: {str(e)}")
            return None
    
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Lecture Transcription Tool")
    parser.add_argument("video_path", help="Path to the lecture video or audio file")
    parser.add_argument("--openai-key", help="OpenAI API key (optional, can use env var OPENAI_API_KEY)")
    parser.add_argument("--hf-token", help="HuggingFace API token for speaker diarization")
    parser.add_argument("--no-punctuation", action="store_true", help="Disable punctuation enhancement")
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--format", choices=["txt", "json", "srt"], default="txt", help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--chunk-size", type=int, default=600, help="Maximum duration in seconds for each audio chunk (default: 600)")
    
    args = parser.parse_args()
    
    print(f"Starting enhanced transcription for: {args.video_path}")
    transcriber = EnhancedLectureTranscriber(
        api_key=args.openai_key, 
        hf_token=args.hf_token,
        verbose=args.verbose
    )

    # Transcribe the lecture
    results = transcriber.transcribe_lecture(
        args.video_path,
        add_punctuation=not args.no_punctuation,
        with_diarization=not args.no_diarization,
        output_format=args.format,
        chunk_duration=args.chunk_size
    )

    # Print results to console
    print("\n===== Lecture Transcription Results =====")
    print(f"Source: {results['source_path']}")
    print(f"Processed in {results['processing_time']:.2f} seconds")
    
    if results.get("chunks_processed", 0) > 1:
        print(f"Audio processed in {results['chunks_processed']} chunks")
        
    if results.get("speakers_detected", 0) > 1:
        print(f"Detected {results['speakers_detected']} speakers")
            
    if "error" in results:
        print(f"\nError: {results['error']}")
    
    if results.get("output_file"):
        print(f"\nTranscript saved to: {results['output_file']}")

    # Cleanup temp files
    transcriber.cleanup()

if __name__ == "__main__":
    main()
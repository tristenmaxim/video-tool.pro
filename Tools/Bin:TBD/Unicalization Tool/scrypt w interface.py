import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import random
import string

def select_file_or_folder():
    """
    Open a dialog to select either a single video file or a folder containing videos.
    
    Returns:
        tuple: (path, is_folder) where:
            - path is the selected file or folder path
            - is_folder is True if a folder was selected, False if a file was selected
    """
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Create a simple dialog to ask whether to select a file or folder
    dialog = tk.Toplevel(root)
    dialog.title("Video Uniqualizer")
    dialog.geometry("400x150")
    dialog.resizable(False, False)
    
    # Center the dialog
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    selection_type = tk.StringVar(value="file")
    result = {"path": None, "is_folder": False, "done": False}
    
    def select_path():
        if selection_type.get() == "file":
            path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                    ("All files", "*.*")
                ]
            )
            result["is_folder"] = False
        else:
            path = filedialog.askdirectory(title="Select Folder Containing Videos")
            result["is_folder"] = True
            
        result["path"] = path
        result["done"] = True
        dialog.destroy()
    
    tk.Label(dialog, text="Do you want to process a single video or an entire folder?", 
             pady=10).pack()
    
    options_frame = tk.Frame(dialog)
    options_frame.pack(pady=5)
    
    tk.Radiobutton(options_frame, text="Single Video File", variable=selection_type, 
                   value="file").pack(anchor=tk.W)
    tk.Radiobutton(options_frame, text="Folder of Videos", variable=selection_type, 
                   value="folder").pack(anchor=tk.W)
    
    tk.Button(dialog, text="Select", command=select_path, width=10).pack(pady=10)
    
    # Wait for the dialog to be closed
    dialog.protocol("WM_DELETE_WINDOW", lambda: (setattr(result, "done", True), dialog.destroy()))
    
    # Wait for selection to complete
    root.wait_window(dialog)
    root.destroy()
    
    return result["path"], result["is_folder"]

def generate_random_id(length=5):
    """Generate a random ID of specified length using letters and numbers"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_random_metadata():
    """Generate random metadata values"""
    # Generate a random title (15-25 characters)
    title_length = random.randint(15, 25)
    random_title = ''.join(random.choices(string.ascii_letters + string.digits, k=title_length))
    
    # Generate a random comment (30-50 characters)
    comment_length = random.randint(30, 50)
    random_comment = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=comment_length))
    
    # Generate a random creation time in the last year
    current_time = time.time()
    one_year_ago = current_time - (365 * 24 * 60 * 60)
    random_time = random.uniform(one_year_ago, current_time)
    creation_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(random_time))
    
    return {
        "title": random_title,
        "comment": random_comment,
        "creation_time": creation_time
    }

def process_video_fast(input_path, output_path, progress_callback=None):
    """
    Apply minimal modifications to a video to make it unique but visually almost identical.
    Uses FFmpeg directly for speed and efficiency.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        progress_callback: Function to call with progress updates
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        start_time = time.time()
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' does not exist.")
            return False
        
        # Update progress
        if progress_callback:
            progress_callback(input_path, 5, "Starting")
        
        # Generate unique metadata
        metadata = generate_random_metadata()
        
        # Use FFmpeg to make minimal changes to the video
        try:
            # Create FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-map_metadata', '-1',  # Strip existing metadata
                '-metadata', f"title={metadata['title']}",
                '-metadata', f"comment={metadata['comment']}",
                '-metadata', f"creation_time={metadata['creation_time']}",
                # Apply minimal video modification (slightly adjust contrast)
                '-vf', 'eq=contrast=1.01:brightness=0.001',
                # Use fast encoding settings
                '-c:v', 'libx264',
                '-preset', 'ultrafast',  # Fastest encoding speed
                '-crf', '23',  # Good quality but faster
                '-c:a', 'copy',  # Copy audio without re-encoding
                output_path,
                '-y'  # Overwrite output file if it exists
            ]
            
            if progress_callback:
                progress_callback(input_path, 10, "Processing")
            
            # Run FFmpeg
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                universal_newlines=True
            )
            
            # Monitor progress by parsing FFmpeg output
            for line in process.stderr:
                if "time=" in line and progress_callback:
                    # Extract time information from FFmpeg output
                    time_parts = line.split("time=")[1].split()[0].split(":")
                    if len(time_parts) == 3:
                        hours, minutes, seconds = time_parts
                        # Approximate progress (this is not perfectly accurate)
                        progress = min(95, 10 + 85 * (float(hours) * 3600 + float(minutes) * 60 + float(seconds)) / 3600)
                        progress_callback(input_path, progress, "Processing")
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                processing_time = time.time() - start_time
                if progress_callback:
                    progress_callback(input_path, 100, "Complete")
                print(f'✓ Finished: {os.path.basename(input_path)} ({processing_time:.1f}s)')
                return True
            else:
                print(f"Error processing {input_path}: FFmpeg returned code {process.returncode}")
                return False
                
        except FileNotFoundError:
            print("Error: FFmpeg executable not found. Please install FFmpeg on your system.")
            print("For macOS: brew install ffmpeg")
            print("For Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("For Windows: Download from https://ffmpeg.org/download.html")
            return False
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def is_video_file(filename):
    """Check if a file is a video based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    ext = os.path.splitext(filename)[1].lower()
    return ext in video_extensions

def create_progress_ui():
    """Create a simple UI to display progress of video processing"""
    window = tk.Tk()
    window.title("Video Uniqualizer - Processing")
    window.geometry("600x400")
    window.resizable(True, True)
    
    # Center the window
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
    
    # Frame for the list of files and their progress
    list_frame = tk.Frame(window)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Scrollable list for progress
    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    progress_list = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
    progress_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar.config(command=progress_list.yview)
    
    # Status label
    status_label = tk.Label(window, text="Initializing...", anchor=tk.W)
    status_label.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    # Cancel button
    cancel_button = tk.Button(window, text="Cancel", state=tk.DISABLED)
    cancel_button.pack(pady=(0, 10))
    
    window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing the window
    
    return window, progress_list, status_label, cancel_button

def process_videos(path, is_folder):
    """
    Process either a single video file or all videos in a folder.
    Videos are processed sequentially, one by one.
    Results are saved in a folder with the same name as the input folder with "_unique" appended.
    Each file is named according to the pattern: "old-name-uniqualisation-XXXXX" where XXXXX is random.
    
    Args:
        path: Path to video file or folder
        is_folder: True if path is a folder, False if it's a file
    """
    # Create a list of files to process
    files_to_process = []
    
    if is_folder:
        # Get all video files in the folder (non-recursive)
        input_folder = path
        # Create output folder (same name as input folder but with "_unique" appended)
        folder_name = os.path.basename(input_folder)
        parent_dir = os.path.dirname(input_folder)
        output_folder = os.path.join(parent_dir, folder_name + "_unique")
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Find all video files in the input folder
        for filename in os.listdir(input_folder):
            full_path = os.path.join(input_folder, filename)
            if os.path.isfile(full_path) and is_video_file(filename):
                # Skip files that already have "uniqualisation" in their name
                if "uniqualisation" not in filename:
                    # Generate a random 5-character ID
                    random_id = generate_random_id(5)
                    
                    # Create the new filename with the pattern: old-name-uniqualisation-XXXXX.ext
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{name}-uniqualisation-{random_id}{ext}"
                    
                    # Create the output path in the _unique folder
                    output_path = os.path.join(output_folder, new_filename)
                    files_to_process.append((full_path, output_path))
    else:
        # Just one file - create output file with the specified pattern
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        
        # Generate a random 5-character ID
        random_id = generate_random_id(5)
        
        # Create the new filename with the pattern: old-name-uniqualisation-XXXXX.ext
        new_filename = f"{name}-uniqualisation-{random_id}{ext}"
        
        # Determine output directory (same as input file)
        output_dir = os.path.dirname(path)
        output_path = os.path.join(output_dir, new_filename)
        
        files_to_process.append((path, output_path))
    
    if not files_to_process:
        messagebox.showinfo("No Videos Found", 
                            "No valid video files were found to process.")
        return
    
    # Create progress UI
    window, progress_list, status_label, cancel_button = create_progress_ui()
    
    # Add all files to the list
    file_progress = {}
    for input_path, output_path in files_to_process:
        basename = os.path.basename(input_path)
        list_index = len(file_progress)
        progress_list.insert(tk.END, f"{basename}: Waiting...")
        file_progress[input_path] = {
            "list_index": list_index,
            "progress": 0,
            "status": "Waiting",
            "output_path": output_path
        }
    
    # Variable to track if processing should be canceled
    cancel_flag = {"value": False}
    
    def cancel_processing():
        cancel_flag["value"] = True
        cancel_button.config(text="Canceling...", state=tk.DISABLED)
    
    cancel_button.config(command=cancel_processing, state=tk.NORMAL)
    
    # Create a function to update progress
    def update_progress(file_path, progress, status):
        if file_path in file_progress:
            file_progress[file_path]["progress"] = progress
            file_progress[file_path]["status"] = status
            
            # Update the UI
            idx = file_progress[file_path]["list_index"]
            basename = os.path.basename(file_path)
            progress_list.delete(idx)
            progress_list.insert(idx, f"{basename}: {progress:.1f}% - {status}")
            progress_list.see(idx)
            
            # Update overall status
            completed = sum(1 for info in file_progress.values() if info["progress"] >= 100)
            current = sum(1 for info in file_progress.values() if 0 < info["progress"] < 100)
            total = len(file_progress)
            
            if current > 0:
                current_file = [os.path.basename(path) for path, info in file_progress.items() 
                               if 0 < info["progress"] < 100][0]
                status_label.config(text=f"Processing file {completed+1}/{total}: {current_file}")
            else:
                status_label.config(text=f"Completed {completed}/{total} videos")
    
    # Process all videos sequentially
    def process_all_videos():
        for input_path, output_path in files_to_process:
            if cancel_flag["value"]:
                break
                
            # Process the current video
            success = process_video_fast(input_path, output_path, 
                                       progress_callback=update_progress)
            
            # If processing was canceled, stop
            if cancel_flag["value"]:
                break
        
        # Update UI after all processing is done
        window.after(100, processing_complete)
    
    def processing_complete():
        completed = sum(1 for info in file_progress.values() if info["progress"] >= 100)
        total = len(file_progress)
        
        if cancel_flag["value"]:
            status_message = "Processing canceled."
        else:
            status_message = f"Processing complete. {completed}/{total} videos processed successfully."
            
            # If we processed a folder, show where the results are stored
            if is_folder and completed > 0:
                folder_name = os.path.basename(os.path.dirname(file_progress[list(file_progress.keys())[0]]["output_path"]))
                status_message += f"\nResults saved in folder: {folder_name}"
                status_message += f"\nFiles are named using the pattern: original-name-uniqualisation-XXXXX"
        
        status_label.config(text=status_message)
        cancel_button.config(text="Close", command=window.destroy, state=tk.NORMAL)
        window.protocol("WM_DELETE_WINDOW", window.destroy)
    
    # Start the processing in a separate thread
    threading.Thread(target=process_all_videos, daemon=True).start()
    
    # Start the UI main loop
    window.mainloop()

def main():
    # Select file or folder
    path, is_folder = select_file_or_folder()
    
    if not path:
        print("No file or folder selected. Exiting.")
        return
    
    # Process the videos
    process_videos(path, is_folder)

if __name__ == "__main__":
    main()
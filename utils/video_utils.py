"""
Video processing utilities for loading, saving, and manipulating video data.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return self.videos.shape[0]

    def __getitem__(self, idx):
        return self.videos[idx]
    
def load_video_as_tensor(video_path, target_size=None):
    """
    Load video file and convert to tensor format.
    
    Args:
        video_path: Path to video file (.mp4, .avi, etc. or .pt/.pth tensor files)
        max_frames: Maximum number of frames per batch
        target_size: Target size for frames (height, width)
        
    Returns:
        Video tensor of shape (C, T, H, W) or empty tensor if failed
    """
    if not os.path.exists(video_path):
        print(f'Video file not found: {video_path}')
        return torch.empty(0)

    # Handle video files (.mp4, .avi, etc.)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return torch.empty(0)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if target_size:
            # Notice that cv2.resize's size parameter is (width, height)
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    result_tensor = torch.tensor(np.array(frames)).float() # [T, H, W, C]
    result_tensor = result_tensor.permute(3, 0, 1, 2) / 255.0  # Convert to [C, T, H, W] and normalize to [0, 1]
    
    return result_tensor


def split_video_tensor(tensor, max_frames=25):
    """
    Split video tensor into chunks of max_frames, stack along batch dimension,
    and discard incomplete chunks.
    
    Args:
        tensor: Video tensor of shape (B, C, T, H, W) or (C, T, H, W)
        max_frames: Maximum number of frames per chunk
        
    Returns:
        Stacked video tensor of shape (B * num_chunks, C, max_frames, H, W)
        where num_chunks = T // max_frames (discards remainder)
    """
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)  # (C, T, H, W) -> (1, C, T, H, W)
    elif tensor.dim() != 5:
        raise ValueError("Input tensor must be 4D or 5D")
    
    B, C, T, H, W = tensor.shape
    
    num_chunks = T // max_frames
    if num_chunks == 0:
        return torch.empty(0, *tensor.shape[1:])  
    tensor = tensor[:, :, :num_chunks * max_frames, :, :]  # Discard remainder
    
    # [B, C, num_chunks, max_frames, H, W]
    chunks = tensor.reshape(B, C, num_chunks, max_frames, H, W)
    
    # [B, num_chunks, C, max_frames, H, W]
    chunks = chunks.permute(0, 2, 1, 3, 4, 5)
    
    # [B * num_chunks, C, max_frames, H, W]
    stacked = chunks.reshape(-1, C, max_frames, H, W)
    
    return stacked
    

def save_tensor_as_video(tensor, output_path, fps=25):
    """
    Save tensor as video file.
    
    Args:
        tensor: Video tensor of shape (B, C, T, H, W) or (C, T, H, W)
        output_path: Output video file path
        fps: Frame rate for output video
    """
    # Handle batch dimension
    if tensor.dim() == 5:
        tensor = tensor[0]  # Take first batch
    
    # Permute dimensions: (C, T, H, W) -> (T, H, W, C)
    tensor = tensor.permute(1, 2, 3, 0)
    
    # Ensure values are in [0, 1] range and convert to [0, 255]
    tensor = torch.clamp(tensor, 0, 1)
    frames = (tensor * 255).cpu().numpy().astype(np.uint8)
    
    T, H, W, C = frames.shape
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    # Write frames
    for t in range(T):
        frame = frames[t]
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")


def create_comparison_video(original, reconstructed, output_path, fps=25):
    """
    Create side-by-side comparison video with text labels.
    
    Args:
        original: Original video tensor (B, C, T, H, W) or (C, T, H, W)
        reconstructed: Reconstructed video tensor (B, C, T, H, W) or (C, T, H, W)
        output_path: Output video file path
        fps: Frame rate for output video
    """
    # Handle batch dimensions
    if original.dim() == 5:
        original = original[0]
    if reconstructed.dim() == 5:
        reconstructed = reconstructed[0]
    
    # Permute dimensions: (C, T, H, W) -> (T, H, W, C)
    original = original.permute(1, 2, 3, 0)
    reconstructed = reconstructed.permute(1, 2, 3, 0)
    
    # Ensure values are in [0, 1] range
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # Convert to numpy arrays
    orig_frames = (original * 255).cpu().numpy().astype(np.uint8)
    recon_frames = (reconstructed * 255).cpu().numpy().astype(np.uint8)

    assert orig_frames.shape == recon_frames.shape, "Original and reconstructed videos must have the same shape"
    
    T, H, W, C = orig_frames.shape
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add space for text labels at the top (40 pixels)
    text_height = 40
    total_height = H + text_height
    
    # Create side-by-side comparison video (original | reconstructed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W * 2, total_height))
    
    for t in range(T):
        orig_frame = orig_frames[t]
        recon_frame = recon_frames[t]
        
        # Create frames with text space at the top
        orig_frame_with_text = np.ones((total_height, W, C), dtype=np.uint8) * 255
        recon_frame_with_text = np.ones((total_height, W, C), dtype=np.uint8) * 255
        
        # Place the actual frames below the text area
        orig_frame_with_text[text_height:, :, :] = orig_frame
        recon_frame_with_text[text_height:, :, :] = recon_frame
        
        # Convert to BGR for OpenCV text rendering
        orig_frame_bgr = cv2.cvtColor(orig_frame_with_text, cv2.COLOR_RGB2BGR)
        recon_frame_bgr = cv2.cvtColor(recon_frame_with_text, cv2.COLOR_RGB2BGR)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (0, 0, 0)  # Black text
        
        # Calculate text positions (center horizontally)
        orig_text = "Original"
        recon_text = "Reconstructed"
        
        orig_text_size = cv2.getTextSize(orig_text, font, font_scale, font_thickness)[0]
        recon_text_size = cv2.getTextSize(recon_text, font, font_scale, font_thickness)[0]
        
        orig_text_x = (W - orig_text_size[0]) // 2
        recon_text_x = (W - recon_text_size[0]) // 2
        text_y = 25  # Position from top
        
        # Add text to frames
        cv2.putText(orig_frame_bgr, orig_text, (orig_text_x, text_y), font, font_scale, text_color, font_thickness)
        cv2.putText(recon_frame_bgr, recon_text, (recon_text_x, text_y), font, font_scale, text_color, font_thickness)
        
        # Horizontally concatenate frames
        combined_frame = np.hstack([orig_frame_bgr, recon_frame_bgr])
        
        out.write(combined_frame)
    
    out.release()
    print(f"Comparison video saved to: {output_path}")

def create_four_panel_video(cover_orig, cover_recon, secret_orig, secret_recon, output_path, fps=25):
    """
    Create four-panel comparison video for secret mode.
    Layout:
    +------------------+------------------+
    |  Cover Original  | Cover Recon      |
    +------------------+------------------+
    | Secret Original  | Secret Recon     |
    +------------------+------------------+
    
    Args:
        cover_orig: Original cover video tensor (B, C, T, H, W) or (C, T, H, W)
        cover_recon: Reconstructed cover video tensor (B, C, T, H, W) or (C, T, H, W)
        secret_orig: Original secret video tensor (B, C, T, H, W) or (C, T, H, W)
        secret_recon: Reconstructed secret video tensor (B, C, T, H, W) or (C, T, H, W)
        output_path: Output video file path
        fps: Frame rate for output video
    """
    # Handle batch dimensions
    videos = [cover_orig, cover_recon, secret_orig, secret_recon]
    processed_videos = []
    
    for video in videos:
        if video.dim() == 5:
            video = video[0]
        # Permute dimensions: (C, T, H, W) -> (T, H, W, C)
        video = video.permute(1, 2, 3, 0)
        # Ensure values are in [0, 1] range
        video = torch.clamp(video, 0, 1)
        # Convert to numpy arrays
        video_frames = (video * 255).cpu().numpy().astype(np.uint8)
        processed_videos.append(video_frames)
    
    cover_orig_frames, cover_recon_frames, secret_orig_frames, secret_recon_frames = processed_videos
    
    # Ensure all videos have the same shape
    min_frames = min(len(frames) for frames in processed_videos)
    T, H, W, C = cover_orig_frames[:min_frames].shape
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add space for text labels at the top (40 pixels)
    text_height = 40
    panel_height = H + text_height
    total_height = panel_height * 2
    total_width = W * 2
    
    # Create four-panel comparison video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    
    for t in range(min_frames):
        # Get frames for current time step
        cover_orig_frame = cover_orig_frames[t]
        cover_recon_frame = cover_recon_frames[t]
        secret_orig_frame = secret_orig_frames[t]
        secret_recon_frame = secret_recon_frames[t]
        
        # Create panels with text space at the top
        panels = []
        frames = [cover_orig_frame, cover_recon_frame, secret_orig_frame, secret_recon_frame]
        labels = ["Cover Original", "Cover Reconstructed", "Secret Original", "Secret Reconstructed"]
        
        for frame, label in zip(frames, labels):
            # Create panel with text space
            panel = np.ones((panel_height, W, C), dtype=np.uint8) * 255
            panel[text_height:, :, :] = frame
            
            # Convert to BGR for OpenCV text rendering
            panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_color = (0, 0, 0)  # Black text
            
            # Calculate text position (center horizontally)
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = (W - text_size[0]) // 2
            text_y = 25  # Position from top
            
            # Add text to panel
            cv2.putText(panel_bgr, label, (text_x, text_y), font, font_scale, text_color, font_thickness)
            panels.append(panel_bgr)
        
        # Arrange panels in 2x2 grid
        top_row = np.hstack([panels[0], panels[1]])  # Cover Original | Cover Recon
        bottom_row = np.hstack([panels[2], panels[3]])  # Secret Original | Secret Recon
        combined_frame = np.vstack([top_row, bottom_row])
        
        out.write(combined_frame)
    
    out.release()
    print(f"Four-panel comparison video saved to: {output_path}")

def save_video_tensor(tensor, output_path, fps=25):
    """
    Save tensor as video file
    
    Args:
        tensor: Video tensor (B, C, T, H, W) or (C, T, H, W)
        output_path: Output video file path
        fps: Frame rate
    """
    try:
        # Handle batch dimension
        if tensor.dim() == 5:
            tensor = tensor[0]  # Take first batch
        
        # Ensure tensor is on CPU and convert to numpy
        tensor = tensor.detach().cpu().float()
        
        # Clamp values to [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Dimension conversion: (C, T, H, W) -> (T, H, W, C)
        if tensor.shape[0] == 3:  # RGB
            tensor = tensor.permute(1, 2, 3, 0)
        else:
            # If not 3 channels, convert to grayscale then duplicate to RGB
            tensor = tensor.mean(dim=0, keepdim=True)  # (1, T, H, W)
            tensor = tensor.permute(1, 2, 3, 0)  # (T, H, W, 1)
            tensor = tensor.repeat(1, 1, 1, 3)  # (T, H, W, 3)
        
        T, H, W, C = tensor.shape
        
        # Convert to numpy and scale to [0, 255]
        frames = (tensor.numpy() * 255).astype(np.uint8)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set video encoder
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        
        for t in range(T):
            frame = frames[t]
            # OpenCV uses BGR format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving video: {e}")
        # Try simple save method
        try:
            # Simplified save: only save first frame as image
            if tensor.dim() >= 3:
                frame = tensor[:, 0] if tensor.dim() == 4 else tensor[0]  # Take first frame
                if frame.dim() == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0)  # (H, W, C)
                frame = torch.clamp(frame, 0, 1)
                frame_np = (frame.detach().cpu().numpy() * 255).astype(np.uint8)
                
                # Save as image
                img_path = str(output_path).replace('.mp4', '_frame0.png')
                cv2.imwrite(img_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                print(f"Saved first frame to: {img_path}")
        except Exception as e2:
            print(f"Fallback save method also failed: {e2}")


def get_video_files(directory, extensions=None):
    """
    Get all video files in directory
    
    Args:
        directory: Video file directory
        extensions: List of supported file extensions
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.pt', '.pth']
    
    video_files = []
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return video_files
    
    for ext in extensions:
        video_files.extend(list(directory.glob(f'*{ext}')))
        video_files.extend(list(directory.glob(f'*{ext.upper()}')))
    
    return sorted(video_files)


def video_tensor_to_frames(tensor):
    """
    Convert video tensor to frame list
    
    Args:
        tensor: Video tensor (C, T, H, W) or (B, C, T, H, W)
        
    Returns:
        List of frames, each element is (H, W, C) numpy array
    """
    if tensor.dim() == 5:
        tensor = tensor[0]  # Take first batch
    
    tensor = tensor.detach().cpu().float()
    tensor = torch.clamp(tensor, 0, 1)
    
    C, T, H, W = tensor.shape
    frames = []
    
    for t in range(T):
        frame = tensor[:, t, :, :]  # (C, H, W)
        if C == 3:
            frame = frame.permute(1, 2, 0)  # (H, W, C)
        else:
            frame = frame.mean(dim=0, keepdim=True).permute(1, 2, 0)  # Convert to grayscale
            frame = frame.repeat(1, 1, 3)  # Duplicate to RGB
        
        frame_np = (frame.numpy() * 255).astype(np.uint8)
        frames.append(frame_np)
    
    return frames


def frames_to_video_tensor(frames):
    """
    Convert frame list to video tensor
    
    Args:
        frames: List of frames, each element is (H, W, C) numpy array
        
    Returns:
        Video tensor (C, T, H, W)
    """
    if not frames:
        return torch.empty(0)
    
    # Convert to tensors
    frame_tensors = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
        
        if frame.dim() == 3:  # (H, W, C)
            frame = frame.permute(2, 0, 1)  # (C, H, W)
        
        frame = frame.float() / 255.0  # Normalize to [0, 1]
        frame_tensors.append(frame)
    
    # Stack as video tensor
    video_tensor = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)
    
    return video_tensor


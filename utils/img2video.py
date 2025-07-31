import random
from pathlib import Path
import shutil
def split_dataset(source_dir, target_dir, seed=42, train_count=2, test_count=2):
    """
    Split video dataset into training and testing sets.
    
    Args:
        source_dir: Source directory containing video categories
        target_dir: Target directory for organized dataset
        seed: Random seed for reproducible splits
        train_count: Number of videos per category for training
        test_count: Number of videos per category for testing
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    train_dir = target_path / "train"
    test_dir = target_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    processed_categories = 0
    total_train_videos = 0
    total_test_videos = 0
    
    print(f"Splitting dataset from {source_dir} to {target_dir}")
    print(f"Target: {train_count} training videos, {test_count} test videos per category")
    
    # Process each category directory
    for category_dir in source_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        print(f"\nProcessing category: {category_dir.name}")
        
        # Find all video files in category
        video_files = []
        for ext in ['*.avi', '*.mp4', '*.mov', '*.mkv']:
            video_files.extend(list(category_dir.glob(ext)))
        
        if len(video_files) < train_count + test_count:
            print(f"Warning: Not enough videos in {category_dir.name} "
                  f"(found {len(video_files)}, need {train_count + test_count})")
            continue
        
        # Randomly select videos
        random.seed(seed)
        random.shuffle(video_files)
        
        train_videos = video_files[:train_count]
        test_videos = video_files[train_count:train_count + test_count]
        
        # Copy training videos
        for video in train_videos:
            dest_path = train_dir / f"{category_dir.name}_{video.name}"
            shutil.copy2(video, dest_path)
            total_train_videos += 1
        
        # Copy test videos
        for video in test_videos:
            dest_path = test_dir / f"{category_dir.name}_{video.name}"
            shutil.copy2(video, dest_path)
            total_test_videos += 1
        
        processed_categories += 1
        print(f"  Copied {len(train_videos)} training videos")
        print(f"  Copied {len(test_videos)} test videos")
    
    print(f"\nDataset split completed!")
    print(f"Processed {processed_categories} categories")
    print(f"Total training videos: {total_train_videos}")
    print(f"Total test videos: {total_test_videos}")
    print(f"Training data saved to: {train_dir}")
    print(f"Test data saved to: {test_dir}")

def imgs2video(imgs_path, video_path, resize_size, fps=25):
    """
    Convert a sequence of images to a video file.
    
    Args:
        imgs_path: Path to the directory containing images
        video_path: Path to save the output video
        fps: Frames per second for the output video
    """
    import cv2
    imgs_path = Path(imgs_path)
    video_path = Path(video_path)
    if not imgs_path.is_dir():
        raise ValueError(f"Images path {imgs_path} is not a directory")
    if not video_path.parent.is_dir():
        raise ValueError(f"Video path {video_path.parent} does not exist")
    # Get all image files in the directory
    img_files = sorted(imgs_path.glob('*.*'), key=lambda x: x.stem)
    if not img_files:
        raise ValueError(f"No images found in {imgs_path}")
    # Read the first image to get dimensions
    first_img = cv2.imread(str(img_files[0]))
    if first_img is None:
        raise ValueError(f"Failed to read the first image: {img_files[0]}")
    height, width, _ = first_img.shape
    if resize_size:
        height, width = resize_size
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    # Write each image to the video
    for img_file in img_files:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Failed to read image {img_file}, skipping")
            continue
        if resize_size:
            img = cv2.resize(img, (width, height))
        video_writer.write(img)
    # Release the VideoWriter
    video_writer.release()
    print(f"Video saved to {video_path} with {len(img_files)} frames at {fps} FPS")
    
    return video_path
    
    
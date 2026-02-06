"""
ComfyRaw - Video processing utilities using OpenCV
"""

import cv2
import numpy as np
from typing import Iterator, List, Optional, Tuple
from .types import CVImage, CVVideo, ensure_float32, ensure_uint8


class VideoProcessor:
    """Static methods for video processing operations"""

    @staticmethod
    def load_frames(path: str, max_frames: Optional[int] = None) -> CVVideo:
        """Load video frames from file"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB and normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from video: {path}")

        return frames

    @staticmethod
    def load_as_batch(path: str, max_frames: Optional[int] = None) -> CVImage:
        """Load video frames as a batch array"""
        frames = VideoProcessor.load_frames(path, max_frames)
        return np.stack(frames, axis=0)

    @staticmethod
    def save(frames: List[np.ndarray], path: str, fps: float = 30.0,
             codec: str = "mp4v") -> None:
        """Save frames as video file"""
        if not frames:
            raise ValueError("No frames to save")

        # Get frame dimensions
        if frames[0].ndim == 4:
            # Batch format (B, H, W, C)
            frames = [frames[0][i] for i in range(frames[0].shape[0])]

        h, w = frames[0].shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))

        for frame in frames:
            # Convert to uint8 BGR
            frame_uint8 = ensure_uint8(frame)
            if frame_uint8.ndim == 3 and frame_uint8.shape[2] >= 3:
                frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            out.write(frame_uint8)

        out.release()

    @staticmethod
    def get_info(path: str) -> dict:
        """Get video information"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

        cap.release()
        return info

    @staticmethod
    def extract_frame(path: str, frame_number: int) -> CVImage:
        """Extract a single frame from video"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read frame {frame_number}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0

        return np.expand_dims(frame, 0)

    @staticmethod
    def frames_iterator(path: str) -> Iterator[np.ndarray]:
        """Iterate over video frames (memory efficient)"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            yield frame

        cap.release()

    @staticmethod
    def batch_to_frames(batch: CVImage) -> CVVideo:
        """Convert batch array to list of frames"""
        return [batch[i] for i in range(batch.shape[0])]

    @staticmethod
    def frames_to_batch(frames: CVVideo) -> CVImage:
        """Convert list of frames to batch array"""
        return np.stack(frames, axis=0)

    @staticmethod
    def resize_video(frames: CVVideo, width: int, height: int) -> CVVideo:
        """Resize all video frames"""
        resized = []
        for frame in frames:
            frame_uint8 = ensure_uint8(frame)
            resized_frame = cv2.resize(frame_uint8, (width, height))
            resized.append(ensure_float32(resized_frame))
        return resized

    @staticmethod
    def trim(frames: CVVideo, start: int, end: Optional[int] = None) -> CVVideo:
        """Trim video to specified frame range"""
        if end is None:
            end = len(frames)
        return frames[start:end]

    @staticmethod
    def concatenate(videos: List[CVVideo]) -> CVVideo:
        """Concatenate multiple videos"""
        result = []
        for video in videos:
            result.extend(video)
        return result

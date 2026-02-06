"""
ComfyRaw - Unit tests for comfy_cv.video module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
import tempfile
from comfy_cv.video import VideoProcessor


class TestVideoLoadSave:
    """Tests for video loading and saving"""

    @pytest.fixture
    def sample_frames(self):
        """Create sample video frames"""
        return [np.random.rand(128, 128, 3).astype(np.float32) for _ in range(10)]

    @pytest.fixture
    def temp_video_path(self, sample_frames, temp_dir):
        """Create a temporary video file"""
        path = os.path.join(temp_dir, "test_video.mp4")
        VideoProcessor.save(sample_frames, path, fps=30.0)
        return path

    def test_save_video(self, sample_frames, temp_dir):
        """Save frames as video"""
        path = os.path.join(temp_dir, "output.mp4")
        VideoProcessor.save(sample_frames, path, fps=30.0)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_save_empty_frames(self, temp_dir):
        """Saving empty frames should raise ValueError"""
        path = os.path.join(temp_dir, "empty.mp4")
        with pytest.raises(ValueError):
            VideoProcessor.save([], path)

    def test_load_frames(self, temp_video_path):
        """Load video frames"""
        frames = VideoProcessor.load_frames(temp_video_path)

        assert isinstance(frames, list)
        assert len(frames) > 0
        assert frames[0].dtype == np.float32
        assert frames[0].ndim == 3

    def test_load_frames_max_frames(self, temp_video_path):
        """Load limited number of frames"""
        frames = VideoProcessor.load_frames(temp_video_path, max_frames=5)

        assert len(frames) <= 5

    def test_load_as_batch(self, temp_video_path):
        """Load video as batch array"""
        batch = VideoProcessor.load_as_batch(temp_video_path)

        assert batch.ndim == 4
        assert batch.dtype == np.float32

    def test_load_nonexistent(self):
        """Loading non-existent video should raise ValueError"""
        with pytest.raises(ValueError):
            VideoProcessor.load_frames("/nonexistent/video.mp4")


class TestVideoInfo:
    """Tests for VideoProcessor.get_info"""

    @pytest.fixture
    def sample_frames(self):
        return [np.random.rand(128, 128, 3).astype(np.float32) for _ in range(30)]

    @pytest.fixture
    def temp_video_path(self, sample_frames, temp_dir):
        path = os.path.join(temp_dir, "info_test.mp4")
        VideoProcessor.save(sample_frames, path, fps=30.0)
        return path

    def test_get_info(self, temp_video_path):
        """Get video information"""
        info = VideoProcessor.get_info(temp_video_path)

        assert "width" in info
        assert "height" in info
        assert "fps" in info
        assert "frame_count" in info
        assert "duration" in info

        assert info["width"] == 128
        assert info["height"] == 128
        assert info["fps"] > 0


class TestExtractFrame:
    """Tests for VideoProcessor.extract_frame"""

    @pytest.fixture
    def sample_frames(self):
        return [np.random.rand(128, 128, 3).astype(np.float32) for _ in range(10)]

    @pytest.fixture
    def temp_video_path(self, sample_frames, temp_dir):
        path = os.path.join(temp_dir, "extract_test.mp4")
        VideoProcessor.save(sample_frames, path, fps=30.0)
        return path

    def test_extract_first_frame(self, temp_video_path):
        """Extract first frame"""
        frame = VideoProcessor.extract_frame(temp_video_path, 0)

        assert frame.ndim == 4
        assert frame.shape[0] == 1
        assert frame.dtype == np.float32

    def test_extract_middle_frame(self, temp_video_path):
        """Extract middle frame"""
        frame = VideoProcessor.extract_frame(temp_video_path, 5)

        assert frame.ndim == 4


class TestFramesIterator:
    """Tests for VideoProcessor.frames_iterator"""

    @pytest.fixture
    def sample_frames(self):
        return [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(5)]

    @pytest.fixture
    def temp_video_path(self, sample_frames, temp_dir):
        path = os.path.join(temp_dir, "iter_test.mp4")
        VideoProcessor.save(sample_frames, path, fps=30.0)
        return path

    def test_iterator(self, temp_video_path):
        """Iterate over frames"""
        count = 0
        for frame in VideoProcessor.frames_iterator(temp_video_path):
            assert frame.ndim == 3
            assert frame.dtype == np.float32
            count += 1

        assert count > 0


class TestBatchFrameConversion:
    """Tests for batch/frame conversion"""

    def test_batch_to_frames(self):
        """Convert batch to frames list"""
        batch = np.random.rand(5, 64, 64, 3).astype(np.float32)
        frames = VideoProcessor.batch_to_frames(batch)

        assert len(frames) == 5
        assert frames[0].shape == (64, 64, 3)

    def test_frames_to_batch(self):
        """Convert frames list to batch"""
        frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(5)]
        batch = VideoProcessor.frames_to_batch(frames)

        assert batch.shape == (5, 64, 64, 3)


class TestResizeVideo:
    """Tests for VideoProcessor.resize_video"""

    def test_resize(self):
        """Resize video frames"""
        frames = [np.random.rand(128, 128, 3).astype(np.float32) for _ in range(5)]
        resized = VideoProcessor.resize_video(frames, 64, 64)

        assert len(resized) == 5
        assert resized[0].shape == (64, 64, 3)


class TestTrim:
    """Tests for VideoProcessor.trim"""

    def test_trim_basic(self):
        """Trim video to range"""
        frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(10)]
        trimmed = VideoProcessor.trim(frames, start=2, end=7)

        assert len(trimmed) == 5

    def test_trim_no_end(self):
        """Trim with no end specified"""
        frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(10)]
        trimmed = VideoProcessor.trim(frames, start=5)

        assert len(trimmed) == 5


class TestConcatenate:
    """Tests for VideoProcessor.concatenate"""

    def test_concatenate(self):
        """Concatenate multiple videos"""
        video1 = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(5)]
        video2 = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(3)]
        video3 = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(2)]

        result = VideoProcessor.concatenate([video1, video2, video3])

        assert len(result) == 10

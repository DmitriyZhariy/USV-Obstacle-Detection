"""Tests for ClipLoader"""
import pytest
from pathlib import Path
from usv.auto_annotation.clip_loader import ClipLoader, ClipNotFoundError

FIXTURES = Path(__file__).parent / "fixtures"
CLIP_NAME = "test_left_FIXTURE_0001"


@pytest.fixture
def loader():
    return ClipLoader(clips_dir=FIXTURES)


def test_load_happy_path(loader):
    clip = loader.load(CLIP_NAME)
    assert clip.clip_name == CLIP_NAME
    assert clip.n_frames == 3
    assert clip.keyframe_idx == 1
    assert clip.frame_height == 64
    assert clip.frame_width == 64
    assert len(clip.frames) == 3
    assert clip.original_video == "FIXTURE.avi"


def test_keyframe_is_center(loader):
    clip = loader.load(CLIP_NAME)
    assert clip.keyframe_idx == clip.n_frames // 2


def test_missing_clip_raises(loader):
    with pytest.raises(ClipNotFoundError):
        loader.load("nonexistent_clip_0000")


def test_meta_df_columns(loader):
    clip = loader.load(CLIP_NAME)
    assert set(clip.meta_df.columns) == {
        "local_seq_idx", "original_video", "original_frame_idx"
    }


def test_list_clips(loader):
    clips = loader.list_clips()
    assert CLIP_NAME in clips
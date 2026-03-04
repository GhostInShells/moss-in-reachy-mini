from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class VideoRecordSettings(BaseSettings):
    """Settings for Reachy Mini video recording.

    Only covers `VIDEO_RECORD_*` env vars introduced by this feature.
    """

    # Resolve repo root: src/moss_in_reachy_mini/video/settings.py -> repo_root/.env
    _repo_root = Path(__file__).resolve().parents[3]

    model_config = SettingsConfigDict(
        env_file=str(_repo_root / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    video_record_fps: int = 25
    video_record_mic_enabled: bool = True
    video_record_mic_rate: int = 48000
    video_record_mic_channels: int = 1

    video_record_x264_crf: int = 18
    video_record_x264_preset: str = "fast"
    video_record_audio_bitrate_kbps: int = 256

    video_record_frame_source: str = "raw"  # raw / annotated
    video_record_scale: str = ""
    video_record_max_width: int = 1280
    video_record_max_height: int = 0

    video_record_keep_tmp: bool = False

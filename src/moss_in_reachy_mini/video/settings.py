from __future__ import annotations

from pathlib import Path

from pydantic import Field
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

    video_record_fps: int = Field(description="Frames per second", default=25)
    video_record_mic_enabled: bool = Field(description="Whether microphone recording is enabled", default=True)
    video_record_mic_rate: int = Field(description="Microphone sample rate in Hz", default=48000)
    video_record_mic_channels: int = Field(description="Number of microphone channels", default=1)

    video_record_x264_crf: int = Field(description="x264 Constant Rate Factor (lower = higher quality)", default=18)
    video_record_x264_preset: str = Field(
        description="x264 encoding preset (speed vs compression trade-off)", default="fast"
    )
    video_record_audio_bitrate_kbps: int = Field(description="Audio bitrate in kbps", default=256)

    video_record_frame_source: str = Field(description="Source of video frames: 'raw' or 'annotated'", default="raw")
    video_record_scale: str = Field(description="Video scaling factor (empty string for no scaling)", default="")
    video_record_max_width: int = Field(description="Maximum video width in pixels (0 for no limit)", default=1280)
    video_record_max_height: int = Field(description="Maximum video height in pixels (0 for no limit)", default=0)

    video_record_keep_tmp: bool = Field(description="Whether to keep temporary files after recording", default=False)

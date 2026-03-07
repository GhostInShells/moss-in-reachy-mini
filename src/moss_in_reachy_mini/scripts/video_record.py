import argparse
import logging
import os
import sys
import time

from ghoshell_common.contracts import LocalWorkspace
from reachy_mini import ReachyMini

from moss_in_reachy_mini.camera.camera_worker import CameraWorker
from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer
from moss_in_reachy_mini.camera.frame_hub import FrameHub
from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorker
from moss_in_reachy_mini.video.settings import VideoRecordSettings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record Reachy Mini camera + mic + robot output audio to mp4 (ffmpeg).",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note appended to filename.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Auto-stop after N seconds (0 = wait for Enter/Ctrl+C).",
    )
    args = parser.parse_args(argv)

    ws_dir = os.path.join(os.path.dirname(__file__), "..", ".workspace")
    ws = LocalWorkspace(os.path.abspath(ws_dir))

    cfg = VideoRecordSettings()

    mini = ReachyMini()
    frame_hub = FrameHub(mini, fps=max(cfg.video_record_fps, 25))

    # Annotated mode needs CameraWorker; raw mode can record directly from FrameHub.
    frame_source = cfg.video_record_frame_source.strip().lower()
    camera_worker = None
    if frame_source in ("annotated", "anno", "overlay"):
        face_recognizer = FaceRecognizer(known_faces_storage=ws.configs().sub_storage("face_recognizer"))
        camera_worker = CameraWorker(reachy_mini=mini, frame_hub=frame_hub, face_recognizer=face_recognizer)

    storage = ws.runtime().sub_storage("video_records")
    recorder = VideoRecorderWorker(
        mini=mini,
        frame_hub=frame_hub,
        camera_worker=camera_worker,
        storage=storage,
        fps=cfg.video_record_fps,
        mic_enabled=cfg.video_record_mic_enabled,
        mic_rate=cfg.video_record_mic_rate,
        mic_channels=cfg.video_record_mic_channels,
        x264_crf=cfg.video_record_x264_crf,
        x264_preset=cfg.video_record_x264_preset,
        audio_bitrate_kbps=cfg.video_record_audio_bitrate_kbps,
        frame_source=frame_source,
        scale=cfg.video_record_scale,
        max_width=cfg.video_record_max_width,
        max_height=cfg.video_record_max_height,
        keep_tmp=cfg.video_record_keep_tmp,
    )

    try:
        with mini:
            frame_hub.start()
            if camera_worker is not None:
                camera_worker.start()

            file_name = recorder.start_recording(note=args.note)
            out_dir = storage.abspath() if hasattr(storage, "abspath") else "(unknown)"
            print(f"Recording started: {file_name}")
            print(f"Output dir: {out_dir}")

            if args.duration and args.duration > 0:
                time.sleep(args.duration)
            else:
                print("Press Enter to stop recording (or Ctrl+C).")
                try:
                    sys.stdin.readline()
                except KeyboardInterrupt:
                    pass

            try:
                saved = recorder.stop_recording()
                print(f"Recording saved: {saved}")
            except Exception as e:
                print(f"Failed to stop recording: {e}")
    finally:
        try:
            recorder.stop()
        except Exception:
            logger.warning("Failed to stop recorder: %s", e)
        try:
            if camera_worker is not None:
                camera_worker.stop()
        except Exception:
            logger.warning("Failed to stop camera worker: %s", e)
        try:
            frame_hub.stop()
        except Exception:
            logger.warning("Failed to stop frame hub: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

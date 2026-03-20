"""Minimal CLI to test Reachy Mini audio playback.

This script intentionally bypasses the MOSS/agent stack.

Examples:
  - Play local file:
      python -m moss_in_reachy_mini.scripts.play_sound_cli /abs/path/to/song.wav

  - Play workspace asset by name (resolved under assets/audio):
      python -m moss_in_reachy_mini.scripts.play_sound_cli "不能说的秘密"

Troubleshooting (macOS dylib conflicts: cv2 vs av):
  - Disable PyAV and force soundfile-only decoding:
      REACHY_MINI_AUDIO_DISABLE_PYAV=1 python -m moss_in_reachy_mini.scripts.play_sound_cli song.wav
"""

import argparse
import asyncio
import logging
from pathlib import Path

from ghoshell_moss_contrib.example_ws import workspace_container
from reachy_mini import ReachyMini

from moss_in_reachy_mini.components.sound import Sound, SoundProvider


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="Test Reachy Mini audio playback")
    parser.add_argument("sound_file", help="Absolute path, URL, or asset name under assets/audio")
    parser.add_argument(
        "--ws",
        default=str(Path(__file__).resolve().parents[1].joinpath(".workspace")),
        help="Workspace directory (default: src/moss_in_reachy_mini/.workspace)",
    )
    parser.add_argument(
        "--backend",
        default="default",
        help="ReachyMini media backend (default: default)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="If >0, stop after N seconds (useful for quick tests)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    ws_dir = Path(args.ws).resolve()
    with workspace_container(ws_dir) as container:
        # Bind the robot.
        mini = ReachyMini(media_backend=args.backend)
        container.set(ReachyMini, mini)
        container.register(SoundProvider())

        sound = container.force_fetch(Sound)

        # Ensure media subsystem is open.
        mini.__enter__()
        try:
            print("Playing:", args.sound_file)
            await sound.play_sound(args.sound_file)

            # Poll status until idle/error (best-effort).
            if args.seconds and args.seconds > 0:
                await asyncio.sleep(float(args.seconds))
                await sound.stop_sound()
            for _ in range(600):
                st = await sound.sound_status()
                # print(st)
                if "state=idle" in st or "state=error" in st or "state=stopped" in st:
                    break
                await asyncio.sleep(0.5)
        finally:
            mini.__exit__(None, None, None)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()

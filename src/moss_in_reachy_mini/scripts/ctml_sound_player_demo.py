"""Run Reachy Mini audio playback control via CTML (no MOSS/agent stack).

This script is intentionally lightweight:
- It does NOT depend on `MossInReachyMini` nor any agent/state machine.
- It uses `ghoshell_moss`'s `CTMLShell` to execute command tags.
- It registers a dedicated CTML channel `reachy_mini.sound`.

Examples:
  - Generate a default CTML sequence:
      python -m moss_in_reachy_mini.scripts.ctml_sound_player_demo \
        "不能说的秘密" --pause-after 1.5 --resume-after 1.0 --stop-after 3

  - Provide a custom CTML file:
      python -m moss_in_reachy_mini.scripts.ctml_sound_player_demo \
        --ctml-file demo.ctml

CTML snippet reference (custom file):

  <reachy_mini:play_sound sound_file="不能说的秘密"/>
  <sleep duration="1.5"/>
  <reachy_mini:pause_sound/>
  <sleep duration="1.0"/>
  <reachy_mini:resume_sound/>
  <sleep duration="3"/>
  <reachy_mini:stop_sound/>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import traceback
from pathlib import Path

from ghoshell_container import IoCContainer
from ghoshell_moss import MOSSShell, new_chan
from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss_contrib.example_ws import workspace_container
from reachy_mini import ReachyMini

from moss_in_reachy_mini.audio.mixer import AudioMixerProvider
from moss_in_reachy_mini.components.sound import Sound, SoundProvider

logging.getLogger("mosshell").propagate = False


def _register_providers(
    container: IoCContainer,
    *,
    robot_name: str,
    connection_mode: str,
    spawn_daemon: bool,
    use_sim: bool,
    timeout: float,
) -> None:
    container.set(
        ReachyMini,
        ReachyMini(
            robot_name=robot_name,
            connection_mode=connection_mode,  # type: ignore[arg-type]
            spawn_daemon=spawn_daemon,
            use_sim=use_sim,
            timeout=timeout,
        ),
    )

    # Mixer first so SoundProvider can route into it.
    container.register(AudioMixerProvider())
    container.register(SoundProvider())


def _build_sound_channel(sound: Sound):
    chan = new_chan(
        "reachy_mini",
        description="Reachy Mini sound playback control (CTML)",
        blocking=True,
    )

    @chan.build.command(
        name="play_sound",
        doc=(
            "Play a sound (local file or URL). If it's a relative path, it is resolved under assets/audio/. "
            "Playback is asynchronous; use pause/resume/stop commands to control it."
        ),
    )
    async def play_sound(sound_file: str) -> None:
        await sound.play_sound(sound_file)

    @chan.build.command(name="pause_sound", doc="Pause current sound playback")
    async def pause_sound() -> None:
        await sound.pause_sound()

    @chan.build.command(name="resume_sound", doc="Resume paused sound playback")
    async def resume_sound() -> None:
        await sound.resume_sound()

    @chan.build.command(name="stop_sound", doc="Stop current sound playback")
    async def stop_sound() -> None:
        await sound.stop_sound()

    @chan.build.command(name="sound_status", doc="Get sound playback status")
    async def sound_status() -> str:
        return await sound.sound_status()

    return chan


def _default_ctml(*, sound_file: str, pause_after: float, resume_after: float, stop_after: float) -> str:
    # Use CTML primitives (<sleep/>) from the CTML main channel.
    chunks: list[str] = [f'<reachy_mini:play_sound sound_file="{sound_file}"/>']

    if pause_after > 0:
        chunks.append(f'<sleep duration="{float(pause_after)}"/>')
        chunks.append("<reachy_mini:pause_sound/>")
    if resume_after > 0:
        chunks.append(f'<sleep duration="{float(resume_after)}"/>')
        chunks.append("<reachy_mini:resume_sound/>")
    if stop_after > 0:
        chunks.append(f'<sleep duration="{float(stop_after)}"/>')
        chunks.append("<reachy_mini:stop_sound/>")

    # Always print a final status.
    chunks.append("<reachy_mini:sound_status/>")
    return "\n".join(chunks)


async def _run_ctml(shell: MOSSShell, ctml: str) -> dict[str, str]:  # type: ignore[no-untyped-def]
    interpreter = await shell.interpreter(kind="clear")
    async with interpreter:
        interpreter.feed(ctml)
        interpreter.commit()
        tasks = await interpreter.wait_tasks(throw=False)

    if not tasks:
        exp = interpreter.exception()
        if exp is not None:
            raise RuntimeError(f"CTML interpreter failed: {type(exp).__name__}: {exp}")

    results: dict[str, str] = {}
    for cid, task in tasks.items():
        try:
            res = task.result(throw=False)
        except Exception as e:
            res = f"{type(e).__name__}: {e}"
        results[cid] = "" if res is None else str(res)
    return results


async def _amain(
    *,
    sound_file: str | None,
    ctml_file: str | None,
    ctml: str | None,
    pause_after: float,
    resume_after: float,
    stop_after: float,
    ws_dir: Path,
    robot_name: str,
    connection_mode: str,
    spawn_daemon: bool,
    use_sim: bool,
    timeout: float,
) -> int:
    if not ws_dir.exists():
        raise FileNotFoundError(f"workspace dir not found: {ws_dir}")

    if ctml_file:
        ctml_text = Path(ctml_file).read_text(encoding="utf-8")
    elif ctml:
        ctml_text = ctml
    else:
        if not sound_file:
            raise ValueError("sound_file is required when --ctml/--ctml-file is not provided")
        ctml_text = _default_ctml(
            sound_file=sound_file,
            pause_after=pause_after,
            resume_after=resume_after,
            stop_after=stop_after,
        )

    with workspace_container(ws_dir) as container:
        try:
            _register_providers(
                container,
                robot_name=robot_name,
                connection_mode=connection_mode,
                spawn_daemon=spawn_daemon,
                use_sim=use_sim,
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize ReachyMini client. "
                f"Root cause: {type(e).__name__}: {e}. "
                "Make sure the Reachy Mini daemon is running and accessible."
            ) from e

        mini = container.force_fetch(ReachyMini)
        sound = container.force_fetch(Sound)

        # Register CTML channel commands.
        shell = new_ctml_shell(container=container)
        shell.main_channel.import_channels(_build_sound_channel(sound))

        with mini:
            async with shell:
                # Ensure the channel runtime is connected before parsing tags.
                # Otherwise CTML may treat commands as unavailable/unknown.
                await shell.wait_connected("reachy_mini")
                results = await _run_ctml(shell, ctml_text)
                print("====> CTML\n" + ctml_text)
                for k, v in results.items():
                    if v:
                        print("  result:", k, "->", v)
                    else:
                        print("  result:", k)

    return 0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_ws = repo_root / ".workspace"

    parser = argparse.ArgumentParser(description="Run play/pause/resume/stop_sound via CTMLShell")
    parser.add_argument(
        "sound_file",
        nargs="?",
        default=None,
        help="Absolute path, URL, or asset name under assets/audio (used when CTML isn't provided)",
    )
    parser.add_argument(
        "--ctml",
        default=None,
        help="Inline CTML string to run",
    )
    parser.add_argument(
        "--ctml-file",
        default=None,
        help="Path to a CTML file to run",
    )
    parser.add_argument(
        "--pause-after",
        type=float,
        default=0.0,
        help="If >0, pause after N seconds (only for auto-generated CTML)",
    )
    parser.add_argument(
        "--resume-after",
        type=float,
        default=0.0,
        help="If >0, resume after N seconds since pause (only for auto-generated CTML)",
    )
    parser.add_argument(
        "--stop-after",
        type=float,
        default=0.0,
        help="If >0, stop after N seconds since resume/play (only for auto-generated CTML)",
    )

    parser.add_argument(
        "--ws",
        default=str(default_ws),
        help=f"Workspace directory (default: {default_ws})",
    )

    parser.add_argument("--robot-name", default="reachy_mini")
    parser.add_argument(
        "--connection-mode",
        default="auto",
        choices=["auto", "localhost_only", "network"],
        help="ReachyMini connection mode",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="ReachyMini connection timeout in seconds",
    )
    parser.add_argument(
        "--spawn-daemon",
        action="store_true",
        help="Try spawning the Reachy Mini daemon",
    )
    parser.add_argument(
        "--use-sim",
        action="store_true",
        help="Use simulated ReachyMini (no real robot required)",
    )

    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print full traceback on error.",
    )
    args = parser.parse_args()
    try:
        code = asyncio.run(
            _amain(
                sound_file=args.sound_file,
                ctml_file=args.ctml_file,
                ctml=args.ctml,
                pause_after=float(args.pause_after),
                resume_after=float(args.resume_after),
                stop_after=float(args.stop_after),
                ws_dir=Path(args.ws).resolve(),
                robot_name=args.robot_name,
                connection_mode=args.connection_mode,
                spawn_daemon=bool(args.spawn_daemon),
                use_sim=bool(args.use_sim),
                timeout=float(args.timeout),
            )
        )
    except Exception as e:
        print(f"ERROR: {e}")
        if args.traceback:
            traceback.print_exc()
        raise SystemExit(1) from e
    raise SystemExit(code)


if __name__ == "__main__":
    main()

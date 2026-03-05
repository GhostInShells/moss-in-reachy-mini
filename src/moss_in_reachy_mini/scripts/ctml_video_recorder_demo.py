import argparse
import asyncio
import logging
import traceback
from pathlib import Path

from ghoshell_container import IoCContainer
from ghoshell_moss import MOSSShell, new_shell
from ghoshell_moss_contrib.example_ws import workspace_container
from reachy_mini import ReachyMini

from framework.abcd.agent import EventBus
from framework.agent.eventbus import QueueEventBus
from moss_in_reachy_mini.camera.camera_worker import CameraWorkerProvider
from moss_in_reachy_mini.camera.frame_hub import FrameHub, FrameHubProvider
from moss_in_reachy_mini.components.antennas import AntennasProvider
from moss_in_reachy_mini.components.body import BodyProvider
from moss_in_reachy_mini.components.head import HeadProvider
from moss_in_reachy_mini.components.head_tracker import HeadTrackerProvider
from moss_in_reachy_mini.components.vision import VisionProvider
from moss_in_reachy_mini.moss import MossInReachyMini, MossInReachyMiniProvider
from moss_in_reachy_mini.state import AsleepStateProvider, BoringStateProvider, WakenStateProvider
from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorkerProvider

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
    container.register(FrameHubProvider())
    container.register(CameraWorkerProvider())
    container.register(VideoRecorderWorkerProvider())

    container.set(EventBus, QueueEventBus())
    container.register(BodyProvider())
    container.register(HeadProvider())
    container.register(AntennasProvider())
    container.register(VisionProvider())
    container.register(HeadTrackerProvider())

    container.register(AsleepStateProvider())
    container.register(WakenStateProvider())
    container.register(BoringStateProvider())
    container.register(MossInReachyMiniProvider())


async def _run_ctml(shell: MOSSShell, ctml: str) -> dict[str, str]:
    async with shell.interpreter_in_ctx() as interpreter:
        interpreter.feed(ctml)
        interpreter.commit()
        return await interpreter.results()


async def _amain(
    *,
    note: str,
    sleep_s: float,
    robot_name: str,
    connection_mode: str,
    spawn_daemon: bool,
    use_sim: bool,
    timeout: float,
) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ws_dir = repo_root / ".workspace"

    if not ws_dir.exists():
        raise FileNotFoundError(f"workspace dir not found: {ws_dir}")

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
                "Make sure the Reachy Mini daemon is running and accessible, "
            ) from e

        mini = container.force_fetch(ReachyMini)
        with mini:
            try:
                # Ensure the single camera capture loop is running.
                frame_hub = container.force_fetch(FrameHub)
                frame_hub.start()
                shell = new_shell(container=container)
                await shell.start()
                moss = container.force_fetch(MossInReachyMini)
                async with moss:
                    shell.main_channel.import_channels(moss.as_channel())

                # Switch to waken state, then recording.
                await _run_ctml(shell, '<reachy_mini:switch_state state_name="waken" />')
                start_ctml = f'<reachy_mini.video_recorder:start_recording note="{note}"/>'
                stop_ctml = "<reachy_mini.video_recorder:stop_recording/>"

                start_results = await _run_ctml(shell, start_ctml)
                print("====> CTML:", start_ctml)
                for k, v in start_results.items():
                    print("  result:", k, "->", v)

                await _run_ctml(shell, '<reachy_mini:switch_state state_name="boring" />')

                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                stop_results = await _run_ctml(shell, stop_ctml)
                print("====> CTML:", stop_ctml)
                for k, v in stop_results.items():
                    print("  result:", k, "->", v)
            finally:
                await shell.close()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run video recorder start/stop via CTMLInterpreter.",
    )
    parser.add_argument(
        "--note",
        default="test",
        help="note passed to start_recording",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="seconds to wait between start and stop",
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
                note=args.note,
                sleep_s=args.sleep,
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

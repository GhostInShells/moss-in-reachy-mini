"""Tests for EnrollingState face enrollment flow."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from framework.abcd.agent_event import CTMLAgentEvent
from framework.agent.eventbus import QueueEventBus
from moss_in_reachy_mini.camera.model import CameraFrame, KnownFace, Position
from moss_in_reachy_mini.state.enrolling import EnrollingState


# ---------------------------------------------------------------------------
# Lightweight fakes – avoid heavy dependencies (ReachyMini, InsightFace …)
# ---------------------------------------------------------------------------


class FakeMini:
    """Minimal stand-in for ReachyMini."""

    def __init__(self):
        self.motors_enabled = False

    def enable_motors(self):
        self.motors_enabled = True


class FakeCameraWorker:
    """Controllable camera worker that returns pre-configured frames."""

    def __init__(self):
        self._frames: List[CameraFrame] = []
        self._index = 0

    def enqueue_frames(self, *frames: CameraFrame):
        self._frames.extend(frames)

    def get_latest_frame(self) -> CameraFrame:
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return frame
        # Default: empty frame (no face)
        return CameraFrame.new()

    def reset(self):
        self._frames.clear()
        self._index = 0


class FakeFaceRecognizer:
    """Minimal face recognizer that can be pre-loaded with known faces."""

    def __init__(self):
        self.known_faces: Dict[str, KnownFace] = {}
        self._positions_to_return: List[List[Position]] = []
        self._pos_index = 0

    def enqueue_positions(self, *position_lists: List[Position]):
        self._positions_to_return.extend(position_lists)

    def get_face_positions(self, image) -> List[Position]:
        if self._pos_index < len(self._positions_to_return):
            positions = self._positions_to_return[self._pos_index]
            self._pos_index += 1
            return positions
        return []


class FakeStorage:
    """Minimal Storage fake with path attribute."""

    def __init__(self, tmp_path: Path):
        self._path = tmp_path

    @property
    def path(self):
        return str(self._path)

    def abspath(self):
        return str(self._path)


class FakeSubStorage:
    def __init__(self, base: Path, name: str):
        self._base = base / name
        self._base.mkdir(parents=True, exist_ok=True)

    @property
    def path(self):
        return str(self._base)

    def sub_storage(self, name: str):
        return FakeSubStorage(self._base, name)

    def abspath(self):
        return str(self._base)


class FakeRuntime:
    def __init__(self, tmp_path: Path):
        self._base = tmp_path / "runtime"
        self._base.mkdir(parents=True, exist_ok=True)

    def sub_storage(self, name: str):
        return FakeSubStorage(self._base, name)


class FakeWorkspace:
    def __init__(self, tmp_path: Path):
        self._tmp_path = tmp_path

    def runtime(self):
        return FakeRuntime(self._tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_with_face(name: str = None, is_recognized: bool = False) -> CameraFrame:
    """Create a CameraFrame that contains a detected face."""
    position = Position(
        track_id=0,
        bbox=np.array([10, 10, 100, 100], dtype=np.float32),
        center=np.array([0.0, 0.0], dtype=np.float32),
        name=name,
        is_recognized=is_recognized,
    )
    # 3-channel 100x100 dummy image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return CameraFrame(
        face_tracking_offsets=[0.0] * 6,
        face_positons=[position],
        track_name="",
        track_lost=False,
        image=image,
    )


def _make_empty_frame() -> CameraFrame:
    """Create a CameraFrame with no faces but with an image."""
    return CameraFrame(
        face_tracking_offsets=[0.0] * 6,
        face_positons=[],
        track_name="",
        track_lost=True,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
    )


async def _drain_events(eventbus: QueueEventBus) -> List[dict]:
    """Drain all events from the eventbus queue, return as list of AgentEvent dicts."""
    events = []
    while not eventbus.queue.empty():
        event = await eventbus.get(timeout=0.1)
        if event is not None:
            events.append(event)
    return events


def _build_state(
    tmp_path: Path,
    camera_worker: FakeCameraWorker = None,
    face_recognizer: FakeFaceRecognizer = None,
    eventbus: QueueEventBus = None,
) -> tuple:
    """Build an EnrollingState with fakes, return (state, eventbus, camera_worker, face_recognizer)."""
    mini = FakeMini()
    cw = camera_worker or FakeCameraWorker()
    fr = face_recognizer or FakeFaceRecognizer()
    eb = eventbus or QueueEventBus()
    ws = FakeWorkspace(tmp_path)

    state = EnrollingState(
        mini=mini,
        camera_worker=cw,
        face_recognizer=fr,
        eventbus=eb,
        workspace=ws,
    )
    # Speed up tests: eliminate real sleeps
    state._pose_wait_seconds = 0
    state._face_detect_attempts = 1
    state._verify_attempts = 1
    return state, eb, cw, fr


def _ctml_texts(events: List[dict]) -> List[str]:
    """Extract ctml text from a list of AgentEvent dicts (only CTML events)."""
    texts = []
    for e in events:
        if e.get("event_type") == "ctml":
            data = e.get("data", {})
            ctml = data.get("ctml", "")
            texts.append(ctml)
    return texts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_target_name_returns_to_waken(tmp_path):
    """No target_name set -> speak error + return to waken."""
    state, eb, _, _ = _build_state(tmp_path)
    state.target_name = None

    await state._start_registration_process()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)
    assert any("未获取到用户名称" in t for t in texts)
    assert any("switch_state" in t and "waken" in t for t in texts)


@pytest.mark.asyncio
async def test_new_user_happy_path(tmp_path):
    """New user: capture -> train -> verify -> success speech + return to waken."""
    cw = FakeCameraWorker()
    fr = FakeFaceRecognizer()
    state, eb, _, _ = _build_state(tmp_path, camera_worker=cw, face_recognizer=fr)
    state.target_name = "Alice"

    # Provide frames with faces for 3 capture steps (front, left, right)
    for _ in range(3):
        cw.enqueue_frames(_make_frame_with_face())

    # Provide frame for verification (_verify_recognition reads from camera_worker)
    cw.enqueue_frames(_make_frame_with_face(name="Alice", is_recognized=True))

    # _verify_recognition calls face_recognizer.get_face_positions(frame.image)
    fr.enqueue_positions([Position(
        track_id=0,
        bbox=np.array([10, 10, 100, 100], dtype=np.float32),
        center=np.array([0.0, 0.0], dtype=np.float32),
        name="Alice",
        is_recognized=True,
    )])

    # Mock train_face to succeed
    with patch.object(state, "_train_face", return_value=True):
        # Mock cv2.imwrite to avoid filesystem writes
        with patch("moss_in_reachy_mini.state.enrolling.cv2.imwrite"):
            await state._start_registration_process()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)

    # Should greet with "开始人脸录入"
    assert any("开始人脸录入" in t for t in texts)
    # Should have success message
    assert any("记住你了" in t for t in texts)
    # Should return to waken
    assert any("switch_state" in t and "waken" in t for t in texts)


@pytest.mark.asyncio
async def test_update_existing_user(tmp_path):
    """Existing user re-enrollment: shows update prompts."""
    cw = FakeCameraWorker()
    fr = FakeFaceRecognizer()
    # Pre-register Alice
    fr.known_faces["Alice"] = KnownFace(
        name="Alice",
        embedding=np.zeros(512, dtype=np.float32),
        sample_count=1,
    )
    state, eb, _, _ = _build_state(tmp_path, camera_worker=cw, face_recognizer=fr)
    state.target_name = "Alice"

    for _ in range(3):
        cw.enqueue_frames(_make_frame_with_face())
    # Verify frame from camera
    cw.enqueue_frames(_make_frame_with_face(name="Alice", is_recognized=True))

    # _verify_recognition calls face_recognizer.get_face_positions(frame.image)
    fr.enqueue_positions([Position(
        track_id=0,
        bbox=np.array([10, 10, 100, 100], dtype=np.float32),
        center=np.array([0.0, 0.0], dtype=np.float32),
        name="Alice",
        is_recognized=True,
    )])

    with patch.object(state, "_train_face", return_value=True):
        with patch("moss_in_reachy_mini.state.enrolling.cv2.imwrite"):
            await state._start_registration_process()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)

    # Should show update prompts instead of new-user prompts
    assert any("已经注册过了" in t for t in texts)
    assert any("人脸数据已更新" in t for t in texts)
    # Should NOT show "记住你了"
    assert not any("记住你了" in t for t in texts)


@pytest.mark.asyncio
async def test_capture_failure_retries_then_fails(tmp_path):
    """Face not detected during capture -> retry once -> fail -> return to waken."""
    cw = FakeCameraWorker()
    state, eb, _, _ = _build_state(tmp_path, camera_worker=cw)
    state.target_name = "Bob"
    state._max_retries = 2
    # No frames with faces enqueued -> _wait_for_face always returns None

    await state._start_registration_process()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)

    # Should retry
    assert any("拍照失败" in t for t in texts)
    # Should eventually fail
    assert any("多次拍照失败" in t for t in texts)
    assert any("switch_state" in t and "waken" in t for t in texts)


@pytest.mark.asyncio
async def test_train_failure_retries_then_fails(tmp_path):
    """Training fails -> retry -> fail -> return to waken."""
    cw = FakeCameraWorker()
    state, eb, _, _ = _build_state(tmp_path, camera_worker=cw)
    state.target_name = "Charlie"
    state._max_retries = 2

    # Provide enough frames for 2 capture rounds (3 per round * 2 = 6)
    for _ in range(6):
        cw.enqueue_frames(_make_frame_with_face())

    with patch.object(state, "_train_face", return_value=False):
        with patch("moss_in_reachy_mini.state.enrolling.cv2.imwrite"):
            await state._start_registration_process()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)

    assert any("学习失败" in t for t in texts)
    assert any("多次学习失败" in t for t in texts)
    assert any("switch_state" in t and "waken" in t for t in texts)


@pytest.mark.asyncio
async def test_verify_failure_retries_then_fails(tmp_path):
    """Verification fails -> retry -> fail -> return to waken."""
    cw = FakeCameraWorker()
    fr = FakeFaceRecognizer()
    state, eb, _, _ = _build_state(tmp_path, camera_worker=cw, face_recognizer=fr)
    state.target_name = "Dave"
    state._max_retries = 2

    # 2 rounds: each needs 3 capture frames + 1 verify frame (no match) = 4 per round
    for _ in range(2):
        for _ in range(3):
            cw.enqueue_frames(_make_frame_with_face())
        # verify frame: has face but wrong name
        cw.enqueue_frames(_make_frame_with_face(name="Unknown", is_recognized=True))

    # verify via get_face_positions returns wrong person
    for _ in range(4):  # 2 rounds * _verify_attempts (but we set to 1 each)
        fr.enqueue_positions([Position(
            track_id=0,
            bbox=np.array([10, 10, 100, 100], dtype=np.float32),
            center=np.array([0.0, 0.0], dtype=np.float32),
            name="NotDave",
            is_recognized=True,
        )])

    with patch.object(state, "_train_face", return_value=True):
        with patch("moss_in_reachy_mini.state.enrolling.cv2.imwrite"):
            await state._start_registration_process()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)

    assert any("还没认出来" in t for t in texts)
    assert any("多次验证未通过" in t for t in texts)
    assert any("switch_state" in t and "waken" in t for t in texts)


@pytest.mark.asyncio
async def test_wait_for_face_returns_none_when_no_face(tmp_path):
    """_wait_for_face returns None when no face is detected."""
    cw = FakeCameraWorker()
    state, _, _, _ = _build_state(tmp_path, camera_worker=cw)
    state._face_detect_attempts = 2

    # Enqueue empty frames
    cw.enqueue_frames(_make_empty_frame(), _make_empty_frame())

    result = await state._wait_for_face()
    assert result is None


@pytest.mark.asyncio
async def test_wait_for_face_returns_image_on_detection(tmp_path):
    """_wait_for_face returns image when a face is detected."""
    cw = FakeCameraWorker()
    state, _, _, _ = _build_state(tmp_path, camera_worker=cw)

    frame_with_face = _make_frame_with_face()
    cw.enqueue_frames(frame_with_face)

    result = await state._wait_for_face()
    assert result is not None
    assert result.shape == (100, 100, 3)


@pytest.mark.asyncio
async def test_verify_recognition_success(tmp_path):
    """_verify_recognition returns True when face_recognizer finds target."""
    cw = FakeCameraWorker()
    fr = FakeFaceRecognizer()
    state, _, _, _ = _build_state(tmp_path, camera_worker=cw, face_recognizer=fr)
    state.target_name = "Eve"

    # Camera returns frame with image
    cw.enqueue_frames(_make_frame_with_face())

    # face_recognizer.get_face_positions returns match
    fr.enqueue_positions([Position(
        track_id=0,
        bbox=np.array([10, 10, 100, 100], dtype=np.float32),
        center=np.array([0.0, 0.0], dtype=np.float32),
        name="Eve",
        is_recognized=True,
    )])

    result = await state._verify_recognition()
    assert result is True


@pytest.mark.asyncio
async def test_verify_recognition_failure(tmp_path):
    """_verify_recognition returns False when target is not recognized."""
    cw = FakeCameraWorker()
    fr = FakeFaceRecognizer()
    state, _, _, _ = _build_state(tmp_path, camera_worker=cw, face_recognizer=fr)
    state.target_name = "Eve"
    state._verify_attempts = 1

    cw.enqueue_frames(_make_frame_with_face())
    fr.enqueue_positions([])  # no positions found

    result = await state._verify_recognition()
    assert result is False


@pytest.mark.asyncio
async def test_speak_emits_ctml_say_event(tmp_path):
    """_speak should emit a CTMLAgentEvent with <say> tag."""
    state, eb, _, _ = _build_state(tmp_path)

    await state._speak("hello world")

    events = await _drain_events(eb)
    texts = _ctml_texts(events)
    assert len(texts) == 1
    assert texts[0] == "<say>hello world</say>"


@pytest.mark.asyncio
async def test_return_to_waken_emits_switch_event(tmp_path):
    """_return_to_waken_state should emit switch_state CTML event."""
    state, eb, _, _ = _build_state(tmp_path)

    await state._return_to_waken_state()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)
    assert len(texts) == 1
    assert 'switch_state' in texts[0]
    assert 'waken' in texts[0]


@pytest.mark.asyncio
async def test_on_self_exit_clears_target_name(tmp_path):
    """on_self_exit should reset target_name to None."""
    state, _, _, _ = _build_state(tmp_path)
    state.target_name = "SomeUser"

    await state.on_self_exit()

    assert state.target_name is None


@pytest.mark.asyncio
async def test_on_self_enter_enables_motors(tmp_path):
    """on_self_enter should enable motors, reset head, and start registration."""
    cw = FakeCameraWorker()
    state, eb, _, _ = _build_state(tmp_path, camera_worker=cw)
    state.target_name = None  # Will trigger early return in _start_registration_process

    await state.on_self_enter()

    assert state.mini.motors_enabled is True
    events = await _drain_events(eb)
    texts = _ctml_texts(events)
    # Should have head_reset event
    assert any("head_reset" in t for t in texts)


@pytest.mark.asyncio
async def test_capture_retry_on_face_not_detected_first_time(tmp_path):
    """Capture retries face detection once if first attempt fails."""
    cw = FakeCameraWorker()
    state, eb, _, _ = _build_state(tmp_path, camera_worker=cw)
    state.target_name = "Frank"
    state._max_retries = 1
    state._face_detect_attempts = 1

    # For step 1 (front): first _wait_for_face returns None, second returns face
    cw.enqueue_frames(_make_empty_frame())  # fail
    cw.enqueue_frames(_make_frame_with_face())  # retry succeeds
    # Steps 2 and 3: succeed immediately
    cw.enqueue_frames(_make_frame_with_face())
    cw.enqueue_frames(_make_frame_with_face())
    # Verification
    cw.enqueue_frames(_make_frame_with_face(name="Frank", is_recognized=True))

    fr = FakeFaceRecognizer()
    fr.enqueue_positions([Position(
        track_id=0,
        bbox=np.array([10, 10, 100, 100], dtype=np.float32),
        center=np.array([0.0, 0.0], dtype=np.float32),
        name="Frank",
        is_recognized=True,
    )])

    state.face_recognizer = fr

    with patch.object(state, "_train_face", return_value=True):
        with patch("moss_in_reachy_mini.state.enrolling.cv2.imwrite"):
            await state._start_registration_process()

    events = await _drain_events(eb)
    texts = _ctml_texts(events)

    # Should have warned about face not detected
    assert any("没有检测到人脸" in t for t in texts)
    # But should succeed overall
    assert any("记住你了" in t for t in texts)


@pytest.mark.asyncio
async def test_out_switchable_is_true():
    """EnrollingState should be interruptible (out_switchable = True)."""
    assert EnrollingState.out_switchable is True


@pytest.mark.asyncio
async def test_state_name():
    """EnrollingState.NAME should be 'enrolling'."""
    assert EnrollingState.NAME == "enrolling"

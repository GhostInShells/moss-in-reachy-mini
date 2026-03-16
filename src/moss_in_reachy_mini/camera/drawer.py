from pathlib import Path

import cv2
import numpy as np
from ghoshell_common.contracts import DefaultFileStorage
from PIL import Image, ImageDraw, ImageFont

from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer
from moss_in_reachy_mini.camera.model import Position

_FONT_PATH = Path(__file__).resolve().parents[1] / ".workspace" / "assets" / "fonts" / "chinese.otf"
_font_cache: ImageFont.FreeTypeFont | None = None


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont | None:
    global _font_cache
    if _font_cache is not None:
        return _font_cache
    if _FONT_PATH.exists():
        _font_cache = ImageFont.truetype(str(_FONT_PATH), size)
    return _font_cache


def _put_text_pil(frame: np.ndarray, text: str, position: tuple, color: tuple) -> np.ndarray:
    """Use PIL to draw Unicode text on an OpenCV frame."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font()
    if font:
        draw.text(position, text, font=font, fill=color)
    else:
        draw.text(position, text, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_detections(frame: np.ndarray, positions: list[Position]) -> np.ndarray:
    if not frame.flags.writeable:
        frame = frame.copy()

    # 绘制结果（只标注已识别的人脸）
    for position in positions:
        if not position.name:
            continue
        x1, y1, x2, y2 = position.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = _put_text_pil(
            frame,
            position.name,
            (x1 + 10, y1 + 4),
            (0, 255, 0),
        )

    return frame


def main():
    # 创建HeadDetector
    face_recognizer = FaceRecognizer(
        known_faces_storage=DefaultFileStorage(dir_="/Users/wangshiqi/projects/ghosts/moss-in-reachy-mini/src/moss_in_reachy_mini/.workspace/configs/face_recognizer"),
        device="cpu",
    )
    face_recognizer.load_known_faces()

    cap = cv2.VideoCapture(0)  # 0 为默认摄像头，也可替换为视频文件路径

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧，获取跟踪结果
        positions = face_recognizer.get_face_positions(frame)

        # 绘制结果
        frame = draw_detections(frame, positions)

        # 显示
        cv2.imshow('Head Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    main()
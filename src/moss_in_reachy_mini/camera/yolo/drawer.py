from typing import List

import cv2
import numpy as np
from ghoshell_common.contracts import DefaultFileStorage
from supervision import Detections, ByteTrack

from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer
from moss_in_reachy_mini.camera.yolo.head_detector import HeadDetector
from moss_in_reachy_mini.camera.yolo.model import Position


def draw_tracks(frame: np.ndarray, detections: Detections) -> np.ndarray:
    if not frame.flags.writeable:
        frame = frame.copy()

    """在图像上绘制边界框和跟踪ID。"""
    if len(detections) == 0:
        return frame

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        track_id = detections.tracker_id[i] if detections.tracker_id is not None else -1

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制ID标签
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_detections(frame: np.ndarray, positions: List[Position]) -> np.ndarray:
    if not frame.flags.writeable:
        frame = frame.copy()

    # 绘制结果
    for position in positions:
        x1, y1, x2, y2 = position.bbox.astype(int)

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制ID标签
        label = f"ID: {position.track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 绘制识别结果
        if position.name:
            cv2.putText(
                frame,
                f"Name: {position.name}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "Unknown",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    return frame


def main():
    # 创建HeadDetector
    face_recognizer = FaceRecognizer(
        known_faces_storage=DefaultFileStorage(dir_="/Users/wangshiqi/projects/ghosts/moss-in-reachy-mini/src/moss_in_reachy_mini/.workspace/configs/face_recognizer")
    )
    face_recognizer.load_known_faces()
    detector = HeadDetector(
        device="cpu",
        face_recognizer=face_recognizer,
        bytetrack=ByteTrack(),
    )

    # 1. 注册已知人脸
    def register_known_face(name: str, image_path: str):
        """注册已知人脸"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_recognizer.add_known_face_from_image(name, img_rgb)

    cap = cv2.VideoCapture(0)  # 0 为默认摄像头，也可替换为视频文件路径

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧，获取跟踪结果
        _, positions = detector.get_head_positions(frame)

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
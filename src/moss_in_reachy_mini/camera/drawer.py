from typing import List

import cv2
import numpy as np
from ghoshell_common.contracts import DefaultFileStorage

from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer
from moss_in_reachy_mini.camera.model import Position


def draw_detections(frame: np.ndarray, positions: List[Position]) -> np.ndarray:
    if not frame.flags.writeable:
        frame = frame.copy()

    # 绘制结果
    for position in positions:
        x1, y1, x2, y2 = position.bbox.astype(int)

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制ID标签
        # label = f"ID: {position.track_id}"
        # cv2.putText(frame, label, (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 绘制识别结果
        if position.name:
            cv2.putText(
                frame,
                position.name,
                (x1+10, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
            )
        # else:
        #     cv2.putText(
        #         frame,
        #         "unknown",
        #         (x1, y1 - 30),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (0, 0, 255),
        #         2
        #     )

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
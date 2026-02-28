import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from ghoshell_common.contracts import FileStorage, Storage
from numpy.typing import NDArray
from supervision import Detections, ByteTrack
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from moss_in_reachy_mini.camera.yolo.model import Position
from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


class HeadDetector:
    """跟踪画面中所有人脸的位置，并识别人脸"""

    def __init__(
            self,
            model_repo: str = "AdamCodd/YOLOv11n-face-detection",
            model_filename: str = "model.pt",
            confidence_threshold: float = 0.3,
            device: str = "cpu",
            # bytetrack
            bytetrack: ByteTrack = None,
            # 人脸识别参数
            face_recognizer: FaceRecognizer = None,
    ) -> None:
        self.logger = logger
        self.confidence_threshold = confidence_threshold

        # 加载 YOLO 人脸检测模型
        try:
            model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
            self.model = YOLO(model_path).to(device)
            logger.info(f"YOLO face detection model loaded from {model_repo}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

        # 初始化跟踪器（ByteTrack）
        self.tracker = bytetrack

        # 初始化人脸识别器
        if face_recognizer:
            self.face_recognizer = face_recognizer

        # 跟踪ID到人名的映射缓存
        self.track_id_to_name: Dict[int, str] = {}
        # 跟踪ID到特征向量的缓存
        self.track_id_to_embedding: Dict[int, NDArray[np.float32]] = {}
        # 未识别跟踪ID的计数器（用于避免频繁识别）
        self.unrecognized_counter: Dict[int, int] = {}

    def _bbox_to_mp_coords(self, bbox: NDArray[np.float32], w: int, h: int) -> NDArray[np.float32]:
        """将边界框中心转换为 [-1, 1] 范围的坐标（MediaPipe风格）。"""
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        norm_x = (center_x / w) * 2.0 - 1.0
        norm_y = (center_y / h) * 2.0 - 1.0
        return np.array([norm_x, norm_y], dtype=np.float32)

    def _recognize_face(
            self,
            img: NDArray[np.uint8],
            bbox: NDArray[np.float32],
            track_id: int
    ) -> Tuple[Optional[str], Optional[NDArray[np.float32]]]:
        """
        识别人脸

        Args:
            img: 原始图像
            bbox: 边界框
            track_id: 跟踪ID

        Returns:
            (名字, 特征向量)
        """
        if not self.face_recognizer:
            return None, None

        try:
            # 如果已经识别过，直接返回缓存
            if track_id in self.track_id_to_name:
                return self.track_id_to_name[track_id], self.track_id_to_embedding.get(track_id)

            # 避免频繁识别未识别的人脸
            if track_id in self.unrecognized_counter:
                self.unrecognized_counter[track_id] += 1
                if self.unrecognized_counter[track_id] < 5:  # 每5帧识别一次
                    return None, None
                self.unrecognized_counter[track_id] = 0

            # 裁剪人脸区域
            face_img = self.face_recognizer.crop_face_from_bbox(img, bbox)
            if face_img is None:
                return None, None

            # 提取特征
            embedding = self.face_recognizer.extract_embedding(face_img)
            if embedding is None:
                return None, None

            # 识别
            name, score = self.face_recognizer.recognize(embedding)

            if name:
                # 识别成功，更新缓存
                self.track_id_to_name[track_id] = name
                self.track_id_to_embedding[track_id] = embedding
                if track_id in self.unrecognized_counter:
                    del self.unrecognized_counter[track_id]
                logger.debug(f"Track {track_id} recognized as '{name}' (score: {score:.3f})")
            else:
                # 未识别，更新计数器
                self.unrecognized_counter[track_id] = self.unrecognized_counter.get(track_id, 0) + 1
                logger.debug(f"Track {track_id} not recognized (score: {score:.3f})")

            return name, embedding

        except Exception as e:
            logger.error(f"Error recognizing face for track {track_id}: {e}")
            return None, None

    def get_head_positions(self, img: NDArray[np.uint8]) -> Tuple[Detections, List[Position]]:
        """
        检测并跟踪所有人脸，返回每个被跟踪人脸的 (track_id, center_coord, roll_angle, name)。

        Args:
            img: 输入图像 (H, W, 3)

        Returns:
            列表，每个元素为 (track_id, center_coord, roll, name)，其中 center_coord 是 [-1,1] 坐标，
            roll 目前固定为 0.0，name 为识别出的人名（如果启用识别）。
        """
        h, w = img.shape[:2]

        try:
            # YOLO 推理
            results = self.model(img, verbose=False)
            # 转换为 supervision 的 Detections 格式
            detections = Detections.from_ultralytics(results[0])

            # 过滤低置信度检测
            if detections.confidence is not None:
                mask = detections.confidence >= self.confidence_threshold
                detections = detections[mask]

            # 如果没有检测到人脸，返回空列表
            if len(detections) == 0:
                logger.debug("No faces detected above confidence threshold")
                return Detections.empty(), []

            # 更新跟踪器，为每个检测分配或更新 track_id
            if self.tracker:
                detections = self.tracker.update_with_detections(detections)

            # 构建结果列表
            positions: List[Position] = []
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                track_id = -1
                if detections.tracker_id:
                    track_id = detections.tracker_id[i]
                confidence = detections.confidence[i] if detections.confidence is not None else 1.0

                # 计算中心坐标
                center = self._bbox_to_mp_coords(bbox, w, h)

                # 识别人脸
                name, embedding = self._recognize_face(img, bbox, track_id)

                # 创建Position对象
                position = Position(
                    track_id=track_id,
                    bbox=bbox,
                    center=center,
                    confidence=confidence,
                    name=name,
                    embedding=embedding,
                    is_recognized=name is not None
                )
                positions.append(position)

            # 清理不再存在的track_id的缓存
            if detections.tracker_id:
                current_ids = set(detections.tracker_id)
                self.track_id_to_name = {k: v for k, v in self.track_id_to_name.items() if k in current_ids}
                self.track_id_to_embedding = {k: v for k, v in self.track_id_to_embedding.items() if k in current_ids}
                self.unrecognized_counter = {k: v for k, v in self.unrecognized_counter.items() if k in current_ids}

            logger.debug(f"Tracked {len(positions)} faces: IDs {[p.track_id for p in positions]}")
            return detections, positions

        except Exception as e:
            logger.error(f"Error in head position tracking: {e}")
            return Detections.empty(), []
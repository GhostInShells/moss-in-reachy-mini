import warnings
# 抑制insightface的skimage弃用警告
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='`estimate` is deprecated since version 0.26'
)


import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from ghoshell_common.contracts import Storage, LoggerItf
from ghoshell_container import IoCContainer, Container
from insightface.app import FaceAnalysis
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from moss_in_reachy_mini.camera.model import KnownFace, Position

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """人脸识别器，使用InsightFace进行人脸识别"""

    def __init__(
            self,
            model_name: str = "buffalo_l",
            det_thresh: float = 0.5,
            det_size: Tuple[int, int] = (480, 480),
            recognition_threshold: float = 0.5,
            device: str = "cpu",
            known_faces_storage: Optional[Storage] = None,
            container: IoCContainer = None,
    ):
        """
        初始化人脸识别器

        Args:
            model_name: InsightFace模型名称
            det_thresh: 检测阈值
            det_size: 检测尺寸
            recognition_threshold: 识别阈值
            device: 设备 (cpu/cuda)
            known_faces_storage: 已知人脸数据存储
        """
        self.recognition_threshold = recognition_threshold
        self.device = device
        self.container = Container(parent=container)
        self.logger = self.container.get(LoggerItf) or logging.getLogger("FaceRecognizer")

        # 初始化InsightFace应用
        try:
            self.app = FaceAnalysis(
                name=model_name,
                providers=['CPUExecutionProvider'] if device == "cpu" else ['CUDAExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
            logger.info(f"InsightFace model '{model_name}' loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise

        # 加载或创建已知人脸数据库
        self.known_faces: Dict[str, KnownFace] = {}
        self.known_faces_storage = known_faces_storage
        self.known_faces_filename = "known_faces.json"
        self.load_known_faces()


    def get_face_positions(self, img: NDArray) -> List[Position]:
        h, w = img.shape[:2]
        start = time.time()
        faces = self.app.get(img)
        self.logger.debug(f"Model cost {time.time() - start} seconds")
        positions: List[Position] = []
        for i, face in enumerate(faces):
            tid = -1

            # 计算中心坐标
            center_x = (face.bbox[0] + face.bbox[2]) / 2.0
            center_y = (face.bbox[1] + face.bbox[3]) / 2.0
            norm_x = (center_x / w) * 2.0 - 1.0
            norm_y = (center_y / h) * 2.0 - 1.0
            center = np.array([norm_x, norm_y], dtype=np.float32)

            # 识别人脸
            name, embedding = self._recognize_with_img(img, face)

            # 创建Position对象
            position = Position(
                track_id=tid,
                bbox=face.bbox,
                center=center,
                confidence=face.det_confidence,
                name=name,
                embedding=embedding,
                is_recognized=name is not None
            )
            positions.append(position)

        return positions

    def _recognize_with_img(
            self,
            img: NDArray[np.uint8],
            face: Any,
    ) -> Tuple[Optional[str], Optional[NDArray[np.float32]]]:
        """
        识别人脸

        Args:
            img: 原始图像
            face: face

        Returns:
            (名字, 特征向量)
        """
        try:

            # 裁剪人脸区域
            face_img = self.crop_face_from_bbox(img, face.bbox)
            if face_img is None:
                return None, None

            # 提取特征
            embedding = face.normed_embedding.astype(np.float32)

            # 识别
            name, score = self.recognize(embedding)

            return name, embedding

        except Exception as e:
            logger.error(f"Error recognizing face error: {e}")
            return None, None

    def extract_embedding(self, face_img: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        """
        从人脸图像中提取特征向量

        Args:
            face_img: 人脸图像 (RGB格式)

        Returns:
            512维特征向量，如果检测失败返回None
        """
        try:
            # 转换到BGR格式（InsightFace使用BGR）
            bgr_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

            # 检测人脸并提取特征
            faces = self.app.get(bgr_img)

            if len(faces) > 0:
                # 返回第一个检测到的人脸特征
                embedding = faces[0].normed_embedding.astype(np.float32)
                return embedding

            return None

        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None

    def crop_face_from_bbox(
            self,
            img: NDArray[np.uint8],
            bbox: NDArray[np.float32],
            padding: float = 0.2
    ) -> Optional[NDArray[np.uint8]]:
        """
        根据边界框裁剪人脸区域

        Args:
            img: 原始图像 (H, W, 3)
            bbox: 边界框 [x1, y1, x2, y2]
            padding: 扩展比例

        Returns:
            裁剪后的人脸图像
        """
        try:
            h, w = img.shape[:2]

            # 计算带padding的边界框
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # 添加padding
            x1 = max(0, int(x1 - width * padding))
            y1 = max(0, int(y1 - height * padding))
            x2 = min(w, int(x2 + width * padding))
            y2 = min(h, int(y2 + height * padding))

            # 裁剪人脸区域
            face_img = img[y1:y2, x1:x2]

            if face_img.size == 0:
                return None

            # 调整大小（可选，InsightFace会自己处理）
            return face_img

        except Exception as e:
            logger.error(f"Error cropping face: {e}")
            return None

    def recognize(
            self,
            embedding: NDArray[np.float32]
    ) -> Tuple[Optional[str], float]:
        """
        识别人脸

        Args:
            embedding: 人脸特征向量

        Returns:
            (名字, 相似度得分)，如果未识别返回(None, 0.0)
        """
        if not self.known_faces:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for name, known_face in self.known_faces.items():
            # 计算余弦相似度
            similarity = self.cosine_similarity(embedding, known_face.embedding)

            self.logger.debug(f"recognize {name} with similarity {similarity:.3f}")

            if similarity > best_score and similarity >= self.recognition_threshold:
                best_score = similarity
                best_match = name

        return best_match, best_score

    def cosine_similarity(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def add_known_face(
            self,
            name: str,
            embedding: NDArray[np.float32],
            metadata: Optional[Dict[str, Any]] = None,
            save: bool = True,
    ) -> None:
        """
        添加已知人脸到数据库

        Args:
            name: 人名
            embedding: 人脸特征向量
            metadata: 额外元数据
            save: 是否直接保存
        """
        if name in self.known_faces:
            # 如果已存在，可以更新或合并特征
            existing = self.known_faces[name]
            # 简单平均合并特征
            total_samples = existing.sample_count + 1
            existing.embedding = (existing.embedding * existing.sample_count + embedding) / total_samples
            existing.sample_count = total_samples
            if metadata:
                existing.metadata.update(metadata)
            logger.info(f"Updated known face '{name}' (samples: {total_samples})")
        else:
            # 新的人脸
            self.known_faces[name] = KnownFace(
                name=name,
                embedding=embedding,
                metadata=metadata or {},
                sample_count=1
            )
            logger.info(f"Added new known face '{name}'")

        if save:
            self.save_known_faces()

    def add_known_face_from_image(
            self,
            name: str,
            face_img: NDArray[np.uint8],
            metadata: Optional[Dict[str, Any]] = None,
            save: bool = True,
    ) -> bool:
        """
        从图像添加已知人脸

        Args:
            name: 人名
            face_img: 人脸图像 (RGB格式)
            metadata: 额外元数据
            save: 是否直接存储

        Returns:
            是否成功添加
        """
        embedding = self.extract_embedding(face_img)
        if embedding is not None:
            self.add_known_face(name, embedding, metadata, save)
            return True
        return False

    def remove_known_face(self, name: str, save: bool=False) -> bool:
        """从数据库中移除已知人脸"""
        if name in self.known_faces:
            del self.known_faces[name]
            logger.info(f"Removed known face '{name}'")
            if save:
                self.save_known_faces()
            return True
        return False

    def rename_known_face(self, old_name: str, new_name: str, save: bool = True) -> bool:
        """重命名已知人脸"""
        if old_name not in self.known_faces:
            return False
        if new_name in self.known_faces:
            raise ValueError(f"'{new_name}'已存在，请换一个名字")
        face = self.known_faces.pop(old_name)
        face.name = new_name
        self.known_faces[new_name] = face
        logger.info(f"Renamed known face '{old_name}' -> '{new_name}'")
        if save:
            self.save_known_faces()
        return True

    def save_known_faces(self) -> None:
        """保存已知人脸数据库到文件"""
        if not self.known_faces_storage:
            return
        try:
            data = {
                'known_faces': {name: face.to_dict() for name, face in self.known_faces.items()},
                'recognition_threshold': self.recognition_threshold
            }
            json_bytes = json.dumps(data, indent=4)
            self.known_faces_storage.put(self.known_faces_filename, json_bytes.encode())
            logger.info(f"Saved {len(self.known_faces)} known faces")

        except Exception as e:
            logger.error(f"Failed to save known faces: {e}")
            raise

    def load_known_faces(self) -> None:
        """从文件加载已知人脸数据库"""
        if not self.known_faces_storage:
            return
        if not self.known_faces_storage.exists(self.known_faces_filename):
            return
        try:
            json_bytes = self.known_faces_storage.get(self.known_faces_filename)
            data = json.loads(json_bytes)

            self.known_faces.clear()
            for name, face_data in data.get('known_faces', {}).items():
                self.known_faces[name] = KnownFace.from_dict(face_data)

            self.recognition_threshold = data.get('recognition_threshold', self.recognition_threshold)
            logger.info(f"Loaded {len(self.known_faces)} known faces")

        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")
            raise

    def get_known_faces(self) -> List[str]:
        """获取所有已知人脸的名字"""
        return list(self.known_faces.keys())

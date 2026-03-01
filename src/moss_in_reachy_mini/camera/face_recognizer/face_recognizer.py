import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from ghoshell_common.contracts import Storage
from insightface.app import FaceAnalysis
from numpy.typing import NDArray

from moss_in_reachy_mini.camera.yolo.model import KnownFace

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """人脸识别器，使用InsightFace进行人脸识别"""

    def __init__(
            self,
            model_name: str = "buffalo_l",
            det_thresh: float = 0.5,
            det_size: Tuple[int, int] = (640, 640),
            recognition_threshold: float = 0.5,
            device: str = "cpu",
            known_faces_storage: Optional[Storage] = None,
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

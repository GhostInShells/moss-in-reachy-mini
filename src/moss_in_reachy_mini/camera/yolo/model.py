from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class Position:
    """人脸位置信息"""
    track_id: int
    bbox: np.ndarray
    center: NDArray[np.float32]
    roll: float = 0.0
    confidence: float = 1.0
    name: Optional[str] = None  # 新增：识别出的名字
    embedding: Optional[NDArray[np.float32]] = None  # 新增：人脸特征向量
    is_recognized: bool = False  # 新增：是否已识别

    @classmethod
    def new(cls, track_id: int, center: NDArray[np.float32], **kwargs) -> 'Position':
        """创建新的Position实例"""
        return cls(track_id=track_id, center=center, **kwargs)


@dataclass
class KnownFace:
    """已知人脸信息"""
    name: str
    embedding: NDArray[np.float32]  # 人脸特征向量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    sample_count: int = 0  # 样本数量

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata,
            'sample_count': self.sample_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnownFace':
        """从字典创建"""
        return cls(
            name=data['name'],
            embedding=np.array(data['embedding'], dtype=np.float32),
            metadata=data.get('metadata', {}),
            sample_count=data.get('sample_count', 0)
        )

def get_position_by_track_id(positions: List[Position], track_id: int) -> Position | None:
    for p in positions:
        if p.track_id == track_id:
            return p
    return None

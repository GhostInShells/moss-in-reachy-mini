import enum
import json


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        # 如果是Enum实例，返回其value
        if isinstance(obj, enum.Enum):
            return obj.value
        # 其他类型按默认逻辑处理
        return super().default(obj)

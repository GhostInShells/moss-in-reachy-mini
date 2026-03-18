from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal


# 嵌套模型：server_tool_use 字段
class ServerToolUse(BaseModel):
    """服务端工具使用统计"""
    web_search_requests: int = Field(default=0, description="网页搜索请求次数")
    web_fetch_requests: int = Field(default=0, description="网页抓取请求次数")


# 嵌套模型：cache_creation 字段
class CacheCreation(BaseModel):
    """缓存创建相关的Token统计"""
    ephemeral_1h_input_tokens: int = Field(default=0, description="1小时临时缓存输入Token数")
    ephemeral_5m_input_tokens: int = Field(default=0, description="5分钟临时缓存输入Token数")


# 嵌套模型：usage 主字段
class UsageStats(BaseModel):
    """整体Token使用和工具调用统计"""
    input_tokens: int = Field(default=0, description="输入Token总数")
    cache_creation_input_tokens: int = Field(default=0, description="缓存创建时的输入Token数")
    cache_read_input_tokens: int = Field(default=0, description="缓存读取时的输入Token数")
    output_tokens: int = Field(default=0, description="输出Token总数")
    server_tool_use: ServerToolUse = Field(
        default_factory=ServerToolUse,  # 嵌套模型用default_factory创建实例
        description="服务端工具使用详情"
    )
    service_tier: str = Field(default="", description="服务等级（如standard）")
    cache_creation: CacheCreation = Field(
        default_factory=CacheCreation,
        description="缓存创建统计"
    )
    inference_geo: str = Field(default="", description="推理服务的地域（为空时表示未返回）")
    iterations: List[Any] = Field(
        default_factory=list,  # 列表用default_factory=list避免可变默认值问题
        description="迭代过程列表，无数据时为空列表"
    )
    speed: str = Field(default="", description="推理速度等级（如standard）")


# 嵌套模型：tool_input 字段（permission_denials内）
class ToolInput(BaseModel):
    """被拒绝的工具调用输入参数"""
    query: str = Field(default="", description="工具调用的查询语句")


# 嵌套模型：permission_denials 列表项
class PermissionDenial(BaseModel):
    """权限拒绝记录"""
    tool_name: str = Field(default="", description="被拒绝的工具名称（如WebSearch）")
    tool_use_id: str = Field(default="", description="工具调用ID")
    tool_input: ToolInput = Field(
        default_factory=ToolInput,
        description="工具调用的输入参数"
    )


# 嵌套模型：modelUsage 内的单个模型统计项
class ModelUsageItem(BaseModel):
    """单个模型的Token使用和成本统计"""
    inputTokens: int = Field(default=0, description="模型输入Token数")
    outputTokens: int = Field(default=0, description="模型输出Token数")
    cacheReadInputTokens: int = Field(default=0, description="模型缓存读取输入Token数")
    cacheCreationInputTokens: int = Field(default=0, description="模型缓存创建输入Token数")
    webSearchRequests: int = Field(default=0, description="模型触发的网页搜索请求数")
    costUSD: float = Field(default=0.0, description="模型调用的美元成本")
    contextWindow: int = Field(default=0, description="模型的上下文窗口大小")
    maxOutputTokens: int = Field(default=0, description="模型最大输出Token数")


# 顶层主模型
class ClaudeResult(BaseModel):
    """AI响应结果的顶层模型（对应你提供的JSON结构）"""
    type: str = Field(default="", description="结果类型（如result）")
    subtype: str = Field(default="", description="结果子类型（如success）")
    is_error: bool = Field(default=False, description="是否为错误结果")
    duration_ms: int = Field(default=0, description="总耗时（毫秒）")
    duration_api_ms: int = Field(default=0, description="API层耗时（毫秒）")
    num_turns: int = Field(default=0, description="对话轮次")
    result: str = Field(default="", description="响应的文本结果")
    stop_reason: str = Field(default="", description="对话停止原因（如end_turn）")
    session_id: str = Field(default="", description="会话ID")
    total_cost_usd: float = Field(default=0.0, description="总调用成本（美元）")
    usage: UsageStats = Field(
        default_factory=UsageStats,
        description="Token使用统计详情"
    )
    modelUsage: Dict[str, ModelUsageItem] = Field(
        default_factory=dict,  # 字典用default_factory=dict
        description="各模型的使用统计（键为模型名）"
    )
    permission_denials: List[PermissionDenial] = Field(
        default_factory=list,
        description="权限拒绝记录列表"
    )
    fast_mode_state: str = Field(default="", description="快速模式状态（如off）")
    uuid: str = Field(default="", description="唯一标识ID")


# default_config.py
from framework.apps.live.barrage_classify.config import BarrageClassifierConfig, KeywordGroupConfig, BarrageType, Priority

# 默认配置 - 你可以在这里修改所有关键词，不需要改动分类器逻辑
DEFAULT_CONFIG = BarrageClassifierConfig(
    # 类型关键词配置
    type_keywords={
        BarrageType.QUESTION: KeywordGroupConfig(
            keywords=[
                "吗", "呢", "怎么", "如何", "为什么", "啥", "多少", "几时",
                "何时", "哪里", "谁", "哪个", "会不会", "能不能", "可不可以",
                "？", "?", "是不是", "有没有", "如何", "怎样", "请教", "问一下"
            ]
        ),
        BarrageType.COMMAND: KeywordGroupConfig(
            keywords=[
                "请", "让", "要", "想", "希望", "建议", "推荐", "来一个",
                "表演", "展示", "演示", "试试", "试一下", "跳", "唱", "转",
                "抓", "写", "画", "做", "展示一下", "表演一下", "来段"
            ]
        ),
        BarrageType.BUSINESS: KeywordGroupConfig(
            keywords=[
                "价格", "价钱", "多少钱", "售价", "报价", "买", "购买", "下单",
                "订购", "预定", "链接", "网址", "官网", "淘宝", "京东", "拼多多",
                "功能", "参数", "配置", "规格", "技术", "原理", "专利", "核心"
            ]
        ),
        BarrageType.PRAISE: KeywordGroupConfig(
            keywords=[
                "好", "棒", "厉害", "强", "牛", "酷", "帅", "美", "漂亮",
                "精彩", "完美", "优秀", "专业", "顶级", "一流", "超赞",
                "喜欢", "爱", "佩服", "崇拜", "羡慕", "666", "888", "999",
                "太棒了", "太好了", "真厉害", "真牛", "真强", "绝了", "无敌"
            ]
        ),
    },

    # 优先级关键词配置 - 针对AI互动直播优化
    priority_keywords={
        Priority.P0: KeywordGroupConfig(
            keywords=[
                # 高度互动请求
                "跳个舞", "来段舞", "表演一下", "展示才艺", "秀一下", "来段表演",
                "跳支舞", "跳一段", "表演节目", "才艺表演", "来点才艺",

                # 挑战/测试请求
                "挑战一下", "试试看", "能做到吗", "敢不敢", "来试试", "测试一下",
                "证明一下", "看看实力", "展示实力", "实力如何",

                # 即时反馈请求
                "评价一下", "说说看法", "你怎么看", "发表意见", "谈谈感想",
                "点评一下", "分析分析", "解读一下", "解释解释",

                # 情感强烈表达
                "太棒了", "太厉害了", "太强了", "无敌了", "绝了", "封神了",
                "天花板", "yyds", "永远的神", "最强", "第一",

                # 特殊互动请求
                "跟我对话", "回答我", "跟我聊天", "互动一下", "聊聊天",
                "说说话", "唠唠嗑", "交流一下", "对话一下",

                # 新闻相关即时请求
                "最新消息", "刚刚发生", "突发", "快讯", "紧急新闻",
                "重大新闻", "重要消息", "最新进展", "实时更新",

                # 游戏/趣味互动
                "玩个游戏", "猜谜语", "讲笑话", "说段子", "脑筋急转弯",
                "互动游戏", "小游戏", "娱乐一下", "轻松一下",

                # 知识/学习请求
                "教教我", "学习一下", "科普一下", "讲解一下", "传授经验",
                "分享知识", "教学一下", "指导一下", "传授技巧",
            ],
            weight=2.0
        ),

        Priority.P1: KeywordGroupConfig(
            keywords=[
                # 一般互动请求
                "请问", "请教", "问一下", "帮忙", "帮助", "求助",
                "建议", "推荐", "介绍", "说明", "解释",

                # 一般赞美
                "好棒", "厉害", "优秀", "专业", "精彩", "完美",
                "喜欢", "爱了", "佩服", "崇拜", "羡慕",

                # 一般问题
                "是什么", "为什么", "怎么样", "如何", "怎么",
                "哪里", "何时", "谁", "哪个",
            ],
            weight=1.5
        ),
    },

    # 停用词配置
    stop_words=[
        "广告", "骗子", "垃圾", "滚", "sb", "傻逼", "fuck", "shit",
        "死妈", "去死", "操你", "垃圾主播", "取关"
    ],

    # 简单问候词配置
    simple_greetings=[
        "你好", "嗨", "hello", "hi", "大家好", "哈喽",
        "早上好", "晚上好", "下午好", "拜拜", "再见"
    ],

    # 默认优先级映射配置
    default_priority_map={
        BarrageType.QUESTION: Priority.P1,
        BarrageType.COMMAND: Priority.P1,
        BarrageType.BUSINESS: Priority.P2,
        BarrageType.PRAISE: Priority.P2,
        BarrageType.GREET: Priority.P3,
        BarrageType.OTHER: Priority.P3,
    },

    # 时间窗口配置
    repeat_detection_window=30,
    history_cleanup_interval=60
)
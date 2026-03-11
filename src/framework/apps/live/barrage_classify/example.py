# main.py
from default_config import DEFAULT_CONFIG
from framework.apps.live.barrage_classify.classifier import BarrageClassifier


def main():
    # 方式1：使用默认配置
    classifier = BarrageClassifier(DEFAULT_CONFIG)

    # 测试弹幕
    test_barrages = [
        ("跳个舞吧", "user1"),
        ("这个多少钱？", "user2"),
        ("太厉害了！", "user3"),
        ("你好", "user4"),
        ("最新消息是什么", "user5"),
        ("玩个游戏吧", "user6"),
    ]

    print("弹幕分类测试:")
    print("-" * 60)

    for barrage, user_id in test_barrages:
        bar_type, priority = classifier.classify(barrage, user_id)
        print(f"弹幕: {barrage}")
        print(f"  用户: {user_id}")
        print(f"  类型: {bar_type.value} | 优先级: {priority.name}")
        print("-" * 60)


if __name__ == "__main__":
    main()
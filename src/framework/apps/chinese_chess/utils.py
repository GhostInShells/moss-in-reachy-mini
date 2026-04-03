def parse_chinese_board(board_text):
    """
    解析中文棋盘的字符画，返回字典，键为坐标（如 'a3'），值为中文棋子名称。
    棋子示例：黑士、黑将、红兵、红帅等。
    """
    lines = board_text.strip().splitlines()
    pieces = {}

    for line in lines:
        # 跳过边框行（包含 --- 的行）
        if '---' in line:
            continue

        # 找到最后一个 '|'，它后面是行号
        last_pipe = line.rfind('|')
        if last_pipe == -1:
            continue

        # 提取行号（y坐标）
        y_str = line[last_pipe+1:].strip()
        if not y_str.isdigit():
            continue
        y = int(y_str)

        # 提取棋盘部分（最后一个 '|' 之前的内容）
        board_part = line[:last_pipe]

        # 按 '|' 分割，第一个元素为空（因为开头有 '|'），所以从索引1开始取9个单元格
        cells = board_part.split('|')[1:10]  # 确保只取前9个

        # 遍历列
        for col_idx, cell in enumerate(cells):
            cell = cell.strip()
            if cell:  # 非空表示有棋子
                col_letter = chr(ord('a') + col_idx)  # 0->a, 1->b, ..., 8->i
                key = f"{col_letter}{y}"
                pieces[key] = cell

    return pieces


def number_to_chinese(num: int) -> str:
    """将数字 1-9 转换为中文数字"""
    chinese_digits = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    if 1 <= num <= 9:
        return chinese_digits[num - 1]
    else:
        return str(num)


def uci_to_chinese_notation(board_text: str, uci_move: str) -> str:
    """
    将 UCI 走棋转换为专业象棋术语（如“车一进一”）。

    Args:
        board_text: 棋盘文本，与 parse_chinese_board 兼容的格式
        uci_move: UCI 格式的走棋，如 "a0a1"

    Returns:
        专业象棋术语字符串
    """
    # 解析棋盘
    pieces = parse_chinese_board(board_text)

    # 解析 UCI 走棋
    if len(uci_move) != 4:
        raise ValueError(f"无效的 UCI 走棋格式: {uci_move}")
    from_uci = uci_move[0:2]
    to_uci = uci_move[2:4]

    # 获取棋子
    piece_name = pieces.get(from_uci)
    if not piece_name:
        piece_name = pieces.get(to_uci)

    if not piece_name:
        raise ValueError(f"起始坐标和目标坐标 {from_uci} 和 {to_uci} 没有棋子")

    # 解析棋子颜色和类型
    color = "红" if piece_name.startswith("红") else "黑"
    piece_type = piece_name[1:]  # 去掉颜色前缀

    # 坐标转换
    def uci_to_coords(uci: str):
        col_char = uci[0].lower()
        row_char = uci[1]
        col_idx = ord(col_char) - ord('a')  # 0-8, a 最左
        row = int(row_char)  # 0-9, 0 红方底线（底部）
        return col_idx, row

    from_col_idx, from_row = uci_to_coords(from_uci)
    to_col_idx, to_row = uci_to_coords(to_uci)

    # 计算纵线编号（从红方视角，右→左 1-9）
    # a 最左为第9纵线，i 最右为第1纵线
    def col_idx_to_file(col_idx: int) -> int:
        return 9 - col_idx  # 9-0=9, 9-8=1

    from_file = col_idx_to_file(from_col_idx)
    to_file = col_idx_to_file(to_col_idx)

    # 计算移动方向
    delta_row = to_row - from_row
    delta_file = to_file - from_file  # 正数表示向右移动（从红方视角）

    # 根据颜色确定前进方向
    # 红方：行增加为进（向黑方），行减少为退
    # 黑方：行减少为进（向红方），行增加为退
    if color == "红":
        forward = delta_row > 0
        backward = delta_row < 0
    else:  # 黑方
        forward = delta_row < 0
        backward = delta_row > 0

    # 生成术语
    # 棋子类型映射（统一使用红方术语）
    piece_map = {
        "帅": "帅", "将": "将",
        "仕": "仕", "士": "士",
        "相": "相", "象": "象",
        "马": "马",
        "车": "车",
        "炮": "炮",
        "兵": "兵", "卒": "卒"
    }
    piece_char = piece_map.get(piece_type, piece_type)

    # 转换为中文数字
    from_file_ch = number_to_chinese(from_file)
    to_file_ch = number_to_chinese(to_file)

    # 判断移动类型
    if delta_file == 0:
        # 纵向移动
        steps = abs(delta_row)
        steps_ch = number_to_chinese(steps)
        if forward:
            return f"{piece_char}{from_file_ch}进{steps_ch}"
        elif backward:
            return f"{piece_char}{from_file_ch}退{steps_ch}"
        else:
            # 没有移动？不应该发生
            return f"{piece_char}{from_file_ch}平{to_file_ch}"
    else:
        # 横向或斜向移动
        if delta_row == 0:
            # 纯横向移动
            return f"{piece_char}{from_file_ch}平{to_file_ch}"
        else:
            # 斜向移动（马、象、士）
            # 对于马、相、象、仕、士，使用“进/退”加上目标纵线
            if piece_char in ("马", "相", "象", "仕", "士"):
                direction = "进" if forward else "退"
                return f"{piece_char}{from_file_ch}{direction}{to_file_ch}"
            else:
                # 其他棋子的斜向移动（如兵过河后斜走？实际上兵不能斜走）
                # 回退到简单表示
                direction = "进" if forward else "退"
                steps_ch = number_to_chinese(abs(delta_row))
                return f"{piece_char}{from_file_ch}到{to_file_ch}{direction}{steps_ch}"


def main():
    board_text = """
     +---+---+---+---+---+---+---+---+---+
     |   |   |   |黑士|黑将|黑士|黑象|   |   | 9
     +---+---+---+---+---+---+---+---+---+
     |   |   |   |   |   |黑车|   |   |   | 8
     +---+---+---+---+---+---+---+---+---+
     |黑象|   |黑炮|   |   |   |   |   |   | 7
     +---+---+---+---+---+---+---+---+---+
     |黑卒|   |   |   |黑马|   |   |   |黑卒| 6
     +---+---+---+---+---+---+---+---+---+
     |   |   |   |   |   |   |黑卒|   |   | 5
     +---+---+---+---+---+---+---+---+---+
     |   |   |红兵|黑马|   |   |   |   |   | 4
     +---+---+---+---+---+---+---+---+---+
     |红兵|   |   |   |红兵|   |红兵|   |红兵| 3
     +---+---+---+---+---+---+---+---+---+
     |红相|   |   |   |红炮|   |红马|   |   | 2
     +---+---+---+---+---+---+---+---+---+
     |   |   |   |   |红仕|   |   |   |红车| 1
     +---+---+---+---+---+---+---+---+---+
     |   |   |   |红仕|红帅|   |红相|   |   | 0
     +---+---+---+---+---+---+---+---+---+
       a   b   c   d   e   f   g   h   i
    """

    pieces = parse_chinese_board(board_text)

    # 打印结果（按坐标排序）
    for coord in sorted(pieces.keys(), key=lambda x: (int(x[1:]), x[0])):
        print(f"{coord}: {pieces[coord]}")

    # 测试 UCI 转专业术语
    print("\n=== UCI 转专业术语测试 ===")
    test_cases = [
        ("i1i2", "红车从 i1 移动到 i2 (车一进一)"),
        ("g2e3", "红马从 g2 移动到 e3 (马三进五)"),
        ("e0e1", "红帅从 e0 移动到 e1 (帅五进一)"),
        ("d0c1", "红仕从 d0 移动到 c1 (仕六进七)"),
        ("c4c5", "红兵从 c4 移动到 c5 (兵七进一)"),
        ("f8f7", "黑车从 f8 移动到 f7 (车四进一)"),
        ("e6d4", "黑马从 e6 移动到 d4 (马5进3)"),
        ("g5g4", "黑卒从 g5 移动到 g4 (卒3进1)"),
    ]

    for uci_move, description in test_cases:
        try:
            notation = uci_to_chinese_notation(board_text, uci_move)
            print(f"{uci_move}: {description} -> {notation}")
        except Exception as e:
            print(f"{uci_move}: 错误 - {e}")


if __name__ == '__main__':
    main()
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


if __name__ == '__main__':
    main()
from pynput import keyboard

def on_press(key):
    try:
        # 普通字符键
        print(f"按下: {key.char}")
    except AttributeError:
        # 特殊键（如 Shift、Ctrl、Enter 等）
        print(f"按下: {key}")

def on_release(key):
    print(f"释放: {key}")
    # 按 Esc 退出监听
    if key == keyboard.Key.esc:
        print("退出监听")
        return False

print("开始监听按键（按 Esc 退出）...")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

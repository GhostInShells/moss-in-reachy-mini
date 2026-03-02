from pathlib import Path

import cv2
from ghoshell_common.contracts import FileStorage, LocalWorkspace

from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer


def from_storage(face_recognizer: FaceRecognizer, storage: FileStorage):
    """
    从目录批量训练已知人脸

    Args:
        face_recognizer: FaceRecognizer
        storage: 数据目录，结构为：
            data_dir/
                person1/
                    image1.jpg
                    image2.jpg
                person2/
                    image1.jpg
    """
    path = storage.abspath()
    from_path(face_recognizer, path)

def from_path(face_recognizer: FaceRecognizer, path: str):
    data_path = Path(path)
    if not data_path.exists():
        print(f"Directory {path} does not exist")
        return

    # 遍历每个人物文件夹
    for person_dir in data_path.iterdir():
        if person_dir.is_dir():
            person_name = person_dir.name
            print(f"Training {person_name}...")

            # 遍历该人物的所有图片
            image_count = 0
            for suffix_format in ["*.jpg", "*.png", "*.jpeg"]:
                for img_path in person_dir.glob(suffix_format):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue

                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # 检测人脸
                        positions = face_recognizer.get_face_positions(img_rgb)

                        if positions:
                            # 使用第一个检测到的人脸
                            position = positions[0]
                            if position.embedding is not None:
                                # 添加到数据库
                                face_recognizer.add_known_face(person_name, position.embedding)
                                image_count += 1
                                print(f"  Added {img_path.name}")
                    except Exception as e:
                        print(f"  Error processing {img_path}: {e}")

            print(f"  Trained {image_count} images for {person_name}")

    face_recognizer.save_known_faces()

def main():
    import pathlib
    ws_dir = pathlib.Path(__file__).parent.parent.joinpath(".workspace")
    ws = LocalWorkspace(str(ws_dir.absolute()))

    face_recognizer = FaceRecognizer(
        known_faces_storage=ws.configs().sub_storage("face_recognizer")
    )

    from_storage(face_recognizer, ws.runtime().sub_storage("vision").sub_storage("faces"))

if __name__ == "__main__":
    main()
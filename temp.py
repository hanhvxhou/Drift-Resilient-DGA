import os

# Các thư mục cần bỏ qua
IGNORE_DIRS = {".git", ".venv"}

def tree(dir_path, output_file):
    with open(output_file, "w", encoding="utf-8") as f:

        def walk(current_path, prefix=""):
            items = sorted(
                item for item in os.listdir(current_path)
                if item not in IGNORE_DIRS
            )

            for i, item in enumerate(items):
                full_path = os.path.join(current_path, item)
                is_last = (i == len(items) - 1)

                connector = "└── " if is_last else "├── "
                f.write(prefix + connector + item + "\n")

                if os.path.isdir(full_path):
                    extension = "    " if is_last else "│   "
                    walk(full_path, prefix + extension)

        f.write(os.path.basename(dir_path) + "\n")
        walk(dir_path)

if __name__ == "__main__":
    folder = r"D:\job\pycharm\Drift-Resilient-DGA"   # Thư mục cần liệt kê
    output = "folder_structure.txt"

    tree(folder, output)

    print(f"Đã lưu cấu trúc vào {output}")
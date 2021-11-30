import os.path

import hashlib


def compute(folder: str) -> str:
    hash = hashlib.sha256()
    paths = [
        os.path.join(folder, file_name)
        for file_name in os.listdir(folder)
        if os.path.splitext(file_name)[1] == ".py"
    ]
    paths = sorted(paths, key=str.casefold)
    for path in paths:
        with open(path, "rb") as file:
            bytes = file.read()
            hash.update(bytes)
    return hash.hexdigest()


if __name__ == "__main__":
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
    print(compute(folder))

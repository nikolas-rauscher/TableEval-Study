import os
from pathlib import Path
from PIL import Image


def _default_image_dir() -> Path:
    project_root = Path(__file__).resolve().parents[4]
    base = os.environ.get("TABLEEVAL_DATA_ROOT", project_root / "data")
    return Path(base) / "ComTQA" / "FinTabNet" / "images"


def parse(samples, image_path=None):
    image_dir = Path(image_path) if image_path else _default_image_dir()
    inputs = []
    for sample in samples:
        with Image.open(image_dir / sample["image_name"]) as image:
            image = image.convert("RGB")
            #  min_size = 28
            #  new_width = max(image.width, min_size)
            # new_height = max(image.height, min_size)
            # image = image.resize((new_width, new_height))
            inputs.append([image.copy(), f'Refer to the provided table and answer the question. Question: {sample["question"]}'])

    return inputs

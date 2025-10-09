import os
from pathlib import Path
from PIL import Image


def _default_image_dir() -> Path:
    project_root = Path(__file__).resolve().parents[4]
    base = os.environ.get("TABLEEVAL_DATA_ROOT", project_root / "data")
    return Path(base) / "ComTQA" / "PubTab1M" / "images"


def parse(samples, image_path=None):
    if image_path is None:
        image_path = _default_image_dir()
    else:
        image_path = Path(image_path)
    inputs = []
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
            inputs.append([image.copy(), f'Refer to the provided table and answer the question. Question: {sample["question"]}. Table caption: {sample["table_title"]} {sample["table_caption"]}. Table footnote: {sample["table_footnote"]}.'])
    return inputs

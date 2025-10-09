import os
from pathlib import Path
from PIL import Image


def _default_image_dir() -> Path:
    project_root = Path(__file__).resolve().parents[4]
    base = os.environ.get("TABLEEVAL_DATA_ROOT", project_root / "data")
    return Path(base) / "numericNLG" / "generated_imgs"


def parse(samples, image_path=None):
    image_base = Path(image_path) if image_path else _default_image_dir()
    inputs = []

    for sample in samples:
        image = Image.open(image_base / sample["image_id"])
        image = image.convert("RGB")
        inputs.append([image.copy(), f'Describe the given table focusing on the insights and trends revealed by the results. The summary must be factual, coherent, and well-written. Do not introduce new information or speculate. Table caption: {sample["caption"]}'])
    return  inputs

import os
from pathlib import Path
from PIL import Image


def _default_image_dir() -> Path:
    project_root = Path(__file__).resolve().parents[4]
    base = os.environ.get("TABLEEVAL_DATA_ROOT", project_root / "data")
    return Path(base) / "Logic2Text" / "images"


def parse(samples, image_path=None):
    image_dir = Path(image_path) if image_path else _default_image_dir()
    inputs = []
    for sample in samples:
        with Image.open(image_dir / sample["image_name"]) as image:
            image = image.convert("RGB")
    
            inputs.append([image.copy(), f'Generate a one sentence statement based on the table and logical form. Logical form: {sample["logic_str"]}. Table title: {sample["title"]}'])
    return  inputs

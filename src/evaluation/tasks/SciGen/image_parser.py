import os
from pathlib import Path
from PIL import Image


def _default_image_dirs() -> dict:
    project_root = Path(__file__).resolve().parents[4]
    base = os.environ.get("TABLEEVAL_DATA_ROOT", project_root / "data")
    return {
        "other": Path(base) / "SciGen" / "test-Other" / "generated_imgs_other",
        "cl": Path(base) / "SciGen" / "test-CL" / "generated_imgs_cl_update_2025_01_08",
    }


def parse(samples, image_root: str | Path | None = None):
    dirs = _default_image_dirs()
    if image_root is not None:
        root = Path(image_root)
        dirs["other"] = root / "test-Other" / "generated_imgs_other"
        dirs["cl"] = root / "test-CL" / "generated_imgs_cl_update_2025_01_08"

    inputs = []
    for sample in samples:
        subset = sample.get("subset", "")
        dir_key = "other" if subset.startswith("other") else "cl"
        file_path = dirs[dir_key] / sample["image_id"]
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            inputs.append([image.copy(), f'Describe the given table focusing on the most important findings reported by reasoning over its content. The summary must be factual, coherent, and well-written. Do not introduce new information or speculate. Table caption: {sample["table_caption"]}'])
    return  inputs

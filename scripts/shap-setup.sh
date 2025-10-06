#!/usr/bin/env bash
# TableEval setup script: installs dependencies, downloads datasets, prepares environment variables.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN=python

export TABLEEVAL_DATA_ROOT="${TABLEEVAL_DATA_ROOT:-${ROOT_DIR}/data}"

PIP_ARGS=(install --upgrade --no-cache-dir)
if "${PYTHON_BIN}" -m pip help install 2>&1 | grep -q -- "--break-system-packages"; then
  PIP_ARGS+=(--break-system-packages)
fi

"${PYTHON_BIN}" -m pip "${PIP_ARGS[@]}" \
  shap==0.45.1 \
  evaluate==0.4.3 \
  litellm==1.55.8

"${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import zipfile

DATASET_ID = "katebor/TableEval"
DATA_ROOT = Path(os.environ["TABLEEVAL_DATA_ROOT"]).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

SPECS = [
    ("LogicNLG/logicnlg.json", "LogicNLG/full_dataset", "LogicNLG/logicnlg_imgs.zip", "LogicNLG/images"),
    ("numericNLG/numericnlg.json", "numericNLG/full_dataset", "numericNLG/numericnlg_imgs.zip", "numericNLG/generated_imgs"),
    ("Logic2Text/logic2text.json", "Logic2Text/full_dataset", "Logic2Text/logic2text_imgs.zip", "Logic2Text/images"),
    ("SciGen/scigen.json", "SciGen/full_dataset", "SciGen/scigen_imgs.zip", "SciGen/images"),
    ("ComTQA/PubTab1M/comtqa_pubtab1m.json", "ComTQA/PubTab1M/full_dataset", "ComTQA/PubTab1M/comtqa_pubtab1m_imgs.zip", "ComTQA/PubTab1M/images"),
    ("ComTQA/FinTabNet/comtqa_fintabnet.json", "ComTQA/FinTabNet/full_dataset", "ComTQA/FinTabNet/comtqa_fintabnet_imgs.zip", "ComTQA/FinTabNet/images"),
]

for data_file, dataset_dir, image_zip, image_dir in SPECS:
    dataset_path = DATA_ROOT / dataset_dir
    if not dataset_path.exists():
        ds = load_dataset(DATASET_ID, data_files=data_file)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(dataset_path))

    image_path = DATA_ROOT / image_dir
    if not image_path.exists():
        image_path.mkdir(parents=True, exist_ok=True)
        zip_path = hf_hub_download(DATASET_ID, filename=image_zip, repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            # Extract and flatten: move files from nested folder to image_path
            for member in zf.namelist():
                if member.endswith('.png'):
                    filename = Path(member).name
                    source = zf.open(member)
                    target = image_path / filename
                    with open(target, 'wb') as f:
                        f.write(source.read())
PY

cat <<EOF >"${ROOT_DIR}/env.sh"
#!/usr/bin/env bash
export TABLEEVAL_DATA_ROOT="${TABLEEVAL_DATA_ROOT}"
export PYTHONPATH="${ROOT_DIR}/src"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
EOF

chmod +x "${ROOT_DIR}/env.sh"

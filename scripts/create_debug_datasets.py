#!/usr/bin/env python3
"""Create debug subsets (20 samples) for all TableEval datasets."""

import os
from pathlib import Path
from datasets import load_from_disk, DatasetDict

DATA_ROOT = Path(os.environ.get('TABLEEVAL_DATA_ROOT', 'data')).resolve()
DEBUG_SIZE = 20

DATASETS = [
    ('LogicNLG', 'LogicNLG/full_dataset', 'LogicNLG/debug_subset'),
    ('numericNLG', 'numericNLG/full_dataset', 'numericNLG/debug_subset'),
    ('Logic2Text', 'Logic2Text/full_dataset', 'Logic2Text/debug_subset'),
    ('SciGen', 'SciGen/full_dataset', 'SciGen/debug_subset'),
    ('ComTQA/PubTab1M', 'ComTQA/PubTab1M/full_dataset', 'ComTQA/PubTab1M/debug_subset'),
    ('ComTQA/FinTabNet', 'ComTQA/FinTabNet/full_dataset', 'ComTQA/FinTabNet/debug_subset'),
]

for name, full_rel, debug_rel in DATASETS:
    full_path = DATA_ROOT / full_rel
    debug_path = DATA_ROOT / debug_rel

    if not full_path.exists():
        print(f"Skip {name}: full dataset not found")
        continue

    if debug_path.exists():
        print(f"Skip {name}: debug subset exists")
        continue

    print(f"Creating {name}...")
    full = load_from_disk(str(full_path))
    subset = DatasetDict({
        split: full[split].select(range(min(DEBUG_SIZE, len(full[split]))))
        for split in full.keys()
    })
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(debug_path))
    print(f"Created {name}")

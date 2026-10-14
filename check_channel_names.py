"""
check_channel_names.py

Prints the EXACT channel name list and order each dataset returns via
MOABB's MotorImagery paradigm -- the same order your cached X arrays'
channel axis is in. Run this and diff the output against lsp.py's
DATASET_CHANNEL_NAMES to confirm (or fix) the assumption flagged in
lsp.py's own comments.

Uses return_epochs=True specifically because that gives back an MNE
Epochs object with a .ch_names attribute in the exact order used for
the array -- more reliable than reconstructing the order by hand from
raw.pick_types() logic.

Usage:
    python check_channel_names.py
"""

from moabb.datasets import BNCI2014_001, BNCI2014_004, PhysionetMI
from moabb.paradigms import MotorImagery

# Same fmin/fmax as the fixed data_loader.py -- doesn't affect which
# channels come back, but keeping it consistent avoids any doubt.
paradigm = MotorImagery(resample=250.0, fmin=0.5, fmax=48.0)

DATASETS_TO_CHECK = [
    ("bci_iv_2a", BNCI2014_001),
    ("bci_iv_2b", BNCI2014_004),
    ("physionet", PhysionetMI),
]

for name, dataset_cls in DATASETS_TO_CHECK:
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    dataset = dataset_cls()
    epochs, y, meta = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)
    ch_names = epochs.ch_names
    print(f"n_channels: {len(ch_names)}")
    print(f"channel order: {ch_names}")
    print(f"\n--- paste-ready for lsp.py's DATASET_CHANNEL_NAMES ---")
    print(f'    "{name}": {ch_names!r},')
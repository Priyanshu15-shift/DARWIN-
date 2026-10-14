"""
diagnose_bci_iv_2b.py

Figures out exactly why paradigm.get_data() rejects BNCI2014_004
(bci_iv_2b) as "not valid for paradigm" after the moabb upgrade, and
checks whether the LeftRightImagery paradigm (built specifically for
2-class motor imagery datasets) works as a substitute.
"""

from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery, LeftRightImagery

dataset = BNCI2014_004()

print("=== Basic dataset properties ===")
print(f"dataset.code: {dataset.code}")
print(f"dataset.paradigm: {dataset.paradigm}")
print(f"dataset.event_id: {dataset.event_id}")

paradigm = MotorImagery(resample=250.0, fmin=0.5, fmax=48.0)
print(f"\n=== MotorImagery paradigm properties ===")
print(f"paradigm.events: {paradigm.events}")
print(f"is_valid(dataset): {paradigm.is_valid(dataset)}")

print(f"\n=== Trying LeftRightImagery instead (built for 2-class datasets) ===")
lr_paradigm = LeftRightImagery(resample=250.0, fmin=0.5, fmax=48.0)
print(f"is_valid(dataset): {lr_paradigm.is_valid(dataset)}")

if lr_paradigm.is_valid(dataset):
    print("\nLeftRightImagery accepts this dataset. Trying to actually pull subject 1...")
    epochs, y, meta = lr_paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)
    print(f"SUCCESS: {len(epochs)} epochs, channels: {epochs.ch_names}, "
          f"labels: {set(y)}")
else:
    print("\nLeftRightImagery ALSO rejects this dataset -- the problem is "
          "deeper than paradigm class choice. Paste this script's full "
          "output back for further diagnosis.")
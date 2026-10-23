# Bolero Isolation Lab

A standalone first-stage experiment for Bolero. It estimates relative depth, creates a ResNet-50 Grad-CAM importance map, runs DETR object detection, and builds an importance-seeded grid region proposal constrained by RGB/depth boundary evidence. Optional GrabCut refinement tests a simple pixel-level boundary baseline.

This is intentionally isolated from the other Bolero notebooks and output folders. It does not reconstruct images yet.

## What is included

- `notebooks/01_depth_subject_isolation.ipynb`: complete self-contained notebook.
- `requirements.txt`: Python dependencies.
- `setup_windows.ps1`: convenience setup for Windows/VS Code.
- `outputs/bolero_isolation/`: created by the notebook and cleared at the start of each `Run All`.
- `annotations/ground_truth_masks/`: manual masks created with the notebook's optional annotation tool; these persist across runs.

## Requirements

- Windows 10/11 or another compatible Python environment.
- CPython **3.11.9** recommended.
- Existing Imagenette 2-160 validation folder with class subfolders, such as `n01440764`, `n02102040`, etc.
- Internet access on the first run to download pretrained model checkpoints; later runs can use the local model cache.
- Several GB of storage for Python packages and model checkpoints. Inference can run on CPU, but it is slower.

## Dataset path

The notebook first checks this path from the existing setup:

`F:\DeepLearningTest\clip_corruption_calibration_repo\clip_corruption_calibration\data\imagenette2-160\val`

If it does not exist, it checks:

`<project folder>\data\imagenette2-160\val`

Change `DATASET_ROOT` in Cell 1 if your dataset is elsewhere. The data is not bundled into this project ZIP.

## Windows / VS Code setup

1. Extract this folder and open **the `bolero_isolation_lab` folder itself** in VS Code.
2. Ensure Python 3.11.9 is installed and the VS Code Python and Jupyter extensions are enabled.
3. Open a PowerShell terminal in this folder and run:

   ```powershell
   .\setup_windows.ps1
   ```

4. Open `notebooks/01_depth_subject_isolation.ipynb`.
5. Select the kernel **Python 3.11.9 (Bolero Isolation)** (the `.venv` environment).
6. Run all cells from the top.

If PowerShell blocks script execution, you can run the commands in `setup_windows.ps1` manually. If `python` is not recognized, use the full path to your Python 3.11.9 executable when creating the virtual environment. If Windows Application Control blocks a native package such as NumPy, do not disable security policy; use an approved Python installation/environment or an approved remote notebook runtime.

### GPU note

`requirements.txt` installs compatible PyTorch packages through pip, but a CUDA-capable build must match your GPU and driver. If CUDA is not available, the notebook falls back to CPU with `DEVICE_SETTING = "auto"`. For NVIDIA GPU-specific setup, use the official PyTorch installation selector: https://pytorch.org/get-started/locally/ .

## Output and safety behavior

- The notebook clears only `outputs/bolero_isolation/` at the start of a new run. Change `OUTPUT_TAG` in Cell 1 to keep runs in separate folders.
- It never removes the dataset or Hugging Face model cache.
- Manual polygon masks are stored separately under `annotations/ground_truth_masks/` and survive output cleaning.
- Original RGB images are stored separately from overlays and masks. Display overlays are never fed into the classifier or detector.

## Method notes

- ResNet-50 uses ImageNet-1K weights. For spatial correspondence, its full-frame transform preserves the entire image without the conventional center crop; therefore these exploratory scores should not be compared directly with standard ImageNet validation metrics.
- Depth Anything V2 Small provides a relative depth estimate, not physical distance. The pseudo-thermal palette is visual only.
- DETR ResNet-50 is a COCO object detector. It does not support every Imagenette class. Missing detections are retained as a limitation, not silently treated as proof that no subject exists.
- The grid starts at 20 x 20 cells and must remain at least 10 x 10. Selected cells are grown from class-importance seeds while discouraging transitions across strong RGB/depth boundaries. Optional GrabCut is a classical refinement baseline, not a learned semantic segmentation model.
- Automated masks are proposals, not ground truth. Use `annotate_mask(item_id)` and `evaluate_manual_masks()` to collect manual reference masks and estimate IoU/Dice before making claims about boundary accuracy.

## Current stage / out of scope

This version covers subject-isolation diagnostics only. Generative reconstruction, local VLM prompting, inpainting, and corrupted-image recovery are intentionally deferred until the isolation stage is evaluated and improved.

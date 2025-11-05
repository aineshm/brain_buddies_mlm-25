<!-- Copilot guidance for the brain_buddies_mlm-25 repository -->
# Copilot instructions — brain_buddies_mlm-25

This repo contains exploratory code, segmentation baselines, parsers, and slides used during the Machine Learning Marathon 2025. Use the notes below to make targeted, low-risk contributions.

1) Big picture / why this repo exists
- Purpose: provide EDA, baseline segmentation methods, and utilities for the Cellular State Image Analysis challenge. The main deliverables are reproducible segmentation experiments, visualization code, and annotation parsing tools.

2) Major components
- `eda_slides_code/` — the working code for EDA and segmentation. Key scripts:
  - `cell_segmentation_baseline.py` — multiple baseline segmentation methods and evaluation.
  - `improved_cell_segmentation.py` — enhanced algorithms used in the final pipeline.
  - `final_cell_segmentation.py` — production-ready pipeline (read before changing anything that affects outputs).
  - `xml_shape_parser.py` — annotation parsing (central to converting XML annotations to masks).
  - `shape_analysis_examples.py`, `quick_frame_viewer.py`, `interactive_frame_explorer.py` — visualization and analysis helpers.
- Top-level docs: `README.md`, `TODO.md`, and `eda_slides_code/README.md` and `SEGMENTATION_GUIDE.md` explain workflows and evaluation.

3) Data & shape expectations
- Input images are multi-frame TIFFs (typically shape ~ (frames, H, W)). All segmentation code assumes grayscale frames.
- XML annotations follow an object-per-frame schema (ellipses, polygons, polylines). Use `XMLShapeParser` in `xml_shape_parser.py` to convert to masks.
- Frame indices used across code: examples reference frames 0, 5, 26. Many evaluation functions index frames by integer keys from parsed XML.

4) Developer workflows & commands
- Run baseline segmentation / quick experiments:
  - `python eda_slides_code/cell_segmentation_baseline.py` (opens visualizations; script class-based — use from CLI or import as a module)
  - `python eda_slides_code/improved_cell_segmentation.py`
  - `python eda_slides_code/final_cell_segmentation.py`
- For reproducible EDA and slides, inspect `PROJECT_SUMMARY.md` and `EDA Presentation.pdf` in the repo root.

5) Project-specific conventions and gotchas
- NumPy 2.0 compatibility: TIF loading has a fallback to PIL when tifffile raises `newbyteorder`; preserve that pattern if you change I/O code.
- Annotation handling: code assumes ellipse annotations often define cells; many evaluation functions create binary masks from ellipses (`cy`, `cx`, `ry`, `rx`) — preserve field names when editing parser output.
- Evaluation metrics and IO: functions return dicts with keys `iou`, `precision`, `recall`, `f1_score`, `ground_truth_cells`, `predicted_cells`. Tests and reporting scripts expect those keys — maintain them or update consumers.
- Visual output naming: scripts save figures as `*_frame_{n}.png`. Keep naming consistent when adding new outputs to avoid breaking slide-generation scripts.

6) Common edit patterns (how to make safe changes)
- When changing parser output schema, update every call site in `eda_slides_code/*` (search for `['shape_type']`, `['cx']`, `['cy']`, `['rx']`, `['ry']`).
- For performance improvements, process frames one-at-a-time (current code is single-frame focused) and preserve memory pattern — do not load all frames into memory unless required.
- When changing thresholds or parameters, expose them as function arguments with sensible defaults and add small examples in `shape_analysis_examples.py`.

7) Testing and smoke checks
- Quick smoke test: run segmentation on a single frame and ensure saved visualization is produced. Example (from repo root):
  ```bash
  python eda_slides_code/cell_segmentation_baseline.py
  # or, in interactive mode
  python -c "from eda_slides_code.cell_segmentation_baseline import CellSegmentationBaseline; print('OK')"
  ```
- If changing file IO behavior, run the code that uses fallback logic (simulate `tifffile` error) or run on one of the provided sample TIF files in `eda_slides_code/exported_frames/`.

8) Files to open before editing
- `eda_slides_code/xml_shape_parser.py` — canonical annotation interface.
- `eda_slides_code/cell_segmentation_baseline.py` and `improved_cell_segmentation.py` — main algorithms and evaluation.
- `eda_slides_code/SEGMENTATION_GUIDE.md` and `README.md` — describes intended outputs, metrics, and saved filenames.

9) Example small tasks you can safely automate
- Add a `--frame` CLI argument to `cell_segmentation_baseline.py` to run comparison for a single frame and save a standardized PNG.
- Add a reproducible `run_all.sh` that runs the three main scripts and stores outputs into `exported_frames/` with timestamped folders.
- Add unit-tests (pytest) for `XMLShapeParser` parsing of ellipse and polygon entries (small JSON fixtures live in `eda_slides_code/parsed_shapes.json`).

10) Questions to clarify with maintainers
- Preferred output directories and naming conventions for slides and final exported images.
- Whether to standardize on a single TIF-loading path (tifffile vs PIL fallback) or keep both for portability.

If you want, I can now:
- Add this file to the repo (done). 
- Create a small smoke-test script that runs `cell_segmentation_baseline` on one sample frame and saves the comparison image. Say the word and I'll implement it.

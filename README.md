# UltraLabel

UltraLabel is a framework for propagating reference-frame annotations in ultrasound videos. It combines frozen visual features, image-guided feature upsampling, coarse-to-fine point tracking, and contour reconstruction.

Additional components and pretrained checkpoints will be released progressively.

## Installation

```bash
python -m pip install -e .
ultralabel validate-config --config configs/default.yaml
```

Python 3.10 or later is required. Optional dependencies are grouped as `vision`, `qc`, `demo`, and `dev` in `pyproject.toml`.

## Project structure

```text
configs/                  Method and deployment parameters
ultralabel/
  data/                   Frame loading, sampling, coordinate transforms
  models/extractor/       Feature extraction and upsampling interfaces
  models/tracker/         Similarity operations and tracking interfaces
  reconstruction/         Contour and registration interfaces
  quality/                Image-quality descriptors and classifier setup
  evaluation/             Segmentation and point metrics
  optimization/           Student optimizer and learning-rate schedule
  pipeline.py             Component orchestration
demo/                     Video annotation interface
tests/                    Utility and API checks
```

## Configuration

Default method and deployment settings are defined in `configs/default.yaml`.

## Working with data

Input frame directories contain PNG, JPEG, or TIFF images. Point coordinates use `(x, y)`.

```bash
ultralabel inspect-frames /path/to/frame_directory
```

See `examples/prepare_reference.py` for preparing reference points from a mask.

## Interactive interface

See [demo/README.md](demo/README.md) for local video playback, reference-point editing, and session export.

## Citation and license

Citation metadata is provided in `CITATION.cff`. See `LICENSE` for terms.

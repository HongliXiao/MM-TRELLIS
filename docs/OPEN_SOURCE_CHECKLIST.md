# Open-source release checklist

## Already cleaned in this package

- Removed `__pycache__` and `*.pyc` files.
- Replaced hard-coded `/workspace/...` paths in `inference.py` with CLI arguments.
- Changed the default TRELLIS checkpoint to `microsoft/TRELLIS-image-large`.
- Added README, input-format docs, setup script, citation file, gitignore, and example run script.
- Fixed `--lrs-axis-angle` parsing so multiple LR triples can be passed from the command line.

## Before pushing to GitHub

- [ ] Confirm the top-level license and third-party license compatibility.
- [ ] Confirm whether `examples_input/` data can be redistributed publicly.
- [ ] Decide whether large example assets should use Git LFS or be hosted externally.
- [ ] Verify installation on a clean Linux CUDA machine.
- [ ] Run the example command and upload one expected output screenshot/GIF to the README.
- [ ] Add links to model checkpoints if you host any MM-TRELLIS-specific checkpoints.
- [ ] Open a release tag after the README command has been tested.

## Minimal smoke checks

```bash
python -m py_compile inference.py
python inference.py --help
```

Full inference requires CUDA, TRELLIS dependencies, and pretrained weights.

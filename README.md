# Orpheus

This repository provides the data generation and model training scripts for

- Cube detection
- Cube localisation

## Cube Localisation Finetuning

New PyTorch scaffold scripts are available in `src/cube_localisation`:

- `dataset.py`: PNG loading + filename label parsing + spatial train/val/test split by random workplate region.
- `model.py`: pretrained backbone factory (`resnet18` default) with regression head.
- `train.py`: finetuning script with TensorBoard logging and checkpointing.
- `evaluate.py`: validation/test evaluation script using the same spatial split logic.

Install dependencies:

```bash
pip install -r src/cube_localisation/requirements.txt
```

Train:

```bash
python src/cube_localisation/train.py --epochs 40 --batch-size 32
```

Evaluate best checkpoint on test split:

```bash
python src/cube_localisation/evaluate.py --checkpoint runs/cube_localisation/<run_name>/checkpoints/best.pt --split test
```

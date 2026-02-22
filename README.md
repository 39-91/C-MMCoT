# C-MMCOT: Multimodal Chain-of-Thought Reasoning Using CLIP Features

A high school student's implementation of enhanced multimodal reasoning for ScienceQA.

## ⚠️ About This Project

This is a student learning project developed during high school. The code is shared as-is for reference purposes. Currently has limited maintenance capacity.


## Model Version

⚠️ **Important**: This implementation uses the **223M (smallest) MMCoT model** (`mm-cot-base-rationale`). This was chosen due to GPU memory constraints during development on a single consumer GPU.

If you want to use larger models, the code should work with `mm-cot-large-rationale`, but this has not been tested.

## What's Included

Modified files for improved MMCoT Stage 1 (rationale generation):
- `main.py` - Training and evaluation script
- `test_load_clip.py` - Visual feature loading test
- `utils_evaluate.py` - Evaluation metrics
- `utils_data.py` - Data loading utilities
- `requirements.txt` - Dependencies

Pre-trained weights available at: https://huggingface.co/YC111yc/C-MMCoT/tree/main

## Quick Start

### 1. Setup Base MMCoT Project

```bash
# Clone original MMCoT
git clone https://github.com/lupantech/mm-cot.git
cd mm-cot

# Install dependencies
pip install -r requirements.txt
```

### 2. Replace Files

```bash
# Copy improved files from this repository
cp main.py {your-mmcot-dir}/
cp test_load_clip.py {your-mmcot-dir}/
cp utils_evaluate.py {your-mmcot-dir}/
```

### 3. Download Base Model

```bash
# Download the 223M base model
huggingface-cli download cooelf/mm-cot-base-rationale
```

Place in: `./models/mm-cot-base-rationale/`

### 4. Download Pre-trained Weights

```bash
huggingface-cli download YC111yc/C-MMCoT
```

Place in: `./models/C-MMCoT/`

### 5. Download Data & Features

Follow the original MMCoT setup:
- ScienceQA dataset: https://github.com/lupantech/ScienceQA
- Vision features: https://huggingface.co/cooelf/vision_features

## Usage

### Inference with Pre-trained Weights

```bash
python main.py \
    --evaluate_dir ./models/C-MMCoT \
    --data_root ./data/scienceqa/ \
    --img_type clip \
    --final_eval
```

### Training from Scratch

```bash
python main.py \
    --data_root ./data/scienceqa/ \
    --model ./models/mm-cot-base-rationale \
    --img_type clip \
    --epoch 50 \
    --lr 5e-5 \
    --bs 2 \
    --output_dir ./experiments/
```

### Test Visual Features

```bash
python test_load_clip.py
```

## Key Differences from Original MMCoT

- Uses CLIP visual features with optimized fusion
- Improved evaluation pipeline
- Only Stage 1 fine-tuned (Stage 2 uses original weights)

## Known Limitations

- Only Stage 1 training implemented
- Pre-extracted visual features required
- Single GPU training only
- Only tested with 223M model
- Limited maintenance capacity

## Troubleshooting

### PyTorch 2.6+ Issue

If you encounter `UnpicklingError` when loading checkpoints with PyTorch 2.6+:

In `transformers/trainer.py`, find `_load_rng_state()` and change:
```python
torch.load(rng_file)
→
torch.load(rng_file, weights_only=False)
```

⚠️ Only use with trusted checkpoints.

## References

- Original MMCoT: https://github.com/lupantech/mm-cot
- ScienceQA: https://github.com/lupantech/ScienceQA
- Weights: https://huggingface.co/YC111yc/C-MMCoT

## Citation

```bibtex
@article{cmmcot2025,
  title={C-MMCOT: Multimodal Chain-of-Thought Reasoning Using CLIP Features},
  journal={Applied and Computational Engineering},
  volume={176},
  year={2025},
  doi={10.54254/2755-2721/2025.25822}
}
```

## Acknowledgments

- Original MMCoT authors and team
- ScienceQA dataset creators
- Hugging Face for model hosting

## License

MIT License

## Publication

**C-MMCOT: Multimodal Chain-of-Thought Reasoning Using CLIP Features**
- Published: Applied and Computational Engineering Vol.176, August 5, 2025
- DOI: https://doi.org/10.54254/2755-2721/2025.25822

---

**Note**: This is a student project shared for educational purposes. Issues and PRs may have delayed responses.

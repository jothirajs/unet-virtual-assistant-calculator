# U-Net Virtual Assistant Calculator

A Streamlit app for configuring a 2D U-Net-style segmentation architecture and estimating parameter count, MACs, FLOPs, raw weight storage, and an analytical activation-tensor footprint. Includes per-layer reporting, an architecture-flow diagram, ablation comparisons, and CSV/JSON exports.

## Features
- Configure input dimensions/channels, output classes, downsampling levels, initial filters, kernel size, upsampling, skip connections, batch normalization, bias, precision, and batch size.
- Inspect per-layer shapes, parameters, and MAC estimates.
- Compare baseline and ablated variants.
- Export CSV layer analysis and JSON configuration/results.
- Run unit tests for calculation behavior.

## Assumptions
- Same-padding 2D U-Net-style implementation; not an exact reproduction of the original 2015 U-Net paper's valid-padding/cropping implementation.
- Two convolutions in each encoder/decoder block and bottleneck.
- Max pooling between encoder levels; transposed-convolution or bilinear upsampling in decoder.
- Skip connections concatenate matching encoder feature maps when enabled.
- Output projection is a 1×1 convolution producing class logits.
- Batch normalization adds two trainable parameters per normalized channel. It is not applied to the final output projection.
- Activations, dropout, softmax, pooling arithmetic, normalization arithmetic, concatenation, and interpolation arithmetic are excluded from MAC totals.

## Metric definitions and limitations
**Parameters:** convolution/transposed-convolution weights and enabled biases, plus trainable batch-normalization scale/offset values.

**MACs:** Conv2D estimate = output H × output W × output channels × input channels × kernel H × kernel W. Transposed-convolution MACs use the same output-based dense-equivalent estimate; overlap and boundary effects are not modeled. Pooling, normalization, activation, concatenation, and bilinear interpolation arithmetic are excluded.

**FLOPs:** reported as 2 × MACs. Other conventions can yield different totals.

**Weight storage:** parameter count × bytes per parameter, converted to MiB. This is raw parameter storage, not total runtime GPU memory.

**Activation tensor footprint:** sum of listed layer-output tensor sizes plus the input tensor for the selected batch size and precision. It is not peak GPU memory and is not a training-memory estimate. Actual memory depends on tensor lifetimes, framework workspaces, gradients, optimizer states, allocator behavior, and inference/training mode.

**Important:** complexity does not establish segmentation quality. This app does not measure runtime latency or actual GPU memory. For empirical results, build/profile the architecture in a deep-learning framework and evaluate it on a documented dataset with suitable segmentation metrics.

## Install
Use Python 3.10+.

```bash
python -m venv .venv
```
Activate it (Windows PowerShell: `.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`), then:

```bash
python -m pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Test
```bash
python -m unittest discover -s tests -v
```

## Upload to GitHub
1. Create/open your public repository.
2. Extract this ZIP.
3. Upload the **contents** of the extracted folder to the repository root. `app.py`, `unet_tool/`, and `tests/` must be at the top level.
4. Commit the files.
5. Run locally using the commands above, or deploy using Streamlit Community Cloud with `app.py` as the entry point.

## Repository layout
```text
unet-virtual-assistant-calculator/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── unet_tool/
│   ├── __init__.py
│   ├── config.py
│   └── engine.py
└── tests/
    └── test_engine.py
```

Choose an appropriate license before public distribution. The included MIT License is a permissive starting point.

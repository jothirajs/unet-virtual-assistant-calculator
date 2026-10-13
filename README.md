# U-Net Virtual Assistant Calculator

A Streamlit application for interactively configuring a 2D U-Net-style segmentation architecture and estimating its model complexity. It is intended for teaching, architecture exploration, and research planning.

## Features

- Interactive configuration of input dimensions/channels, output classes, downsampling depth, initial filters, convolution kernel, upsampling method, skip connections, batch normalization, bias, precision, and batch size.
- Layer-by-layer output shapes, trainable parameters, and MAC estimates.
- FLOPs, raw model-weight storage, and an analytical activation-tensor footprint.
- Baseline-versus-ablation comparison.
- CSV and JSON export.
- Automated unit tests through GitHub Actions.

## Architecture assumptions

- Same-padding 2D U-Net-style implementation. It is **not** an exact reproduction of the original 2015 U-Net paper's valid-padding/cropping implementation.
- Two convolutions in each encoder block, bottleneck, and decoder block.
- Max pooling between encoder levels.
- Decoder uses either learnable transposed convolution or parameter-free bilinear upsampling.
- Skip connections concatenate matching encoder feature maps when enabled.
- Output projection is a 1×1 convolution that produces class logits.
- Batch normalization contributes two trainable parameters per normalized channel and is not applied to the output projection.

## Metric definitions and limitations

**Parameters.** Convolution/transposed-convolution weights and enabled biases, plus trainable batch-normalization scale and offset parameters.

**MACs.** Conv2D estimate = output height × output width × output channels × input channels × kernel height × kernel width. Transposed-convolution MACs use an output-based dense-equivalent estimate; overlap and boundary effects are not modeled. Pooling, normalization, activation functions, concatenation, and bilinear interpolation arithmetic are excluded.

**FLOPs.** Reported as 2 × MACs. Other conventions can yield different totals.

**Weight storage.** Parameter count × bytes per parameter, converted to MiB (1 MiB = 1,048,576 bytes). This is raw parameter storage, not total runtime GPU memory.

**Activation tensor footprint.** Sum of listed layer-output tensor sizes plus the input tensor for the selected batch size and precision. It is not peak GPU memory and is not a training-memory estimate. Actual memory depends on tensor lifetimes, framework workspaces, gradients, optimizer states, allocator behavior, and inference/training mode.

**Research caution.** Complexity estimates do not establish segmentation quality, clinical utility, or statistical significance. This app does not measure runtime latency or actual GPU memory. For empirical results, implement/profile the architecture in a deep-learning framework and evaluate it on a documented dataset using appropriate segmentation metrics.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run locally

From the repository root:

```bash
streamlit run app.py
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

A GitHub Actions workflow also runs the unit tests on pushes and pull requests.

## Deploy

The app can be deployed through Streamlit Community Cloud by selecting this repository, the `main` branch, and `app.py` as the application entry point.

## Repository structure

```text
unet-virtual-assistant-calculator/
├── app.py
├── config.py                 # legacy root-level module; app uses unet_tool/
├── engine.py                 # legacy root-level module; app uses unet_tool/
├── requirements.txt
├── README.md
├── LICENSE
├── unet_tool/
│   ├── __init__.py
│   ├── config.py
│   └── engine.py
├── tests/
│   └── test_engine.py
└── .github/
    └── workflows/
        └── tests.yml
```

The root-level `config.py`, `engine.py`, `__init__.py`, and `test_engine.py` came from the initial upload. The working application imports the package under `unet_tool/`; the package is now included explicitly.

## License

The repository includes an MIT License. Review it against your intended distribution and institutional requirements before release.

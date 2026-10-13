# U-Net Virtual Assistant Calculator

A Streamlit application for interactively configuring a 2D U-Net-style segmentation architecture and estimating its model complexity. Intended for teaching, architecture exploration, and research planning.

## Features

- Configure input dimensions/channels, output classes, downsampling depth, convolution kernel, upsampling, skip connections, batch normalization, bias, precision, and batch size.
- Edit the number of convolution layers and output-filter count independently for every encoder, bottleneck, and decoder stage.
- Recalculate parameters, MACs, FLOPs, weight storage, and activation-tensor footprint whenever the configuration changes.
- Inspect per-layer input/output channels, output tensor shapes, parameters, and MAC estimates.
- Compare baseline and ablated architectures.
- Export layer analysis as CSV and the complete architecture configuration and analysis as JSON.
- Run automated unit tests on GitHub Actions.

## Architecture assumptions

- Same-padding 2D U-Net-style implementation. This is **not** an exact reproduction of the original 2015 U-Net paper's valid-padding/cropping implementation.
- Encoder, bottleneck, and decoder stages support 1–4 convolution layers in the Streamlit editor. Every convolution has its own editable output-filter count.
- Max pooling is used between encoder levels.
- Decoder uses either learnable transposed convolution or parameter-free bilinear upsampling.
- Skip connections concatenate matching encoder feature maps when enabled.
- Output projection is a 1×1 convolution that produces class logits.
- Batch normalization contributes two trainable parameters per normalized channel and is not applied to the output projection.

## Metric definitions and limitations

**Parameters.** Convolution/transposed-convolution weights and enabled biases, plus trainable batch-normalization scale and offset parameters.

**MACs.** Conv2D estimate = output height × output width × output channels × input channels × kernel height × kernel width. Transposed-convolution MACs use an output-based dense-equivalent estimate; overlap and boundary effects are not modeled. Pooling, normalization, activations, concatenation, and bilinear interpolation arithmetic are excluded.

**FLOPs.** Reported as 2 × MACs. Other conventions can yield different totals.

**Weight storage.** Parameter count × bytes per parameter, converted to MiB (1 MiB = 1,048,576 bytes). This is raw parameter storage, not total runtime GPU memory.

**Activation tensor footprint.** Sum of listed layer-output tensor sizes plus the input tensor for the selected batch size and precision. It is not peak GPU memory or a training-memory estimate. Actual memory depends on tensor lifetimes, framework workspaces, gradients, optimizer states, allocator behavior, and inference/training mode.

**Research caution.** Complexity estimates do not establish segmentation quality, clinical utility, or statistical significance. This app does not measure runtime latency or actual GPU memory. For empirical results, implement/profile the architecture in a deep-learning framework and evaluate it on a documented dataset using suitable segmentation metrics.

## Install and run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies and launch:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions checks Python syntax and runs the unit tests on pushes and pull requests.

## Deploy as a website using Streamlit Community Cloud

Deployment needs to be authorized from your own Streamlit account.

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with GitHub.
3. Select the repository `jothirajs/unet-virtual-assistant-calculator`, branch `main`, and main file path `app.py`.
4. Deploy and wait for the build to finish.
5. Test the layer editor, recalculated metrics, ablation tab, CSV/JSON downloads, and architecture diagram.

The repository includes `packages.txt` for the Graphviz system package used by the architecture diagram and `requirements.txt` for Python dependencies.

## Repository structure

```text
unet-virtual-assistant-calculator/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── LICENSE
├── .gitignore
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

## License

The repository includes an MIT License. Review it against your intended distribution and institutional requirements before release.

from dataclasses import replace
from typing import Any

from .config import UNetConfig


def _conv_params(cin: int, cout: int, kernel: int, use_bias: bool) -> int:
    return kernel * kernel * cin * cout + (cout if use_bias else 0)


def _conv_macs(height: int, width: int, cin: int, cout: int, kernel: int) -> int:
    return height * width * cout * cin * kernel * kernel


def analyze_unet(config: UNetConfig) -> dict[str, Any]:
    """Estimate U-Net complexity from the configured per-layer filter lists."""
    config.validate()
    layers: list[dict[str, Any]] = []
    total_params = 0
    total_macs = 0
    activation_elements = (
        config.input_height * config.input_width * config.input_channels * config.batch_size
    )

    def add_layer(name, kind, h, w, cin, cout, params=0, macs=0, note=""):
        nonlocal total_params, total_macs, activation_elements
        elements = h * w * cout * config.batch_size
        activation_elements += elements
        total_params += params
        total_macs += macs
        layers.append({
            "layer": name,
            "type": kind,
            "output_shape": f"{h}×{w}×{cout}",
            "height": h,
            "width": w,
            "input_channels": cin,
            "output_channels": cout,
            "kernel": "",
            "parameters": int(params),
            "MACs": int(macs),
            "output_elements": int(elements),
            "notes": note,
        })

    def add_conv(name, h, w, cin, cout, kernel=None):
        kernel = kernel or config.conv_kernel
        params = _conv_params(cin, cout, kernel, config.use_bias)
        macs = _conv_macs(h, w, cin, cout, kernel)
        add_layer(name, "Conv2D", h, w, cin, cout, params, macs)
        layers[-1]["kernel"] = f"{kernel}×{kernel}"
        if config.batch_norm:
            bn_params = 2 * cout
            total_params_local = bn_params
            nonlocal total_params
            total_params += total_params_local
            layers[-1]["parameters"] += bn_params
            layers[-1]["notes"] = "Includes batch-normalization trainable parameters"
        return cout

    h, w, cin = config.input_height, config.input_width, config.input_channels
    skips: list[tuple[int, int, int]] = []
    encoder_specs = config.effective_encoder_filters()
    bottleneck_specs = config.effective_bottleneck_filters()
    decoder_specs = config.effective_decoder_filters()

    for level, filters_for_stage in enumerate(encoder_specs):
        for idx, cout in enumerate(filters_for_stage, start=1):
            cin = add_conv(f"Encoder {level + 1} Conv {idx}", h, w, cin, cout)
        skips.append((h, w, cin))
        add_layer(
            f"Encoder {level + 1} MaxPool", "MaxPool2D",
            h // config.pool_kernel, w // config.pool_kernel, cin, cin,
            note="Pooling arithmetic excluded from MACs",
        )
        h //= config.pool_kernel
        w //= config.pool_kernel

    for idx, cout in enumerate(bottleneck_specs, start=1):
        cin = add_conv(f"Bottleneck Conv {idx}", h, w, cin, cout)

    for level in reversed(range(config.levels)):
        target_h, target_w, skip_channels = skips[level]
        target_filters = skip_channels

        if config.upsampling == "transpose":
            params = (
                config.upsample_kernel ** 2 * cin * target_filters
                + (target_filters if config.use_bias else 0)
            )
            macs = _conv_macs(
                target_h, target_w, cin, target_filters, config.upsample_kernel
            )
            add_layer(
                f"Decoder {level + 1} TransposedConv", "ConvTranspose2D",
                target_h, target_w, cin, target_filters, params, macs,
                note="Dense-equivalent MAC estimate; overlap/boundary effects excluded",
            )
            layers[-1]["kernel"] = f"{config.upsample_kernel}×{config.upsample_kernel}"
            cin = target_filters
        else:
            add_layer(
                f"Decoder {level + 1} BilinearUpsample", "BilinearUpsample",
                target_h, target_w, cin, cin,
                note="Interpolation arithmetic excluded from MACs",
            )

        if config.skip_connections:
            joined_channels = cin + skip_channels
            add_layer(
                f"Decoder {level + 1} SkipConcat", "Concatenate",
                target_h, target_w, joined_channels, joined_channels,
                note=f"Concatenates decoder ({cin}) and encoder ({skip_channels}) channels",
            )
            cin = joined_channels

        for idx, cout in enumerate(decoder_specs[config.levels - 1 - level], start=1):
            cin = add_conv(f"Decoder {level + 1} Conv {idx}", target_h, target_w, cin, cout)

    params = _conv_params(cin, config.output_classes, 1, config.use_bias)
    macs = _conv_macs(
        config.input_height, config.input_width, cin, config.output_classes, 1
    )
    add_layer(
        "Output Projection", "Conv2D 1×1",
        config.input_height, config.input_width, cin, config.output_classes,
        params, macs, note="Output logits; activation excluded",
    )
    layers[-1]["kernel"] = "1×1"

    weight_bytes = total_params * config.precision_bytes
    activation_bytes = activation_elements * config.precision_bytes
    return {
        "config": config.to_dict(),
        "summary": {
            "parameters": int(total_params),
            "trainable_parameters": int(total_params),
            "MACs": int(total_macs),
            "FLOPs_2x_MACs": int(2 * total_macs),
            "weight_memory_bytes": int(weight_bytes),
            "weight_memory_MiB": weight_bytes / (1024 ** 2),
            "activation_tensor_footprint_bytes": int(activation_bytes),
            "activation_tensor_footprint_MiB": activation_bytes / (1024 ** 2),
            "activation_footprint_definition": (
                "Sum of listed layer-output tensors plus input; not peak GPU memory"
            ),
            "MAC_definition": (
                "Conv/transposed-conv estimates; pooling, BN arithmetic, activations, "
                "concat, and interpolation excluded"
            ),
            "layer_count": len(layers),
        },
        "layers": layers,
    }


def apply_ablation(config: UNetConfig, ablation: str) -> UNetConfig:
    """Return a copied configuration with one architecture change."""
    if ablation == "Baseline":
        return replace(config)
    if ablation == "Without skip connections":
        return replace(config, skip_connections=False)
    if ablation == "Without batch normalization":
        return replace(config, batch_norm=False)
    if ablation == "Bilinear upsampling":
        return replace(config, upsampling="bilinear")
    if ablation == "Without convolution biases":
        return replace(config, use_bias=False)
    raise ValueError(f"Unknown ablation: {ablation}")


def compare_analyses(config: UNetConfig, ablations=None) -> list[dict[str, Any]]:
    if ablations is None:
        ablations = [
            "Baseline",
            "Without skip connections",
            "Without batch normalization",
            "Bilinear upsampling",
            "Without convolution biases",
        ]
    rows = []
    for name in ablations:
        summary = analyze_unet(apply_ablation(config, name))["summary"]
        rows.append({
            "variant": name,
            "parameters": summary["parameters"],
            "MACs": summary["MACs"],
            "FLOPs": summary["FLOPs_2x_MACs"],
            "weight_memory_MiB": round(summary["weight_memory_MiB"], 3),
            "activation_footprint_MiB": round(
                summary["activation_tensor_footprint_MiB"], 3
            ),
        })
    return rows

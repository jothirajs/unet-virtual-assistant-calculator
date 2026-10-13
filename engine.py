from dataclasses import replace
from typing import Any
from .config import UNetConfig

def _conv_params(cin, cout, kernel, use_bias):
    return kernel * kernel * cin * cout + (cout if use_bias else 0)

def _conv_macs(height, width, cin, cout, kernel):
    return height * width * cout * cin * kernel * kernel

def analyze_unet(config: UNetConfig) -> dict[str, Any]:
    """Analytical 2D U-Net estimate; definitions and limitations are documented in README."""
    config.validate()
    layers = []
    total_params = 0
    total_macs = 0
    activation_elements = config.input_height * config.input_width * config.input_channels * config.batch_size

    def add_layer(name, kind, h, w, cin, cout, params=0, macs=0, note=""):
        nonlocal total_params, total_macs, activation_elements
        elements = h * w * cout * config.batch_size
        activation_elements += elements
        total_params += params
        total_macs += macs
        layers.append({
            "layer": name, "type": kind, "output_shape": f"{h}×{w}×{cout}",
            "height": h, "width": w, "input_channels": cin, "output_channels": cout,
            "parameters": int(params), "MACs": int(macs), "output_elements": int(elements), "notes": note
        })

    h, w, cin = config.input_height, config.input_width, config.input_channels
    skips = []
    for level in range(config.levels):
        cout = config.base_filters * (2 ** level)
        for idx in (1, 2):
            p = _conv_params(cin, cout, config.conv_kernel, config.use_bias)
            m = _conv_macs(h, w, cin, cout, config.conv_kernel)
            add_layer(f"Encoder {level+1} Conv {idx}", "Conv2D", h, w, cin, cout, p, m)
            if config.batch_norm:
                bn = 2 * cout
                total_params += bn
                layers[-1]["parameters"] += bn
                layers[-1]["notes"] = "Includes batch-normalization trainable parameters"
            cin = cout
        skips.append((h, w, cout))
        add_layer(f"Encoder {level+1} MaxPool", "MaxPool2D", h // config.pool_kernel,
                  w // config.pool_kernel, cout, cout, note="Pooling arithmetic excluded from MACs")
        h //= config.pool_kernel
        w //= config.pool_kernel

    bottleneck = config.base_filters * (2 ** config.levels)
    for idx in (1, 2):
        p = _conv_params(cin, bottleneck, config.conv_kernel, config.use_bias)
        m = _conv_macs(h, w, cin, bottleneck, config.conv_kernel)
        add_layer(f"Bottleneck Conv {idx}", "Conv2D", h, w, cin, bottleneck, p, m)
        if config.batch_norm:
            bn = 2 * bottleneck
            total_params += bn
            layers[-1]["parameters"] += bn
            layers[-1]["notes"] = "Includes batch-normalization trainable parameters"
        cin = bottleneck

    for level in reversed(range(config.levels)):
        th, tw, skip_channels = skips[level]
        cout = config.base_filters * (2 ** level)
        if config.upsampling == "transpose":
            p = config.upsample_kernel ** 2 * cin * cout + (cout if config.use_bias else 0)
            m = _conv_macs(th, tw, cin, cout, config.upsample_kernel)
            add_layer(f"Decoder {level+1} TransposedConv", "ConvTranspose2D", th, tw,
                      cin, cout, p, m,
                      note="Dense-equivalent MAC estimate; overlap/boundary effects excluded")
            cin = cout
        else:
            add_layer(f"Decoder {level+1} BilinearUpsample", "BilinearUpsample", th, tw,
                      cin, cin, note="Interpolation arithmetic excluded from MACs")
        if config.skip_connections:
            joined = cin + skip_channels
            add_layer(f"Decoder {level+1} SkipConcat", "Concatenate", th, tw, joined, joined,
                      note=f"Concatenates decoder ({cin}) and encoder ({skip_channels}) channels")
            block_input = joined
        else:
            block_input = cin
        cin = block_input
        for idx in (1, 2):
            p = _conv_params(cin, cout, config.conv_kernel, config.use_bias)
            m = _conv_macs(th, tw, cin, cout, config.conv_kernel)
            add_layer(f"Decoder {level+1} Conv {idx}", "Conv2D", th, tw, cin, cout, p, m)
            if config.batch_norm:
                bn = 2 * cout
                total_params += bn
                layers[-1]["parameters"] += bn
                layers[-1]["notes"] = "Includes batch-normalization trainable parameters"
            cin = cout

    p = _conv_params(cin, config.output_classes, 1, config.use_bias)
    m = _conv_macs(config.input_height, config.input_width, cin, config.output_classes, 1)
    add_layer("Output Projection", "Conv2D 1×1", config.input_height, config.input_width,
              cin, config.output_classes, p, m, note="Output logits; activation excluded")

    weight_bytes = total_params * config.precision_bytes
    activation_bytes = activation_elements * config.precision_bytes
    return {
        "config": config.to_dict(),
        "summary": {
            "parameters": int(total_params), "trainable_parameters": int(total_params),
            "MACs": int(total_macs), "FLOPs_2x_MACs": int(2 * total_macs),
            "weight_memory_bytes": int(weight_bytes), "weight_memory_MiB": weight_bytes / (1024 ** 2),
            "activation_tensor_footprint_bytes": int(activation_bytes),
            "activation_tensor_footprint_MiB": activation_bytes / (1024 ** 2),
            "activation_footprint_definition": "Sum of listed layer-output tensors plus input; not peak GPU memory",
            "MAC_definition": "Conv/transposed-conv estimates; pooling, BN arithmetic, activations, concat, interpolation excluded",
            "layer_count": len(layers),
        },
        "layers": layers,
    }

def apply_ablation(config: UNetConfig, ablation: str) -> UNetConfig:
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

def compare_analyses(config: UNetConfig, ablations=None):
    if ablations is None:
        ablations = ["Baseline", "Without skip connections", "Without batch normalization",
                     "Bilinear upsampling", "Without convolution biases"]
    rows = []
    for name in ablations:
        s = analyze_unet(apply_ablation(config, name))["summary"]
        rows.append({
            "variant": name, "parameters": s["parameters"], "MACs": s["MACs"],
            "FLOPs": s["FLOPs_2x_MACs"], "weight_memory_MiB": round(s["weight_memory_MiB"], 3),
            "activation_footprint_MiB": round(s["activation_tensor_footprint_MiB"], 3)
        })
    return rows

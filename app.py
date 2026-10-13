import json

import pandas as pd
import streamlit as st

from unet_tool import UNetConfig, analyze_unet, compare_analyses

st.set_page_config(
    page_title="U-Net Virtual Assistant Calculator",
    page_icon="🧠",
    layout="wide",
)
st.title("U-Net Virtual Assistant Calculator")
st.caption(
    "Interactive U-Net design workspace for layer-level architecture editing, "
    "complexity analysis, teaching, and research."
)
st.info(
    "Metrics are analytical estimates, not measured runtime benchmarks. "
    "Edits to layer filters and convolution counts recalculate the model immediately."
)


def layer_filter_editor(stage_label, default_filters, key_prefix):
    """Render an editable sequence of convolution output-channel counts."""
    count = st.slider(
        f"{stage_label}: number of convolution layers",
        min_value=1,
        max_value=4,
        value=min(max(len(default_filters), 1), 4),
        key=f"{key_prefix}_count",
        help="Add or remove convolution layers within this stage.",
    )
    filters = []
    cols = st.columns(min(count, 4))
    for idx in range(count):
        default_value = default_filters[idx] if idx < len(default_filters) else default_filters[-1]
        with cols[idx % len(cols)]:
            filters.append(
                int(
                    st.number_input(
                        f"Conv {idx + 1} filters",
                        min_value=1,
                        max_value=2048,
                        value=int(default_value),
                        step=8,
                        key=f"{key_prefix}_filters_{idx + 1}",
                    )
                )
            )
    return tuple(filters)


with st.sidebar:
    st.header("Input and model settings")
    height = st.selectbox("Input height", [64, 128, 256, 512], index=2)
    width = st.selectbox("Input width", [64, 128, 256, 512], index=2)
    in_ch = st.number_input("Input channels", 1, 16, 3)
    classes = st.number_input("Output classes", 1, 100, 2)
    levels = st.slider("Downsampling levels", 1, 5, 4)
    base = st.selectbox("Initial filters (default plan)", [8, 16, 32, 64], index=3)
    kernel = st.selectbox("Convolution kernel", [1, 3, 5], index=1)
    upsampling = st.selectbox("Upsampling", ["transpose", "bilinear"])
    skips = st.checkbox("Skip connections", True)
    batch_norm = st.checkbox("Batch normalization", False)
    bias = st.checkbox("Convolution biases", True)
    precision = st.selectbox("Precision", ["FP32", "FP16", "INT8"])
    batch_size = st.number_input("Batch size", 1, 64, 1)

st.subheader("Layer editor")
st.caption(
    "Set the number of convolutions and the output-filter count for every convolution. "
    "The output of each layer becomes the input-channel count of the next layer."
)

encoder_filters = []
with st.expander("Encoder layers", expanded=True):
    for level in range(levels):
        defaults = (base * (2 ** level),) * 2
        with st.container(border=True):
            st.markdown(f"**Encoder stage {level + 1}**")
            encoder_filters.append(
                layer_filter_editor(
                    f"Encoder {level + 1}",
                    defaults,
                    f"enc_{level + 1}",
                )
            )

with st.expander("Bottleneck layers", expanded=True):
    bottleneck_filters = layer_filter_editor(
        "Bottleneck",
        (base * (2 ** levels),) * 2,
        "bottleneck",
    )

decoder_filters = []
with st.expander("Decoder layers", expanded=True):
    for level in reversed(range(levels)):
        defaults = (base * (2 ** level),) * 2
        with st.container(border=True):
            st.markdown(f"**Decoder stage {level + 1}**")
            decoder_filters.append(
                layer_filter_editor(
                    f"Decoder {level + 1}",
                    defaults,
                    f"dec_{level + 1}",
                )
            )

bytes_per_value = {"FP32": 4, "FP16": 2, "INT8": 1}[precision]
config = UNetConfig(
    input_height=int(height),
    input_width=int(width),
    input_channels=int(in_ch),
    output_classes=int(classes),
    levels=int(levels),
    base_filters=int(base),
    conv_kernel=int(kernel),
    upsampling=upsampling,
    skip_connections=skips,
    batch_norm=batch_norm,
    use_bias=bias,
    precision_bytes=bytes_per_value,
    batch_size=int(batch_size),
    encoder_filters=tuple(encoder_filters),
    bottleneck_filters=tuple(bottleneck_filters),
    decoder_filters=tuple(decoder_filters),
)

try:
    result = analyze_unet(config)
except ValueError as err:
    st.error(f"Configuration error: {err}")
    st.stop()

summary = result["summary"]
st.subheader("Updated complexity estimate")
metric_cols = st.columns(4)
metric_cols[0].metric("Trainable parameters", f"{summary['parameters']:,}")
metric_cols[1].metric("MACs", f"{summary['MACs'] / 1e9:.3f} G")
metric_cols[2].metric("FLOPs (2 × MACs)", f"{summary['FLOPs_2x_MACs'] / 1e9:.3f} G")
metric_cols[3].metric("Weight storage", f"{summary['weight_memory_MiB']:.2f} MiB")
metric_cols2 = st.columns(2)
metric_cols2[0].metric(
    "Activation tensor footprint",
    f"{summary['activation_tensor_footprint_MiB']:.2f} MiB",
)
metric_cols2[1].metric("Layer count", str(summary["layer_count"]))
st.caption(
    "Activation footprint is the sum of listed layer-output tensor sizes plus the input; "
    "it is not peak GPU memory. FLOPs are defined as 2 × MACs."
)

tab1, tab2, tab3 = st.tabs(
    ["Layer-by-layer analysis", "Ablation comparison", "Export and assumptions"]
)
with tab1:
    df = pd.DataFrame(result["layers"])
    st.dataframe(
        df[
            [
                "layer",
                "type",
                "output_shape",
                "input_channels",
                "output_channels",
                "kernel",
                "parameters",
                "MACs",
                "notes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download layer analysis (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "unet_layer_analysis.csv",
        "text/csv",
    )

with tab2:
    comp = pd.DataFrame(compare_analyses(config))
    baseline_params = int(
        comp.loc[comp["variant"] == "Baseline", "parameters"].iloc[0]
    )
    comp["parameter_change_%"] = comp["parameters"].apply(
        lambda value: round((value - baseline_params) / baseline_params * 100, 2)
        if baseline_params
        else 0
    )
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.bar_chart(comp.set_index("variant")[["parameters"]])
    st.caption(
        "Complexity ablations do not establish segmentation accuracy or clinical utility."
    )

with tab3:
    payload = {
        "model": "Configurable 2D U-Net-style analytical estimate",
        "config": config.to_dict(),
        "summary": summary,
        "layers": result["layers"],
    }
    st.download_button(
        "Download full analysis (JSON)",
        json.dumps(payload, indent=2),
        "unet_analysis.json",
        "application/json",
    )
    st.json(
        {
            "configuration": config.to_dict(),
            "metric_definitions": {
                "MACs": summary["MAC_definition"],
                "FLOPs": "Reported as 2 × MACs.",
                "Weight storage": "Parameter count × bytes per parameter.",
                "Activation footprint": summary["activation_footprint_definition"],
            },
        }
    )

st.divider()
st.subheader("Architecture flow")
dot = (
    'digraph UNet { rankdir=LR; graph [bgcolor="transparent"]; '
    'node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#4C78A8"]; '
    f'input [label="Input\\n{height}×{width}×{in_ch}"]; '
    f'enc [label="Encoder\\n{levels} stages"]; '
    f'bn [label="Bottleneck\\n{len(bottleneck_filters)} convolutions"]; '
    f'dec [label="Decoder\\n{upsampling}"]; '
    f'out [label="Output logits\\n{classes} classes"]; '
    'input -> enc -> bn -> dec -> out; '
    + (
        'enc -> dec [style=dashed, label="skip connections", constraint=false]; '
        if skips
        else ""
    )
    + "}"
)
st.graphviz_chart(dot)
st.caption(
    "Same-padding 2D U-Net-style model. This is not an exact reproduction of the "
    "original valid-padding U-Net. Runtime, peak memory, and segmentation accuracy "
    "must be measured separately."
)

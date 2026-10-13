import json

import pandas as pd
import streamlit as st

from unet_tool import UNetConfig, analyze_unet, compare_analyses

st.set_page_config(
    page_title="U-Net Studio | Architecture Calculator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #17324d;
        --muted: #5f7185;
        --blue: #2867a8;
        --blue-soft: #eaf3fc;
        --line: #d9e5f0;
        --surface: #ffffff;
      }
      .stApp { background: #f5f8fc; color: var(--ink); }
      [data-testid="stHeader"] { background: rgba(245,248,252,.92); }
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
      h1, h2, h3 { color: #17324d; letter-spacing: -0.025em; }
      h1 { font-size: 2.05rem !important; font-weight: 750 !important; }
      h2 { font-size: 1.35rem !important; font-weight: 700 !important; }
      h3 { font-size: 1.05rem !important; font-weight: 700 !important; }
      p, label, .stMarkdown { color: #304a63; }
      [data-testid="stSidebar"] { background: #edf4fb; border-right: 1px solid #d9e5f0; }
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #17324d; }
      [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #dce7f1; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 2px 8px rgba(27,65,101,.035);
        min-height: 112px;
      }
      [data-testid="stMetricLabel"] { color: #5f7185; font-size: .82rem; }
      [data-testid="stMetricValue"] { color: #17324d; font-weight: 750; font-size: 1.55rem; }
      [data-testid="stTabs"] button { font-weight: 650; }
      [data-testid="stTabs"] button[aria-selected="true"] {
        color: #215f9e; border-bottom-color: #2b73b9;
      }
      div[data-testid="stExpander"] { background: #fff; border: 1px solid #dce7f1; border-radius: 12px; }
      div[data-testid="stDownloadButton"] button, div.stButton button {
        border-radius: 9px; font-weight: 650; border: 1px solid #b9d2e8;
      }
      div[data-testid="stDownloadButton"] button[kind="primary"],
      div.stButton button[kind="primary"] { background: #2867a8; color: #fff; }
      .hero {
        background: linear-gradient(115deg, #ffffff 0%, #f0f7ff 100%);
        border: 1px solid #d7e6f5; border-radius: 18px; padding: 24px 28px;
        margin-bottom: 18px; box-shadow: 0 5px 20px rgba(32,75,118,.04);
      }
      .eyebrow {
        text-transform: uppercase; color: #2867a8; font-size: .75rem;
        letter-spacing: .12em; font-weight: 800; margin-bottom: 7px;
      }
      .hero-title { color: #17324d; font-size: 2rem; line-height: 1.15; font-weight: 800; margin-bottom: 8px; }
      .hero-copy { color: #536b82; font-size: 1rem; max-width: 900px; line-height: 1.55; }
      .step-label {
        display: inline-block; background: #e5f1fd; color: #225f99; border-radius: 7px;
        padding: 4px 9px; font-size: .75rem; font-weight: 800; letter-spacing: .04em;
        margin-bottom: 8px;
      }
      .section-note {
        background: #eef6fd; border-left: 4px solid #6ca5d9; padding: 11px 14px;
        border-radius: 0 9px 9px 0; color: #355571; font-size: .9rem; margin: 8px 0 16px 0;
      }
      .small-muted { color: #667b8f; font-size: .82rem; }
      .status-pill {
        display: inline-block; border: 1px solid #cfe2f3; background: #f5faff;
        color: #2b6397; border-radius: 999px; padding: 5px 10px; font-size: .78rem; font-weight: 700;
      }
      @media (max-width: 800px) {
        .hero { padding: 18px; }
        .hero-title { font-size: 1.55rem; }
        [data-testid="stMetric"] { padding: 12px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Research workspace · 2D segmentation</div>
      <div class="hero-title">U-Net Studio</div>
      <div class="hero-copy">
        Design a U-Net layer by layer, inspect how each change affects model complexity,
        compare ablations, and export a reproducible configuration for your research notes.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def layer_filter_editor(stage_label, default_filters, key_prefix):
    """Render a clear, compact editor for the convolutions in one stage."""
    current_count = min(max(len(default_filters), 1), 4)
    count = st.select_slider(
        f"Convolution layers in {stage_label.lower()}",
        options=[1, 2, 3, 4],
        value=current_count,
        key=f"{key_prefix}_count",
        help="Choose how many convolution layers this stage contains.",
    )
    filters = []
    cols = st.columns(count)
    for idx in range(count):
        default_value = (
            default_filters[idx]
            if idx < len(default_filters)
            else default_filters[-1]
        )
        with cols[idx]:
            filters.append(
                int(
                    st.number_input(
                        f"Conv {idx + 1} · output filters",
                        min_value=1,
                        max_value=2048,
                        value=int(default_value),
                        step=8,
                        key=f"{key_prefix}_filters_{idx + 1}",
                        help="Number of feature maps produced by this convolution.",
                    )
                )
            )
    st.caption(
        "Output filters become the next layer's input channels. "
        "Changes update the estimates automatically."
    )
    return tuple(filters)


with st.sidebar:
    st.markdown("## Model setup")
    st.caption("Start with the image and network settings. Use the workspace tabs for detailed editing.")

    with st.expander("1 · Input and output", expanded=True):
        size_col1, size_col2 = st.columns(2)
        with size_col1:
            height = st.selectbox("Image height", [64, 128, 256, 512], index=2)
        with size_col2:
            width = st.selectbox("Image width", [64, 128, 256, 512], index=2)
        in_ch = st.number_input(
            "Input channels", min_value=1, max_value=16, value=3,
            help="For example, RGB images use 3 channels and grayscale images use 1.",
        )
        classes = st.number_input(
            "Output classes", min_value=1, max_value=100, value=2,
            help="Number of output logits per pixel, including background if applicable.",
        )

    with st.expander("2 · U-Net structure", expanded=True):
        levels = st.slider(
            "Downsampling levels", min_value=1, max_value=5, value=4,
            help="Each level halves the spatial dimensions through max pooling.",
        )
        base = st.selectbox(
            "Initial filter count", [8, 16, 32, 64], index=3,
            help="Used to create the default filter plan; individual layers can be edited.",
        )
        kernel = st.selectbox("Convolution kernel size", [1, 3, 5], index=1)
        upsampling = st.radio(
            "Decoder upsampling",
            ["transpose", "bilinear"],
            format_func=lambda value: "Transposed convolution" if value == "transpose" else "Bilinear (no learned weights)",
            help="Transposed convolution learns upsampling weights; bilinear interpolation does not.",
        )
        skips = st.toggle("Use skip connections", value=True)
        batch_norm = st.toggle("Use batch normalization", value=False)
        bias = st.toggle("Use convolution biases", value=True)

    with st.expander("3 · Resource assumptions", expanded=False):
        precision = st.selectbox("Storage precision", ["FP32", "FP16", "INT8"])
        batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=1)
        st.caption(
            "Precision affects the storage estimates only; this tool does not quantize or run a model."
        )

encoder_filters = []
bottleneck_filters = ()
decoder_filters = []

bytes_per_value = {"FP32": 4, "FP16": 2, "INT8": 1}[precision]
# Use the current UI values to build the conventional default plan, then let users override each stage.
st.markdown('<div class="step-label">STEP 1 OF 3</div>', unsafe_allow_html=True)
st.markdown("## Design your layers")
st.markdown(
    '<div class="section-note">Choose a stage below. For each stage, set how many convolutions it contains and the output filters for each convolution. You can leave the default two-convolution design unchanged.</div>',
    unsafe_allow_html=True,
)

encoder_tab, bottleneck_tab, decoder_tab = st.tabs(
    ["① Encoder · downsampling", "② Bottleneck · deepest features", "③ Decoder · upsampling"]
)
with encoder_tab:
    st.markdown("### Encoder stages")
    st.caption("The encoder extracts features while spatial resolution decreases.")
    for level in range(levels):
        defaults = (base * (2 ** level),) * 2
        with st.container(border=True):
            st.markdown(f"**Stage {level + 1}**")
            st.caption(
                f"Default feature maps: {defaults[0]} · Resolution after pooling: "
                f"{height // (2 ** (level + 1))} × {width // (2 ** (level + 1))}"
            )
            encoder_filters.append(
                layer_filter_editor(
                    f"Encoder stage {level + 1}",
                    defaults,
                    f"enc_{level + 1}",
                )
            )

with bottleneck_tab:
    st.markdown("### Bottleneck")
    st.caption("The deepest representation connects the encoder and decoder.")
    with st.container(border=True):
        bottleneck_filters = layer_filter_editor(
            "Bottleneck",
            (base * (2 ** levels),) * 2,
            "bottleneck",
        )

with decoder_tab:
    st.markdown("### Decoder stages")
    st.caption("The decoder restores spatial resolution and combines encoder features through skip connections.")
    for level in reversed(range(levels)):
        defaults = (base * (2 ** level),) * 2
        with st.container(border=True):
            st.markdown(f"**Stage {level + 1}**")
            st.caption(f"Target resolution: {height // (2 ** level)} × {width // (2 ** level)}")
            decoder_filters.append(
                layer_filter_editor(
                    f"Decoder stage {level + 1}",
                    defaults,
                    f"dec_{level + 1}",
                )
            )

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
st.markdown('<div class="step-label">STEP 2 OF 3</div>', unsafe_allow_html=True)
st.markdown("## Complexity at a glance")
st.caption("These estimates refresh when you change any setting or layer above.")

metric_cols = st.columns(4)
metric_cols[0].metric(
    "Trainable parameters",
    f"{summary['parameters']:,}",
    help="Number of trainable weights and enabled biases, including batch-normalization parameters.",
)
metric_cols[1].metric(
    "Compute · MACs",
    f"{summary['MACs'] / 1e9:.3f} G",
    help="Estimated multiply–accumulate operations for convolution layers.",
)
metric_cols[2].metric(
    "Compute · FLOPs",
    f"{summary['FLOPs_2x_MACs'] / 1e9:.3f} G",
    help="Reported using the convention 1 MAC = 2 FLOPs.",
)
metric_cols[3].metric(
    "Weight storage",
    f"{summary['weight_memory_MiB']:.2f} MiB",
    help="Raw parameter storage at the selected precision; not total GPU memory.",
)
metric_cols2 = st.columns(2)
metric_cols2[0].metric(
    "Activation footprint",
    f"{summary['activation_tensor_footprint_MiB']:.2f} MiB",
    help="Sum of output tensor sizes plus input; this is not peak GPU memory.",
)
metric_cols2[1].metric("Total layer rows", f"{summary['layer_count']}")
st.caption(
    "Important: these are analytical estimates, not measured latency, peak GPU memory, "
    "or evidence of segmentation accuracy."
)

st.markdown('<div class="step-label">STEP 3 OF 3</div>', unsafe_allow_html=True)
st.markdown("## Inspect, compare, and export")
analysis_tab, compare_tab, export_tab = st.tabs(
    ["Layer analysis", "Ablation comparison", "Export and definitions"]
)

with analysis_tab:
    st.markdown("### Layer-by-layer breakdown")
    st.caption("Follow the model from input through the encoder, bottleneck, decoder, and output projection.")
    df = pd.DataFrame(result["layers"])
    display_df = df[
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
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download layer analysis · CSV",
        df.to_csv(index=False).encode("utf-8"),
        "unet_layer_analysis.csv",
        "text/csv",
        type="primary",
    )

    st.markdown("### Architecture map")
    dot = (
        'digraph UNet { rankdir=LR; graph [bgcolor="transparent", nodesep="0.35", ranksep="0.55"]; '
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, margin="0.16,0.12", fillcolor="#EAF3FC", color="#78A6D0", fontcolor="#17324D"]; '
        'edge [color="#6687A5", arrowsize=0.7, fontname="Arial", fontsize=9]; '
        f'input [label="INPUT\\n{height} × {width} × {in_ch}", fillcolor="#DCECFB"]; '
        f'enc [label="ENCODER\\n{levels} stages"]; '
        f'bn [label="BOTTLENECK\\n{len(bottleneck_filters)} conv layers", fillcolor="#DCECFB"]; '
        f'dec [label="DECODER\\n{levels} stages"]; '
        f'out [label="OUTPUT\\n{classes} classes", fillcolor="#DCECFB"]; '
        'input -> enc -> bn -> dec -> out; '
        + (
            'enc -> dec [style=dashed, label="skip connections", constraint=false]; '
            if skips
            else ""
        )
        + "}"
    )
    st.graphviz_chart(dot, use_container_width=True)

with compare_tab:
    st.markdown("### What changes when you remove or replace a component?")
    st.caption("Ablations compare complexity only. They do not indicate whether segmentation accuracy improves or declines.")
    comp = pd.DataFrame(compare_analyses(config))
    baseline_params = int(comp.loc[comp["variant"] == "Baseline", "parameters"].iloc[0])
    comp["parameter_change_%"] = comp["parameters"].apply(
        lambda value: round((value - baseline_params) / baseline_params * 100, 2)
        if baseline_params
        else 0
    )
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.markdown("#### Parameter count by variant")
    st.bar_chart(comp.set_index("variant")[["parameters"]], horizontal=True)

with export_tab:
    payload = {
        "model": "Configurable 2D U-Net-style analytical estimate",
        "config": config.to_dict(),
        "summary": summary,
        "layers": result["layers"],
    }
    st.markdown("### Save this configuration")
    st.caption("Use JSON to preserve the exact settings and analytical results for later reference.")
    st.download_button(
        "Download complete analysis · JSON",
        json.dumps(payload, indent=2),
        "unet_analysis.json",
        "application/json",
        type="primary",
    )
    with st.expander("Metric definitions and limitations", expanded=True):
        st.markdown(
            """
            - **Parameters:** convolution/transposed-convolution weights and enabled biases, plus trainable batch-normalization scale and offset.
            - **MACs:** estimated for convolutions. Pooling, normalization arithmetic, activations, concatenation, and bilinear interpolation are excluded.
            - **FLOPs:** reported as **2 × MACs**.
            - **Weight storage:** parameter count multiplied by bytes per parameter.
            - **Activation footprint:** sum of listed output tensors plus input. This is **not peak GPU memory**.
            - **Research note:** runtime, actual GPU memory, and segmentation quality require a model implementation and empirical evaluation.
            """
        )
    with st.expander("Current architecture configuration"):
        st.json(config.to_dict())

st.divider()
st.markdown(
    '<div class="small-muted">U-Net Studio · Same-padding 2D U-Net-style complexity estimator · Not an exact reproduction of the original valid-padding U-Net.</div>',
    unsafe_allow_html=True,
)

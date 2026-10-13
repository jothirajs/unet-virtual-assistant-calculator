import json
import pandas as pd
import streamlit as st
from unet_tool import UNetConfig, analyze_unet, compare_analyses

st.set_page_config(page_title="U-Net Virtual Assistant Calculator", page_icon="🧠", layout="wide")
st.title("U-Net Virtual Assistant Calculator")
st.caption("Interactive analytical calculator for 2D U-Net architecture design, teaching, and research.")
st.info("These are analytical estimates, not measured runtime benchmarks.")

with st.sidebar:
    st.header("Architecture configuration")
    height = st.selectbox("Input height", [64, 128, 256, 512], index=2)
    width = st.selectbox("Input width", [64, 128, 256, 512], index=2)
    in_ch = st.number_input("Input channels", 1, 16, 3)
    classes = st.number_input("Output classes", 1, 100, 2)
    levels = st.slider("Downsampling levels", 1, 5, 4)
    base = st.selectbox("Initial filters", [8, 16, 32, 64], index=3)
    kernel = st.selectbox("Convolution kernel", [1, 3, 5], index=1)
    upsampling = st.selectbox("Upsampling", ["transpose", "bilinear"])
    skips = st.checkbox("Skip connections", True)
    batch_norm = st.checkbox("Batch normalization", False)
    bias = st.checkbox("Convolution biases", True)
    precision = st.selectbox("Precision", ["FP32", "FP16", "INT8"])
    batch_size = st.number_input("Batch size", 1, 64, 1)

bytes_per_value = {"FP32": 4, "FP16": 2, "INT8": 1}[precision]
config = UNetConfig(input_height=int(height), input_width=int(width), input_channels=int(in_ch),
                    output_classes=int(classes), levels=int(levels), base_filters=int(base),
                    conv_kernel=int(kernel), upsampling=upsampling, skip_connections=skips,
                    batch_norm=batch_norm, use_bias=bias, precision_bytes=bytes_per_value,
                    batch_size=int(batch_size))
try:
    result = analyze_unet(config)
except ValueError as err:
    st.error(str(err))
    st.stop()

s = result["summary"]
cols = st.columns(4)
cols[0].metric("Trainable parameters", f"{s['parameters']:,}")
cols[1].metric("MACs", f"{s['MACs']/1e9:.3f} G")
cols[2].metric("FLOPs (2 × MACs)", f"{s['FLOPs_2x_MACs']/1e9:.3f} G")
cols[3].metric("Weight storage", f"{s['weight_memory_MiB']:.2f} MiB")
cols2 = st.columns(2)
cols2[0].metric("Activation tensor footprint", f"{s['activation_tensor_footprint_MiB']:.2f} MiB")
cols2[1].metric("Layer rows", str(s["layer_count"]))
st.caption("Activation footprint is the sum of layer-output tensor sizes plus input, not peak GPU memory.")

tab1, tab2, tab3 = st.tabs(["Layer-by-layer analysis", "Ablation comparison", "Export"])
with tab1:
    df = pd.DataFrame(result["layers"])
    st.dataframe(df[["layer", "type", "output_shape", "parameters", "MACs", "notes"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Download layer analysis (CSV)", df.to_csv(index=False).encode(),
                       "unet_layer_analysis.csv", "text/csv")
with tab2:
    comp = pd.DataFrame(compare_analyses(config))
    baseline = int(comp.loc[comp["variant"] == "Baseline", "parameters"].iloc[0])
    comp["parameter_change_%"] = comp["parameters"].apply(
        lambda x: round((x - baseline) / baseline * 100, 2) if baseline else 0)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.bar_chart(comp.set_index("variant")[["parameters"]])
    st.caption("Complexity ablations do not establish segmentation accuracy or clinical utility.")
with tab3:
    payload = {"model": "2D U-Net analytical estimate", "config": config.to_dict(),
               "summary": s, "layers": result["layers"]}
    st.download_button("Download full analysis (JSON)", json.dumps(payload, indent=2),
                       "unet_analysis.json", "application/json")
    st.json({"configuration": config.to_dict(), "metric_definitions": {
        "MACs": s["MAC_definition"], "FLOPs": "Reported as 2 × MACs.",
        "Weight storage": "Parameter count × bytes per parameter.",
        "Activation footprint": s["activation_footprint_definition"]}})

st.divider()
st.subheader("Architecture flow")
dot = (
    'digraph UNet { rankdir=LR; graph [bgcolor="transparent"]; '
    'node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#4C78A8"]; '
    f'input [label="Input\\n{height}×{width}×{in_ch}"]; '
    f'enc [label="Encoder\\n{levels} levels"]; '
    f'bn [label="Bottleneck\\n{base * (2 ** levels)} filters"]; '
    f'dec [label="Decoder\\n{upsampling}"]; '
    f'out [label="Output logits\\n{classes} classes"]; '
    'input -> enc -> bn -> dec -> out; '
    + ('enc -> dec [style=dashed, label="skip connections", constraint=false]; ' if skips else '')
    + '}'
)
st.graphviz_chart(dot)
st.caption("Same-padding 2D U-Net-style model; not an exact reproduction of the original valid-padding U-Net.")

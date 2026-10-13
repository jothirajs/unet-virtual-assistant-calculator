import unittest
from dataclasses import replace

from unet_tool import UNetConfig, analyze_unet, apply_ablation, compare_analyses


class TestUNetEngine(unittest.TestCase):
    def test_baseline_counts_are_positive(self):
        summary = analyze_unet(UNetConfig())["summary"]
        self.assertGreater(summary["parameters"], 0)
        self.assertGreater(summary["MACs"], 0)
        self.assertEqual(summary["FLOPs_2x_MACs"], 2 * summary["MACs"])

    def test_weight_memory_matches_parameter_count(self):
        summary = analyze_unet(UNetConfig(precision_bytes=4))["summary"]
        self.assertAlmostEqual(
            summary["weight_memory_MiB"],
            summary["parameters"] * 4 / (1024 ** 2),
            places=8,
        )

    def test_invalid_spatial_size_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_unet(UNetConfig(input_height=250))

    def test_disabling_bias_reduces_parameters(self):
        config = UNetConfig()
        baseline = analyze_unet(config)["summary"]["parameters"]
        no_bias = analyze_unet(
            apply_ablation(config, "Without convolution biases")
        )["summary"]["parameters"]
        self.assertLess(no_bias, baseline)

    def test_batch_norm_adds_trainable_parameters(self):
        config = UNetConfig()
        baseline = analyze_unet(config)["summary"]["parameters"]
        with_bn = analyze_unet(replace(config, batch_norm=True))["summary"]["parameters"]
        self.assertGreater(with_bn, baseline)

    def test_skip_connections_can_be_selected_per_level(self):
        config = UNetConfig(skip_connection_levels=(2,))
        layers = analyze_unet(config)["layers"]
        skip_layers = [row["layer"] for row in layers if row["type"] == "Concatenate"]
        self.assertEqual(skip_layers, ["Decoder 2 SkipConcat"])
        self.assertEqual(config.to_dict()["skip_connection_levels"], [2])

    def test_ablation_table_contains_five_variants(self):
        self.assertEqual(len(compare_analyses(UNetConfig())), 5)

    def test_custom_layer_filters_and_counts_change_complexity(self):
        baseline = analyze_unet(UNetConfig())["summary"]
        custom = UNetConfig(
            encoder_filters=((16,), (32, 32, 32), (64,), (128, 128)),
            bottleneck_filters=(256, 128, 256),
            decoder_filters=((64,), (32, 32), (16, 16, 16), (8,)),
        )
        result = analyze_unet(custom)
        self.assertNotEqual(result["summary"]["parameters"], baseline["parameters"])
        self.assertEqual(result["config"]["encoder_filters"][1], [32, 32, 32])
        self.assertTrue(any(row["layer"] == "Encoder 2 Conv 3" for row in result["layers"]))
        self.assertTrue(any(row["layer"] == "Bottleneck Conv 3" for row in result["layers"]))

    def test_empty_custom_stage_is_rejected(self):
        config = UNetConfig(encoder_filters=((), (32, 32), (64, 64), (128, 128)))
        with self.assertRaises(ValueError):
            analyze_unet(config)


if __name__ == "__main__":
    unittest.main()

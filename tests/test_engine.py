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

    def test_ablation_table_contains_five_variants(self):
        self.assertEqual(len(compare_analyses(UNetConfig())), 5)


if __name__ == "__main__":
    unittest.main()

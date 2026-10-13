import unittest
from dataclasses import replace
from unet_tool import UNetConfig, analyze_unet, apply_ablation, compare_analyses

class TestUNetEngine(unittest.TestCase):
    def test_baseline_counts_positive(self):
        s = analyze_unet(UNetConfig())["summary"]
        self.assertGreater(s["parameters"], 0)
        self.assertGreater(s["MACs"], 0)
        self.assertEqual(s["FLOPs_2x_MACs"], 2 * s["MACs"])

    def test_weight_memory(self):
        s = analyze_unet(UNetConfig())["summary"]
        self.assertAlmostEqual(s["weight_memory_MiB"], s["parameters"] * 4 / (1024**2), places=8)

    def test_invalid_spatial_size(self):
        with self.assertRaises(ValueError):
            analyze_unet(UNetConfig(input_height=250))

    def test_bias_ablation_reduces_parameters(self):
        cfg = UNetConfig()
        base = analyze_unet(cfg)["summary"]["parameters"]
        no_bias = analyze_unet(apply_ablation(cfg, "Without convolution biases"))["summary"]["parameters"]
        self.assertLess(no_bias, base)

    def test_batch_norm_adds_parameters(self):
        cfg = UNetConfig()
        base = analyze_unet(cfg)["summary"]["parameters"]
        with_bn = analyze_unet(replace(cfg, batch_norm=True))["summary"]["parameters"]
        self.assertGreater(with_bn, base)

    def test_ablation_count(self):
        self.assertEqual(len(compare_analyses(UNetConfig())), 5)

if __name__ == "__main__":
    unittest.main()

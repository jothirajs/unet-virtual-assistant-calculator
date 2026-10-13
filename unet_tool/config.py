from dataclasses import dataclass, asdict
from typing import Literal


@dataclass
class UNetConfig:
    """Configuration for a configurable same-padding 2D U-Net-style model."""

    input_height: int = 256
    input_width: int = 256
    input_channels: int = 3
    output_classes: int = 2
    levels: int = 4
    base_filters: int = 64
    conv_kernel: int = 3
    pool_kernel: int = 2
    upsample_kernel: int = 2
    padding: Literal["same"] = "same"
    upsampling: Literal["transpose", "bilinear"] = "transpose"
    skip_connections: bool = True
    batch_norm: bool = False
    use_bias: bool = True
    precision_bytes: int = 4
    batch_size: int = 1

    def validate(self) -> None:
        values = {
            "input_height": self.input_height,
            "input_width": self.input_width,
            "input_channels": self.input_channels,
            "output_classes": self.output_classes,
            "levels": self.levels,
            "base_filters": self.base_filters,
            "conv_kernel": self.conv_kernel,
            "pool_kernel": self.pool_kernel,
            "upsample_kernel": self.upsample_kernel,
            "precision_bytes": self.precision_bytes,
            "batch_size": self.batch_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.levels > 8:
            raise ValueError("levels must be 8 or fewer.")
        divisor = self.pool_kernel ** self.levels
        if self.input_height % divisor or self.input_width % divisor:
            raise ValueError(
                f"Input height and width must be divisible by pool_kernel**levels ({divisor})."
            )
        if self.padding != "same":
            raise ValueError("Only same padding is currently supported.")
        if self.upsampling not in ("transpose", "bilinear"):
            raise ValueError("upsampling must be 'transpose' or 'bilinear'.")

    def to_dict(self) -> dict:
        return asdict(self)

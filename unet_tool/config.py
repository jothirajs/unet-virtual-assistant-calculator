from dataclasses import dataclass, asdict
from typing import Literal


@dataclass
class UNetConfig:
    """Configuration for a configurable same-padding 2D U-Net-style model.

    Each nested filter list describes the output-channel count of each convolution
    in that stage. Empty/None specifications fall back to the conventional
    two-convolution geometric U-Net layout.
    """

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
    # Optional 1-based U-Net levels with active encoder-to-decoder skip connections.
    # None preserves the legacy global skip_connections setting.
    skip_connection_levels: tuple[int, ...] | None = None
    batch_norm: bool = False
    use_bias: bool = True
    precision_bytes: int = 4
    batch_size: int = 1
    encoder_filters: tuple[tuple[int, ...], ...] | None = None
    bottleneck_filters: tuple[int, ...] | None = None
    decoder_filters: tuple[tuple[int, ...], ...] | None = None

    def effective_encoder_filters(self) -> tuple[tuple[int, ...], ...]:
        if self.encoder_filters is not None:
            return tuple(tuple(stage) for stage in self.encoder_filters)
        return tuple(
            (self.base_filters * (2 ** level),) * 2
            for level in range(self.levels)
        )

    def effective_bottleneck_filters(self) -> tuple[int, ...]:
        if self.bottleneck_filters is not None:
            return tuple(self.bottleneck_filters)
        return (self.base_filters * (2 ** self.levels),) * 2

    def effective_decoder_filters(self) -> tuple[tuple[int, ...], ...]:
        if self.decoder_filters is not None:
            return tuple(tuple(stage) for stage in self.decoder_filters)
        return tuple(
            (self.base_filters * (2 ** level),) * 2
            for level in reversed(range(self.levels))
        )

    def effective_skip_connection_levels(self) -> tuple[int, ...]:
        if self.skip_connection_levels is not None:
            return tuple(sorted(set(self.skip_connection_levels)))
        return tuple(range(1, self.levels + 1)) if self.skip_connections else ()

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
        skip_levels = self.effective_skip_connection_levels()
        if any(not isinstance(level, int) or level < 1 or level > self.levels for level in skip_levels):
            raise ValueError("skip_connection_levels must contain valid 1-based levels.")
        if len(skip_levels) != len(set(skip_levels)):
            raise ValueError("skip_connection_levels must not contain duplicates.")

        encoder = self.effective_encoder_filters()
        bottleneck = self.effective_bottleneck_filters()
        decoder = self.effective_decoder_filters()
        if len(encoder) != self.levels or len(decoder) != self.levels:
            raise ValueError("Encoder and decoder specifications must match the number of levels.")
        for stage_name, stages in (("encoder_filters", encoder), ("decoder_filters", decoder)):
            for index, stage in enumerate(stages, start=1):
                if not stage:
                    raise ValueError(f"{stage_name} stage {index} must contain at least one convolution.")
                if any(not isinstance(filters, int) or filters <= 0 for filters in stage):
                    raise ValueError(f"Every filter count in {stage_name} must be a positive integer.")
        if not bottleneck or any(not isinstance(filters, int) or filters <= 0 for filters in bottleneck):
            raise ValueError("bottleneck_filters must contain positive integer filter counts.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["encoder_filters"] = [list(stage) for stage in self.effective_encoder_filters()]
        data["bottleneck_filters"] = list(self.effective_bottleneck_filters())
        data["decoder_filters"] = [list(stage) for stage in self.effective_decoder_filters()]
        data["skip_connection_levels"] = list(self.effective_skip_connection_levels())
        return data

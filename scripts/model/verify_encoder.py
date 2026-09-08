import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow direct execution from repository root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.encoder import Encoder, EncoderStage


class DummyAttention(nn.Module):
    """
    Minimal attention block used only for validating the
    Encoder's attention-factory interface.

    This is NOT the Phase 10 CBAM implementation.
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        # Actual parameter so parameter registration and
        # gradient flow through the attention interface can
        # be verified.
        self.scale = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

    def forward(self, x):
        return x * self.scale


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"  PASS  {message}")


def test_encoder_structure(encoder):
    print("\n[1] Encoder architecture")

    stages = [
        encoder.stage1,
        encoder.stage2,
        encoder.stage3,
        encoder.stage4,
    ]

    expected_channels = [
        (12, 64),
        (64, 128),
        (128, 256),
        (256, 512),
    ]

    for index, (stage, (expected_in, expected_out)) in enumerate(
        zip(stages, expected_channels),
        start=1,
    ):
        check(
            isinstance(stage, EncoderStage),
            f"stage{index} is EncoderStage",
        )

        first_conv = stage.double_conv[0]

        check(
            first_conv.in_channels == expected_in,
            f"stage{index} input channels = {expected_in}",
        )

        check(
            first_conv.out_channels == expected_out,
            f"stage{index} output channels = {expected_out}",
        )

    check(
        encoder.stage1 is not encoder.stage2,
        "stage1 and stage2 are independent",
    )

    check(
        encoder.stage2 is not encoder.stage3,
        "stage2 and stage3 are independent",
    )

    check(
        encoder.stage3 is not encoder.stage4,
        "stage3 and stage4 are independent",
    )


def test_convolution_configuration(encoder):
    print("\n[2] Convolution configuration")

    stages = [
        encoder.stage1,
        encoder.stage2,
        encoder.stage3,
        encoder.stage4,
    ]

    for index, stage in enumerate(stages, start=1):
        conv_layers = [
            module
            for module in stage.double_conv
            if isinstance(module, nn.Conv2d)
        ]

        bn_layers = [
            module
            for module in stage.double_conv
            if isinstance(module, nn.BatchNorm2d)
        ]

        elu_layers = [
            module
            for module in stage.double_conv
            if isinstance(module, nn.ELU)
        ]

        check(
            len(conv_layers) == 2,
            f"stage{index} has exactly 2 Conv2d layers",
        )

        check(
            len(bn_layers) == 2,
            f"stage{index} has exactly 2 BatchNorm2d layers",
        )

        check(
            len(elu_layers) == 2,
            f"stage{index} has exactly 2 ELU layers",
        )

        for conv_index, conv in enumerate(
            conv_layers,
            start=1,
        ):
            check(
                conv.kernel_size == (3, 3),
                f"stage{index} Conv{conv_index} kernel = (3, 3)",
            )

            check(
                conv.padding == (1, 1),
                f"stage{index} Conv{conv_index} padding = (1, 1)",
            )

            check(
                conv.stride == (1, 1),
                f"stage{index} Conv{conv_index} stride = (1, 1)",
            )

        check(
            all(elu.inplace for elu in elu_layers),
            f"stage{index} all ELU activations use inplace=True",
        )


def test_pooling_configuration(encoder):
    print("\n[3] MaxPool configuration")

    stages = [
        encoder.stage1,
        encoder.stage2,
        encoder.stage3,
        encoder.stage4,
    ]

    for index, stage in enumerate(stages, start=1):
        pool = stage.pool

        check(
            pool.kernel_size == 2,
            f"stage{index} pool kernel_size = 2",
        )

        check(
            pool.stride == 2,
            f"stage{index} pool stride = 2",
        )

        check(
            pool.padding == 0,
            f"stage{index} pool padding = 0",
        )


def test_input_and_output_geometry(encoder):
    print("\n[4] Input and output geometry")

    batch_size = 2

    x = torch.randn(
        batch_size,
        12,
        101,
        241,
    )

    latent, skips = encoder(x)

    expected_input_shape = (
        batch_size,
        12,
        101,
        241,
    )

    expected_skip_shapes = [
        (batch_size, 64, 101, 241),
        (batch_size, 128, 50, 120),
        (batch_size, 256, 25, 60),
        (batch_size, 512, 12, 30),
    ]

    expected_latent_shape = (
        batch_size,
        512,
        6,
        15,
    )

    check(
        tuple(x.shape) == expected_input_shape,
        f"input shape = {tuple(x.shape)}",
    )

    check(
        len(skips) == 4,
        f"number of skip connections = {len(skips)}",
    )

    for index, (skip, expected) in enumerate(
        zip(skips, expected_skip_shapes),
        start=1,
    ):
        check(
            tuple(skip.shape) == expected,
            f"skip{index} shape = {tuple(skip.shape)}",
        )

    check(
        tuple(latent.shape) == expected_latent_shape,
        f"latent shape = {tuple(latent.shape)}",
    )


def test_odd_dimension_behavior(encoder):
    print("\n[5] Odd-dimension downsampling behavior")

    x = torch.randn(
        1,
        12,
        101,
        241,
    )

    expected_latitude_progression = [
        101,
        50,
        25,
        12,
        6,
    ]

    expected_longitude_progression = [
        241,
        120,
        60,
        30,
        15,
    ]

    stages = [
        encoder.stage1,
        encoder.stage2,
        encoder.stage3,
        encoder.stage4,
    ]

    current = x

    actual_latitude_progression = [
        current.shape[2]
    ]

    actual_longitude_progression = [
        current.shape[3]
    ]

    for stage in stages:
        _, current = stage(current)

        actual_latitude_progression.append(
            current.shape[2]
        )

        actual_longitude_progression.append(
            current.shape[3]
        )

    check(
        actual_latitude_progression
        == expected_latitude_progression,
        "latitude progression = 101 → 50 → 25 → 12 → 6",
    )

    check(
        actual_longitude_progression
        == expected_longitude_progression,
        "longitude progression = 241 → 120 → 60 → 30 → 15",
    )

    expected_pooled_shapes = [
        (50, 120),
        (25, 60),
        (12, 30),
        (6, 15),
    ]

    current = x

    for index, (stage, expected_spatial) in enumerate(
        zip(stages, expected_pooled_shapes),
        start=1,
    ):
        _, current = stage(current)

        check(
            tuple(current.shape[2:]) == expected_spatial,
            f"stage{index} pooled spatial shape = "
            f"{expected_spatial}",
        )

def test_skip_timing(encoder):
    print("\n[6] Skip connection timing")

    x = torch.randn(
        1,
        12,
        101,
        241,
    )

    skip1, down1 = encoder.stage1(x)

    with torch.no_grad():
        conv_output = encoder.stage1.double_conv(x)

        expected_skip = encoder.stage1.attention(
            conv_output
        )

        expected_downsampled = encoder.stage1.pool(
            expected_skip
        )

    check(
        torch.allclose(skip1, expected_skip),
        "skip is captured after attention",
    )

    check(
        torch.allclose(down1, expected_downsampled),
        "downsampled tensor is produced from attention output",
    )

    check(
        skip1.shape[2:] == (101, 241),
        "skip retains pre-pooling spatial dimensions",
    )

    check(
        down1.shape[2:] == (50, 120),
        "downsampled tensor has pooled spatial dimensions",
    )


def test_numerical_sanity(encoder):
    print("\n[7] Numerical sanity")

    x = torch.randn(
        2,
        12,
        101,
        241,
    )

    latent, skips = encoder(x)

    check(
        not torch.isnan(latent).any().item(),
        "latent contains no NaN",
    )

    check(
        not torch.isinf(latent).any().item(),
        "latent contains no Inf",
    )

    for index, skip in enumerate(skips, start=1):
        check(
            not torch.isnan(skip).any().item(),
            f"skip{index} contains no NaN",
        )

        check(
            not torch.isinf(skip).any().item(),
            f"skip{index} contains no Inf",
        )


def test_elu_configuration(encoder):
    print("\n[8] ELU activation configuration")

    elu_layers = [
        module
        for module in encoder.modules()
        if isinstance(module, nn.ELU)
    ]

    expected_count = 8

    check(
        len(elu_layers) == expected_count,
        f"total ELU layers = {len(elu_layers)}",
    )

    check(
        all(module.inplace for module in elu_layers),
        "all ELU activations use inplace=True",
    )


def test_parameter_sanity(encoder):
    print("\n[9] Parameter sanity")

    total_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )

    non_trainable_parameters = (
        total_parameters - trainable_parameters
    )

    print(f"  Total parameters       : {total_parameters:,}")
    print(f"  Trainable parameters   : {trainable_parameters:,}")
    print(f"  Non-trainable          : {non_trainable_parameters:,}")

    check(
        total_parameters > 0,
        "encoder has parameters",
    )

    check(
        trainable_parameters > 0,
        "encoder has trainable parameters",
    )

    check(
        trainable_parameters <= total_parameters,
        "trainable parameters do not exceed total parameters",
    )


def test_gradient_flow(encoder):
    print("\n[10] Gradient flow")

    encoder.zero_grad(set_to_none=True)

    x = torch.randn(
        2,
        12,
        101,
        241,
        requires_grad=True,
    )

    latent, skips = encoder(x)

    loss = latent.mean()

    loss.backward()

    check(
        x.grad is not None,
        "input receives gradients",
    )

    check(
        torch.isfinite(x.grad).all().item(),
        "input gradients are finite",
    )

    trainable_parameters = [
        parameter
        for parameter in encoder.parameters()
        if parameter.requires_grad
    ]

    check(
        len(trainable_parameters) > 0,
        "trainable parameters exist",
    )

    check(
        all(
            parameter.grad is not None
            for parameter in trainable_parameters
        ),
        "all trainable parameters receive gradients",
    )

    check(
        all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in trainable_parameters
        ),
        "all parameter gradients are finite",
    )


def test_attention_factory(encoder):
    print("\n[11] Attention factory integration")

    expected_channels = [
        64,
        128,
        256,
        512,
    ]

    stages = [
        encoder.stage1,
        encoder.stage2,
        encoder.stage3,
        encoder.stage4,
    ]

    for index, (stage, expected) in enumerate(
        zip(stages, expected_channels),
        start=1,
    ):
        check(
            isinstance(stage.attention, DummyAttention),
            f"stage{index} attention is DummyAttention",
        )

        check(
            stage.attention.channels == expected,
            f"stage{index} attention channels = {expected}",
        )

    attentions = [
        stage.attention
        for stage in stages
    ]

    for i in range(len(attentions)):
        for j in range(i + 1, len(attentions)):
            check(
                attentions[i] is not attentions[j],
                f"stage{i + 1} and stage{j + 1} "
                f"attention instances are independent",
            )


def main():
    print("=" * 90)
    print("OceanEmbed — Phase 9.5 FINAL Encoder Verification")
    print("=" * 90)

    torch.manual_seed(42)

    # Use the test attention factory rather than the real CBAM.
    # Real CBAM implementation belongs to Phase 10.
    encoder = Encoder(
        attention_block=DummyAttention,
    )

    encoder.eval()

    test_encoder_structure(encoder)
    test_convolution_configuration(encoder)
    test_pooling_configuration(encoder)
    test_input_and_output_geometry(encoder)
    test_odd_dimension_behavior(encoder)
    test_skip_timing(encoder)
    test_numerical_sanity(encoder)
    test_elu_configuration(encoder)
    test_parameter_sanity(encoder)
    test_gradient_flow(encoder)
    test_attention_factory(encoder)

    print("\n" + "=" * 90)
    print("PHASE 9.5 FINAL VERIFICATION: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()
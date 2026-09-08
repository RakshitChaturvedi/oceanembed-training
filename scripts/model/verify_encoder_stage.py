import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running directly from the repository root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.encoder import EncoderStage


class DummyAttention(nn.Module):
    """
    Minimal attention block used to verify the EncoderStage
    attention factory interface.

    It records the channel count and applies a deterministic
    transformation so we can verify that the skip connection
    is captured after attention.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x * 2.0


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"  PASS  {message}")


def test_basic_stage_geometry():
    print("\n[1] Basic EncoderStage geometry")

    stage = EncoderStage(
        in_channels=12,
        out_channels=64,
    )

    x = torch.randn(2, 12, 101, 241)

    skip, downsampled = stage(x)

    check(
        skip.shape == (2, 64, 101, 241),
        f"skip shape = {tuple(skip.shape)}",
    )

    check(
        downsampled.shape == (2, 64, 50, 120),
        f"downsampled shape = {tuple(downsampled.shape)}",
    )


def test_deeper_stage_geometry():
    print("\n[2] Deeper EncoderStage geometry")

    stage = EncoderStage(
        in_channels=128,
        out_channels=256,
    )

    x = torch.randn(2, 128, 25, 60)

    skip, downsampled = stage(x)

    check(
        skip.shape == (2, 256, 25, 60),
        f"skip shape = {tuple(skip.shape)}",
    )

    check(
        downsampled.shape == (2, 256, 12, 30),
        f"downsampled shape = {tuple(downsampled.shape)}",
    )


def test_convolution_block():
    print("\n[3] Double convolution block")

    stage = EncoderStage(
        in_channels=12,
        out_channels=64,
    )

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
        f"Conv2d layers = {len(conv_layers)}",
    )

    check(
        len(bn_layers) == 2,
        f"BatchNorm2d layers = {len(bn_layers)}",
    )

    check(
        len(elu_layers) == 2,
        f"ELU layers = {len(elu_layers)}",
    )

    for i, conv in enumerate(conv_layers, start=1):
        check(
            conv.kernel_size == (3, 3),
            f"Conv{i} kernel = {conv.kernel_size}",
        )

        check(
            conv.padding == (1, 1),
            f"Conv{i} padding = {conv.padding}",
        )

        check(
            conv.stride == (1, 1),
            f"Conv{i} stride = {conv.stride}",
        )

    check(
        all(layer.inplace for layer in elu_layers),
        "all ELU activations use inplace=True",
    )


def test_pooling():
    print("\n[4] Pooling configuration")

    stage = EncoderStage(
        in_channels=12,
        out_channels=64,
    )

    pool = stage.pool

    check(
        pool.kernel_size == 2,
        f"pool kernel_size = {pool.kernel_size}",
    )

    check(
        pool.stride == 2,
        f"pool stride = {pool.stride}",
    )

    check(
        pool.padding == 0,
        f"pool padding = {pool.padding}",
    )


def test_attention_factory():
    print("\n[5] Attention factory injection")

    stage = EncoderStage(
        in_channels=12,
        out_channels=64,
        attention_block=DummyAttention,
    )

    check(
        isinstance(stage.attention, DummyAttention),
        "attention block instantiated from factory",
    )

    check(
        stage.attention.channels == 64,
        f"attention channels = {stage.attention.channels}",
    )


def test_skip_is_after_attention():
    print("\n[6] Skip connection is captured after attention")

    stage = EncoderStage(
        in_channels=12,
        out_channels=64,
        attention_block=DummyAttention,
    )

    x = torch.randn(1, 12, 101, 241)

    # Get the double-convolution output independently.
    with torch.no_grad():
        conv_output = stage.double_conv(x)

    skip, downsampled = stage(x)

    expected_skip = conv_output * 2.0

    check(
        torch.allclose(skip, expected_skip),
        "skip contains attention-transformed features",
    )

    expected_downsampled = stage.pool(expected_skip)

    check(
        torch.allclose(downsampled, expected_downsampled),
        "downsampled tensor is produced from attention output",
    )


def test_default_attention():
    print("\n[7] Default attention behavior")

    stage = EncoderStage(
        in_channels=12,
        out_channels=64,
    )

    check(
        isinstance(stage.attention, nn.Identity),
        "attention defaults to nn.Identity",
    )


def test_gradient_flow():
    print("\n[8] Gradient flow")

    stage = EncoderStage(
        in_channels=12,
        out_channels=64,
    )

    x = torch.randn(
        2,
        12,
        101,
        241,
        requires_grad=True,
    )

    skip, downsampled = stage(x)

    loss = skip.mean() + downsampled.mean()
    loss.backward()

    check(
        x.grad is not None,
        "input receives gradients",
    )

    check(
        torch.isfinite(x.grad).all().item(),
        "input gradients are finite",
    )

    parameter_gradients = [
        parameter.grad
        for parameter in stage.parameters()
        if parameter.requires_grad
    ]

    check(
        all(gradient is not None for gradient in parameter_gradients),
        "all trainable parameters receive gradients",
    )

    check(
        all(
            torch.isfinite(gradient).all().item()
            for gradient in parameter_gradients
        ),
        "all parameter gradients are finite",
    )


def main():
    print("=" * 90)
    print("OceanEmbed — Phase 9.2 EncoderStage Verification")
    print("=" * 90)

    torch.manual_seed(42)

    test_basic_stage_geometry()
    test_deeper_stage_geometry()
    test_convolution_block()
    test_pooling()
    test_attention_factory()
    test_skip_is_after_attention()
    test_default_attention()
    test_gradient_flow()

    print("\n" + "=" * 90)
    print("PHASE 9.2 VERIFICATION: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()
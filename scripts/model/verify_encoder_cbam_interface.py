import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.encoder import Encoder


class DummyCBAM(nn.Module):
    """
    Stand-in for the real CBAM implementation.

    Used only to verify the Phase 9.4 factory interface.
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        # Give every instance an actual parameter so that
        # parameter registration can also be inspected.
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"  PASS  {message}")


def test_cbam_factory_instantiation():
    print("\n[1] CBAM factory instantiation")

    encoder = Encoder(
        attention_block=DummyCBAM,
    )

    expected_channels = [64, 128, 256, 512]

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
            isinstance(stage.attention, DummyCBAM),
            f"stage{index} contains DummyCBAM",
        )

        check(
            stage.attention.channels == expected,
            f"stage{index} CBAM channels = {expected}",
        )


def test_cbam_instances_are_independent():
    print("\n[2] CBAM instance independence")

    encoder = Encoder(
        attention_block=DummyCBAM,
    )

    attentions = [
        encoder.stage1.attention,
        encoder.stage2.attention,
        encoder.stage3.attention,
        encoder.stage4.attention,
    ]

    for i in range(len(attentions)):
        for j in range(i + 1, len(attentions)):
            check(
                attentions[i] is not attentions[j],
                f"stage{i + 1} and stage{j + 1} CBAM instances "
                f"are independent",
            )


def test_cbam_parameters_are_registered():
    print("\n[3] CBAM parameter registration")

    encoder = Encoder(
        attention_block=DummyCBAM,
    )

    stages = [
        encoder.stage1,
        encoder.stage2,
        encoder.stage3,
        encoder.stage4,
    ]

    expected_channels = [64, 128, 256, 512]

    for index, (stage, expected) in enumerate(
        zip(stages, expected_channels),
        start=1,
    ):
        parameters = list(stage.attention.parameters())

        check(
            len(parameters) > 0,
            f"stage{index} CBAM has registered parameters",
        )

        check(
            parameters[0].shape == (1, expected, 1, 1),
            f"stage{index} CBAM parameter shape = "
            f"{tuple(parameters[0].shape)}",
        )


def test_cbam_parameters_are_distinct():
    print("\n[4] CBAM parameters are distinct")

    encoder = Encoder(
        attention_block=DummyCBAM,
    )

    attentions = [
        encoder.stage1.attention,
        encoder.stage2.attention,
        encoder.stage3.attention,
        encoder.stage4.attention,
    ]

    parameters = [
        list(attention.parameters())[0]
        for attention in attentions
    ]

    for i in range(len(parameters)):
        for j in range(i + 1, len(parameters)):
            check(
                parameters[i] is not parameters[j],
                f"stage{i + 1} and stage{j + 1} parameters "
                f"are independent",
            )


def test_forward_with_cbam_interface():
    print("\n[5] Forward pass with CBAM interface")

    encoder = Encoder(
        attention_block=DummyCBAM,
    )

    x = torch.randn(
        2,
        12,
        101,
        241,
    )

    latent, skips = encoder(x)

    check(
        latent.shape == (2, 512, 6, 15),
        f"latent shape = {tuple(latent.shape)}",
    )

    expected_skip_shapes = [
        (2, 64, 101, 241),
        (2, 128, 50, 120),
        (2, 256, 25, 60),
        (2, 512, 12, 30),
    ]

    for index, (skip, expected) in enumerate(
        zip(skips, expected_skip_shapes),
        start=1,
    ):
        check(
            skip.shape == expected,
            f"skip{index} shape = {tuple(skip.shape)}",
        )


def test_backward_with_cbam_interface():
    print("\n[6] Backward pass with CBAM interface")

    encoder = Encoder(
        attention_block=DummyCBAM,
    )

    x = torch.randn(
        2,
        12,
        101,
        241,
        requires_grad=True,
    )

    latent, _ = encoder(x)

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

    cbam_parameters = []

    for stage in [
        encoder.stage1,
        encoder.stage2,
        encoder.stage3,
        encoder.stage4,
    ]:
        cbam_parameters.extend(stage.attention.parameters())

    check(
        all(parameter.grad is not None for parameter in cbam_parameters),
        "all CBAM parameters receive gradients",
    )

    check(
        all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in cbam_parameters
        ),
        "all CBAM parameter gradients are finite",
    )


def main():
    print("=" * 90)
    print("OceanEmbed — Phase 9.4 CBAM Interface Verification")
    print("=" * 90)

    torch.manual_seed(42)

    test_cbam_factory_instantiation()
    test_cbam_instances_are_independent()
    test_cbam_parameters_are_registered()
    test_cbam_parameters_are_distinct()
    test_forward_with_cbam_interface()
    test_backward_with_cbam_interface()

    print("\n" + "=" * 90)
    print("PHASE 9.4 VERIFICATION: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()
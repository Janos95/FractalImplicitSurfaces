import argparse
from pathlib import Path
import math
import time

import numpy as np
import torch
from PIL import Image

EMBEDDING_DIM = 512
ATTENTION_DIM = 256
AFFINE_HIDDEN_DIM = 128
MAX_CONTRAST = 1.8
SOFTMAX_TEMPERATURE = 0.25
RANGE_SIZE = 4
DOMAIN_SIZE = 8
IMAGE_SIZE = 512
NUM_RANGE_BLOCKS = (IMAGE_SIZE // RANGE_SIZE) ** 2
NUM_DOMAIN_BLOCKS = (IMAGE_SIZE // DOMAIN_SIZE) ** 2

DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 1e-3
SEED = 1337
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR / "lenna.png"
TARGET_RANGE_COORD = (64, 64)
DEFAULT_CHECKPOINT_PATH = SCRIPT_DIR / "fractal_attention_weights.pt"

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
    device = torch.device("mps")
else:
    raise RuntimeError("No CUDA or MPS device available.")


def fmt(value: float) -> str:
    return f"{value:.4e}"


def get_blocks(image: torch.Tensor, block_size: int) -> torch.Tensor:
    height, width = image.shape
    assert height == width, "image must be square"
    blocks_per_side = height // block_size
    image = image.contiguous()
    return (
        image.view(blocks_per_side, block_size, blocks_per_side, block_size)
        .permute(0, 2, 1, 3)
        .reshape(blocks_per_side * blocks_per_side, block_size, block_size)
    )


def get_pooled_blocks(image: torch.Tensor, block_size: int) -> torch.Tensor:
    pooled = (
        image[..., ::2, ::2]
        + image[..., ::2, 1::2]
        + image[..., 1::2, ::2]
        + image[..., 1::2, 1::2]
    ) * 0.25
    return get_blocks(pooled, block_size)


def reassemble_blocks(blocks: torch.Tensor, image_size: int) -> torch.Tensor:
    num_blocks, block_size, _ = blocks.shape
    blocks_per_side = image_size // block_size
    return (
        blocks.view(blocks_per_side, blocks_per_side, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(image_size, image_size)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural fractal compression.")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Optimizer learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Where to save the trained model weights (default: %(default)s).",
    )
    return parser.parse_args()


class FractalAttention(torch.nn.Module):
    def __init__(self, range_positional: torch.Tensor, domain_positional: torch.Tensor) -> None:
        super().__init__()
        self.register_parameter(
            "range_latents",
            torch.nn.Parameter(torch.randn(NUM_RANGE_BLOCKS, EMBEDDING_DIM)),
        )
        self.register_parameter(
            "domain_latents",
            torch.nn.Parameter(torch.randn(NUM_DOMAIN_BLOCKS, EMBEDDING_DIM)),
        )
        self.register_buffer("range_positional", range_positional)
        self.register_buffer("domain_positional", domain_positional)
        self.range_proj = torch.nn.Linear(EMBEDDING_DIM, ATTENTION_DIM)
        self.domain_proj = torch.nn.Linear(EMBEDDING_DIM, ATTENTION_DIM)
        self.range_contrast_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)
        self.domain_contrast_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)
        self.range_offset_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)
        self.domain_offset_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)

    def forward(self, pooled_domains: torch.Tensor) -> torch.Tensor:
        range_base = self.range_latents + self.range_positional
        domain_base = self.domain_latents + self.domain_positional

        range_repr = self.range_proj(range_base)
        domain_repr = self.domain_proj(domain_base)

        range_contrast = self.range_contrast_proj(range_base)
        domain_contrast = self.domain_contrast_proj(domain_base)
        contrast_logits = (range_contrast @ domain_contrast.t()) / math.sqrt(AFFINE_HIDDEN_DIM)
        contrast = torch.tanh(contrast_logits) * MAX_CONTRAST  # (R, D)

        range_offset = self.range_offset_proj(range_base)
        domain_offset = self.domain_offset_proj(domain_base)
        offset_logits = (range_offset @ domain_offset.t()) / math.sqrt(AFFINE_HIDDEN_DIM)
        offset = torch.tanh(offset_logits)  # (R, D)

        logits = (range_repr @ domain_repr.t()) / (math.sqrt(range_repr.shape[-1]) * SOFTMAX_TEMPERATURE)
        weights = torch.softmax(logits, dim=-1)

        domain_flat = pooled_domains.view(NUM_DOMAIN_BLOCKS, -1)
        scaled_domains = torch.matmul(weights * contrast, domain_flat)
        offset_effective = (weights * offset).sum(dim=1, keepdim=True)
        decoded = (scaled_domains + offset_effective).view(-1, RANGE_SIZE, RANGE_SIZE)
        return decoded


def main() -> None:
    args = parse_args()
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    checkpoint_path = args.checkpoint_path
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    else:
        torch.mps.manual_seed(SEED)  # type: ignore[attr-defined]

    image = np.asarray(Image.open(IMAGE_PATH).convert("L"), dtype=np.float32) / 255.0
    image_t = torch.from_numpy(image).to(device)

    range_blocks = get_blocks(image_t, RANGE_SIZE)
    pooled_domains = get_pooled_blocks(image_t, RANGE_SIZE)
    domain_positional = torch.zeros(NUM_DOMAIN_BLOCKS, EMBEDDING_DIM, device=device)
    range_positional = torch.zeros(NUM_RANGE_BLOCKS, EMBEDDING_DIM, device=device)

    model = FractalAttention(range_positional, domain_positional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        start = time.perf_counter()
        pred_blocks = model(pooled_domains)
        loss = (pred_blocks - range_blocks).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        else:
            torch.mps.synchronize()  # type: ignore[attr-defined]

        duration = time.perf_counter() - start
        print(
            f"Epoch {epoch + 1}/{num_epochs} - block MSE: {fmt(loss.item())} "
            f"- {duration:.3f}s"
        )

    model.eval()

    def collage_once(source_image: torch.Tensor) -> torch.Tensor:
        pooled = get_pooled_blocks(source_image, RANGE_SIZE)
        blocks = model(pooled)
        return torch.clamp(reassemble_blocks(blocks, IMAGE_SIZE), 0.0, 1.0)

    with torch.no_grad():
        final_blocks = torch.clamp(model(pooled_domains), 0.0, 1.0)
        per_block_mse = (final_blocks - range_blocks).pow(2).mean(dim=(1, 2))

        blocks_per_side = IMAGE_SIZE // RANGE_SIZE
        target_linear_index = TARGET_RANGE_COORD[0] * blocks_per_side + TARGET_RANGE_COORD[1]
        target_block_error = float(per_block_mse[target_linear_index].item())
        mean_error = float(per_block_mse.mean().item())
        median_error = float(per_block_mse.median().item())

        collage = reassemble_blocks(final_blocks, IMAGE_SIZE)
        collage_error = float((collage - image_t).pow(2).mean().item())

        rng = np.random.default_rng(42)
        reconstruction = torch.from_numpy(
            rng.random((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        ).to(device)
        for _ in range(16):
            reconstruction = collage_once(reconstruction)

        psnr = 10 * np.log10(1.0 / (reconstruction - image_t).pow(2).mean().item())

    print(f"Mean block MSE: {fmt(mean_error)}")
    print(f"Median block MSE: {fmt(median_error)}")
    print(f"Target block {TARGET_RANGE_COORD} MSE: {fmt(target_block_error)}")
    print(f"Collage error (MSE): {fmt(collage_error)}")
    print(f"PSNR after 16 iterations: {fmt(psnr)}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

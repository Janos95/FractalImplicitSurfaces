import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import torch

from neural_compress import (
    EMBEDDING_DIM,
    FractalAttention,
    IMAGE_SIZE,
    NUM_DOMAIN_BLOCKS,
    NUM_RANGE_BLOCKS,
    RANGE_SIZE,
    device,
    get_pooled_blocks,
    reassemble_blocks,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "fractal_attention_weights.pt"
SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive neural fractal decompression.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained weights (default: %(default)s)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> FractalAttention:
    range_positional = torch.zeros(NUM_RANGE_BLOCKS, EMBEDDING_DIM, device=device)
    domain_positional = torch.zeros(NUM_DOMAIN_BLOCKS, EMBEDDING_DIM, device=device)
    model = FractalAttention(range_positional, domain_positional).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def collage_once(model: FractalAttention, source_image: torch.Tensor) -> torch.Tensor:
    pooled = get_pooled_blocks(source_image, RANGE_SIZE)
    blocks = model(pooled)
    collage = torch.clamp(reassemble_blocks(blocks, IMAGE_SIZE), 0.0, 1.0)
    return collage


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = load_model(args.checkpoint)

    rng = np.random.default_rng(SEED)
    image = torch.from_numpy(rng.random((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)).to(device)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    im = ax.imshow((image.cpu().numpy() * 255.0).clip(0.0, 255.0), cmap="gray", vmin=0, vmax=255)
    ax.set_title("Iteration 0")
    ax.axis("off")

    iteration = {"count": 0, "image": image}

    def do_iteration(_=None) -> None:
        with torch.no_grad():
            iteration["image"] = collage_once(model, iteration["image"])
        iteration["count"] += 1
        im.set_data((iteration["image"].cpu().numpy() * 255.0).clip(0.0, 255.0))
        ax.set_title(f"Iteration {iteration['count']}")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == " ":
            do_iteration()

    button_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    button = Button(button_ax, "Iterate")
    button.on_clicked(do_iteration)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()

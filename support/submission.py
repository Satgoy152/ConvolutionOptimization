#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
from torch.utils.data import TensorDataset
from collections import defaultdict
import warnings

from reader import load_mnist

os.environ["CUDA_CACHE_DISABLE"] = "1"
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
torch.backends.cudnn.benchmark = True

MODEL_DIR = "support"
model_prefix = "eecs471"
model_path = os.path.join(MODEL_DIR, f"{model_prefix}-model.pth")

time_dict = defaultdict(
    lambda: [torch.cuda.Event(enable_timing=True) for _ in range(2)]
)


def load_fashion_mnist(path, dataset_size, device):
    test_images, test_labels = load_mnist(path, rows=72, cols=72, kind="t10k-72")
    images = torch.tensor(test_images, dtype=torch.float32, device=device)
    labels = torch.tensor(test_labels, dtype=torch.float32, device=device)

    dataset = TensorDataset(images, labels)
    return torch.utils.data.Subset(dataset, range(dataset_size))


def take_time_pre(layer, input):
    torch.cuda.nvtx.range_push("Convolution" if type(layer) == KERNEL else str(layer))
    time_dict[layer][0].record()


def take_time(layer_name, input, output):
    time_dict[layer_name][1].record()
    torch.cuda.nvtx.range_pop()


class Forward(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )

    def forward(self, input):
        return torch.ops.eecs471.forward(input, self.weight, self.out_channels)


KERNEL = Forward  # nn.Conv2d #


def load_model(device):
    model: nn.Module = nn.Sequential(
        KERNEL(1, 12, kernel_size=7, bias=False),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        KERNEL(12, 24, kernel_size=7, bias=False),
        nn.Flatten(),
        nn.Linear(27 * 27 * 24, 160),
        nn.Tanh(),
        nn.Linear(160, 10),
    ).to(device)
    model.eval()

    for layer in model.children():
        layer.register_forward_pre_hook(take_time_pre)
        layer.register_forward_hook(take_time)
    return model


@torch.no_grad()
def evaluate(model, test_loader, zero=False):
    correct = total = 0

    for images, labels in test_loader:
        outputs = model(torch.zeros_like(images) if zero else images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()

    return correct / total


def main():
    dataset_size = 10000
    # Parse command line arguments
    if len(sys.argv) > 1:
        dataset_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        print("Usage:", sys.argv[0], "<dataset size>")
        print("    <dataset_size> = [0 - 10000]")
        sys.exit(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading fashion-mnist data...")
    test_dataset = load_fashion_mnist("fashion-mnist", dataset_size, device)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # Use full dataset as one batch
        shuffle=False,
    )
    state_dict = torch.load(model_path, weights_only=True)

    # warm up CUDA by running kernel with uninitialized weights
    # (so students can't just cache the answer)
    print("Warming up CUDA kernels...")
    if KERNEL == Forward:
        torch.ops.load_library("support/build/libeecs471.so")
    model = load_model(device)
    torch.cuda.nvtx.range_push("Warmup")
    evaluate(model, test_loader, zero=True)
    torch.cuda.nvtx.range_pop()

    print("Loading model weights...")
    model.load_state_dict(state_dict)
    print("Running New Inference")
    torch.cuda.nvtx.range_push("Inference")
    accuracy = evaluate(model, test_loader)
    torch.cuda.nvtx.range_pop()
    for k, (s, e) in time_dict.items():
        if type(k) == KERNEL:
            print("Op Time:", s.elapsed_time(e) / 1000)
    print(f"Correctness: {accuracy:.4f} Model: {model_prefix}")


if __name__ == "__main__":
    main()

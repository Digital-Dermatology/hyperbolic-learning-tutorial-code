import os

import torch
import torchvision
from torch.optim import AdamW
from torchinfo import summary
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef
from tqdm import tqdm

from nets.euclidean import make_euclidean_convnet
from utils.torch_utils import get_available_device, set_seeds

num_workers = 0
batch_size = 128
num_epochs = 10


def evaluate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    metrics: MetricCollection,
    device: torch.device,
):
    metrics.reset()
    with torch.no_grad():
        for data, labels in loader:
            outputs = model(data.to(device))
            metrics(outputs, labels.to(device))


def print_metrics(metrics: MetricCollection) -> None:
    print({k.replace("Multiclass", ""): v.item() for k, v in metrics.compute().items()})


if __name__ == "__main__":
    device = torch.device(get_available_device())
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root="data", train=False, download=True, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    classes = train_dataset.classes
    assert test_dataset.classes == classes
    num_classes = len(classes)

    convnet = make_euclidean_convnet(out_channels=num_classes, conv_channels=(64, 64), fc_channels=(128,))
    convnet.to(device)
    summary(convnet)

    metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassMatthewsCorrCoef(num_classes=num_classes),
        ]
    )
    metrics.to(device)

    evaluate(loader=test_dataloader, model=convnet, metrics=metrics, device=device)
    print("Metrics before training:")
    print_metrics(metrics)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = AdamW(convnet.parameters())

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        for data, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = convnet(data.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
        evaluate(loader=test_dataloader, model=convnet, metrics=metrics, device=device)
        print_metrics(metrics)

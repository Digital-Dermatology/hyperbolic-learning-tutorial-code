import os

import torch
import torchvision
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef

from nets.euclidean import make_euclidean_convnet

num_workers = 0#os.cpu_count()
batch_size = 128


def evaluate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    metrics: MetricCollection,
):
    with torch.no_grad():
        for data, labels in loader:
            outputs = model(data)
            metrics(outputs, labels)


if __name__ == "__main__":
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

    convnet = make_euclidean_convnet(out_channels=num_classes)

    metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassMatthewsCorrCoef(num_classes=num_classes),
        ]
    )

    evaluate(loader=test_dataloader, model=convnet, metrics=metrics)
    print(metrics.compute())

    # criterion = torch.nn.CrossEntropyLoss()
    # from torch.optim import AdamW
    #
    # optimizer = AdamW(convnet.parameters())
    #
    # for x, y in train_dataset:
    #     print(x.min(), x.max())
    #     break

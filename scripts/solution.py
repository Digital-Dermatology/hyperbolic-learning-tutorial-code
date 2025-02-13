from typing import Tuple

import pandas as pd
import plotly.express as px
import torch
import torchvision
from hypll.optim import RiemannianAdam
from hypll.tensors import ManifoldTensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassMatthewsCorrCoef
from tqdm import tqdm

from src.nets.euclidean import make_euclidean_net
from src.nets.fully_hyperbolic import make_fully_hyperbolic_net
from src.nets.last_hyperbolic import make_last_hyperbolic_net
from src.utils.torch_utils import get_available_device

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


def print_metrics(metrics: MetricCollection, prefix: str = "") -> None:
    print(
        prefix,
        {k.replace("Multiclass", ""): v.item() for k, v in metrics.compute().items()},
    )


def compute_embeddings(
    loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings, predictions, labels = [], [], []
    with torch.no_grad():
        for data, labels_batch in loader:
            embeddings_batch = model[:-1](data.to(device))
            predictions_batch = model[-1](embeddings_batch).argmax(dim=-1)
            if isinstance(embeddings_batch, ManifoldTensor):
                embeddings_batch = embeddings_batch.tensor
            embeddings.append(embeddings_batch)
            predictions.append(predictions_batch)
            labels.append(labels_batch)
    embeddings = torch.cat(embeddings, dim=0)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, predictions, labels


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

    train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
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

    metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassMatthewsCorrCoef(num_classes=num_classes),
        ]
    )
    metrics.to(device)

    parameters = {
        "out_channels": num_classes,
        "conv_channels": (32, 64, 128),
        "fc_channels": (128, 32, 2),
    }
    models = {
        "euclidean": make_euclidean_net(**parameters),
        "last_hyperbolic": make_last_hyperbolic_net(**parameters),
        "fully_hyperbolic": make_fully_hyperbolic_net(**parameters),
    }

    for model_name, model in models.items():
        model.to(device)
        print(model)

        evaluate(loader=test_dataloader, model=model, metrics=metrics, device=device)
        print("Metrics before training:")
        print_metrics(metrics)

        criterion = torch.nn.CrossEntropyLoss()
        criterion.to(device)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1} of {num_epochs}")
            metrics.reset()
            for data, labels in tqdm(train_dataloader):
                optimizer.zero_grad()
                outputs = model(data.to(device))
                labels = labels.to(device)
                metrics(outputs, labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print_metrics(metrics, "Train: ")
            evaluate(
                loader=test_dataloader, model=model, metrics=metrics, device=device
            )
            print_metrics(metrics, "Test: ")

        embeddings, predictions, labels = compute_embeddings(
            loader=test_dataloader, model=model, device=device
        )
        df = pd.DataFrame(embeddings.cpu().numpy())
        df["prediction"] = predictions.cpu().numpy()
        df["label"] = [classes[i] for i in labels.cpu().numpy()]
        fig = px.scatter(data_frame=df, x=0, y=1, color="label")
        fig.update_layout(yaxis_scaleanchor="x")
        if model_name != "euclidean":
            curvature = model[-1].manifold.c().item()
            radius = curvature**-0.5
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=-radius,
                y0=-radius,
                x1=radius,
                y1=radius,
            )
        fig.write_html(f"{model_name}_{num_epochs}.html")

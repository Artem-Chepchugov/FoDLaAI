import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import os

from models.cnn_models import (
    SimpleCNNWithKernel,
    HybridCNN,
    ShallowCNN,
    MediumCNN,
    DeepCNN,
    CNNWithResidual,
)

from utils.training_utils import train_model, evaluate_model
from utils.visualization_utils import (
    plot_loss_curves,
    plot_confusion_matrix,
    plot_activation_maps,
    plot_gradient_histogram,
)


def analyze_gradient_flow(model, model_name="model_name"):
    """
    Анализирует поток градиентов в модели и сохраняет гистограмму их значений.

    Args:
        model (nn.Module): Обученная модель.
        model_name (str): Название модели для сохранения графика.
    """
    grads = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.abs().mean().item())

    if grads:
        dataset_name = getattr(model, "dataset_name", "dataset")
        plot_gradient_histogram(
            grads, model_name, f"plots/{dataset_name}_{model_name}_gradient_flow"
        )
    else:
        logging.warning(f"Нет доступных градиентов для анализа модели {model_name}.")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(name):
    """
    Загружает датасет MNIST или CIFAR-10 и возвращает соответствующие загрузчики данных.

    Args:
        name (str): Название датасета ("MNIST" или "CIFAR-10").

    Returns:
        tuple: (train_loader, test_loader, in_channels, num_classes)
    """
    if name == "MNIST":
        transform = transforms.ToTensor()
        train_set = datasets.MNIST(
            root="data", train=True, transform=transform, download=True
        )
        test_set = datasets.MNIST(
            root="data", train=False, transform=transform, download=True
        )
        in_channels = 1
        num_classes = 10
    elif name == "CIFAR-10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(
            root="data", train=True, transform=transform, download=True
        )
        test_set = datasets.CIFAR10(
            root="data", train=False, transform=transform, download=True
        )
        in_channels = 3
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader, in_channels, num_classes


def run_kernel_size_experiments(dataset_name):
    """
    Запускает эксперименты по сравнению моделей с различными размерами ядер свёртки.

    Args:
        dataset_name (str): Название датасета ("MNIST" или "CIFAR-10").
    """
    logging.info(f"\n=== Kernel Size Experiment: {dataset_name} ===")
    train_loader, test_loader, in_channels, num_classes = load_dataset(dataset_name)

    models = {
        "Kernel3x3_v2": SimpleCNNWithKernel(in_channels, 3, num_classes),
        "Kernel5x5_v2": SimpleCNNWithKernel(in_channels, 5, num_classes),
        "Kernel7x7_v2": SimpleCNNWithKernel(in_channels, 7, num_classes),
        "Hybrid1x1_3x3_v2": HybridCNN(in_channels, num_classes),
    }
    losses = {}
    results = {}
    for name, model in models.items():
        logging.info(f"\nTraining: {name}")
        train_time, train_losses, avg_gradients = train_model(
            model, train_loader, device
        )
        accuracy, infer_time, y_pred, y_true = evaluate_model(
            model, test_loader, device
        )
        model.dataset_name = dataset_name
        analyze_gradient_flow(model, model_name=name)

        losses[name] = train_losses
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # ===
        results[name] = {
            "acc": accuracy,
            "params": params,
            "train_time": train_time,
            "infer_time": infer_time,
        }

        sample_batch = next(iter(test_loader))[0].to(device)
        activation = model.conv(sample_batch[:1])
        plot_activation_maps(
            activation,
            layer_name=name,
            filename_prefix=f"plots/{dataset_name}_kernel_{name}",
        )
        plot_confusion_matrix(
            y_true,
            y_pred,
            list(map(str, range(num_classes))),
            f"plots/cm_{dataset_name}_kernel_{name}.png",
        )
    plot_loss_curves(losses, f"{dataset_name}_kernels_v2")

    print(f"\n=== Сравнение моделей на {dataset_name} (Kernel Size) ===")
    print(f"{'Model':30} {'Acc':<10} {'Params':<12} {'TrainTime':<12} {'InferTime'}")
    for name, r in results.items():
        print(
            f"{name:30} {r['acc']:<10.4f} {r['params']:<12} {r['train_time']:<12.2f} {r['infer_time']:.4f}"
        )


def run_depth_experiments(dataset_name):
    """
    Запускает эксперименты по сравнению моделей с различной глубиной сети.

    Args:
        dataset_name (str): Название датасета ("MNIST" или "CIFAR-10").
    """
    logging.info(f"\n=== Depth Experiment: {dataset_name} ===")
    train_loader, test_loader, in_channels, num_classes = load_dataset(dataset_name)
    models = {
        "ShallowCNN_v2": ShallowCNN(in_channels, num_classes),
        "MediumCNN_v2": MediumCNN(in_channels, num_classes),
        "DeepCNN_v2": DeepCNN(in_channels, num_classes),
        "CNNWithResidual_v2": CNNWithResidual(in_channels, num_classes),
    }
    losses = {}
    results = {}
    for name, model in models.items():
        logging.info(f"\nTraining: {name}")
        train_time, train_losses, avg_gradients = train_model(
            model, train_loader, device
        )
        accuracy, infer_time, y_pred, y_true = evaluate_model(
            model, test_loader, device
        )
        model.dataset_name = dataset_name
        analyze_gradient_flow(model, model_name=name)

        losses[name] = train_losses
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[name] = {
            "acc": accuracy,
            "params": params,
            "train_time": train_time,
            "infer_time": infer_time,
        }

        sample_batch = next(iter(test_loader))[0].to(device)
        activation = model.conv(sample_batch[:1])
        plot_activation_maps(
            activation,
            layer_name=name,
            filename_prefix=f"plots/{dataset_name}_depth_{name}",
        )
        plot_confusion_matrix(
            y_true,
            y_pred,
            list(map(str, range(num_classes))),
            f"plots/cm_{dataset_name}_depth_{name}.png",
        )
    plot_loss_curves(losses, f"{dataset_name}_depth_v2")

    print(f"\n=== Сравнение моделей на {dataset_name} (Depth) ===")
    print(f"{'Model':30} {'Acc':<10} {'Params':<12} {'TrainTime':<12} {'InferTime'}")
    for name, r in results.items():
        print(
            f"{name:30} {r['acc']:<10.4f} {r['params']:<12} {r['train_time']:<12.2f} {r['infer_time']:.4f}"
        )


if __name__ == "__main__":
    for dataset in ["MNIST", "CIFAR-10"]:
        run_kernel_size_experiments(dataset)
        run_depth_experiments(dataset)

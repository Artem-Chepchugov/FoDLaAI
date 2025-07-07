import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import os

from models.fc_models import FullyConnectedNet
from models.cnn_models import (
    SimpleCNN,
    CNNWithResidual,
    CNNWithResidualAndRegularization,
)
from utils.training_utils import train_model, evaluate_model
from utils.visualization_utils import plot_confusion_matrix, plot_loss_curves

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
os.makedirs("plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiments():
    datasets_info = [
        {
            "name": "MNIST",
            "input_size": 28 * 28,
            "in_channels": 1,
            "num_classes": 10,
            "dataset": datasets.MNIST,
            "transform": transforms.ToTensor(),
            "use_regularized_cnn": False,
            "include_simple_cnn": True,
        },
        {
            "name": "CIFAR-10",
            "input_size": 32 * 32 * 3,
            "in_channels": 3,
            "num_classes": 10,
            "dataset": datasets.CIFAR10,
            "transform": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
            "use_regularized_cnn": True,
            "include_simple_cnn": False,
        },
    ]

    for ds in datasets_info:
        logging.info(f"\n=== {ds['name']} ===")
        train_set = ds["dataset"](
            root="data", train=True, download=True, transform=ds["transform"]
        )
        test_set = ds["dataset"](
            root="data", train=False, download=True, transform=ds["transform"]
        )
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64)

        # Словарь моделей
        models = {
            "FullyConnectedNet": FullyConnectedNet(ds["input_size"], ds["num_classes"]),
            "CNNWithResidual": CNNWithResidual(ds["in_channels"], ds["num_classes"]),
        }
        if ds["include_simple_cnn"]:
            models["SimpleCNN"] = SimpleCNN(ds["in_channels"], ds["num_classes"])
        if ds["use_regularized_cnn"]:
            models["CNNWithResidualAndRegularization"] = (
                CNNWithResidualAndRegularization(ds["in_channels"], ds["num_classes"])
            )

        results = {}
        losses = {}

        for name, model in models.items():
            logging.info(f"\nTraining: {name}")
            train_time, loss_curve, avg_gradients = train_model(
                model, train_loader, device, analyze_gradients=True
            )
            acc, infer_time, preds, targets = evaluate_model(model, test_loader, device)

            results[name] = {
                "acc": acc,
                "params": sum(p.numel() for p in model.parameters()),
                "train_time": train_time,
                "infer_time": infer_time,
            }
            losses[name] = loss_curve

            if avg_gradients:
                logging.info(f"Gradient magnitudes for {name}:")
                for i, g in enumerate(avg_gradients):
                    logging.info(f"  Layer {i}: {g:.6f}")

                mean_grad = sum(avg_gradients) / len(avg_gradients)
                logging.info(f"Average gradient magnitude for {name}: {mean_grad:.6f}")

            plot_confusion_matrix(
                targets,
                preds,
                classes=list(range(ds["num_classes"])),
                filename=f"plots/cm_{ds['name']}_{name}.png",
            )

        # Таблица сравнения
        print(f"\n=== Сравнение моделей на {ds['name']} ===")
        print(
            f"{'Model':30} {'Acc':<10} {'Params':<12} {'TrainTime':<12} {'InferTime'}"
        )
        for name, r in results.items():
            print(
                f"{name:30} {r['acc']:<10.4f} {r['params']:<12} {r['train_time']:<12.2f} {r['infer_time']:.4f}"
            )

        # Кривые потерь
        plot_loss_curves(losses, ds["name"])


if __name__ == "__main__":
    run_experiments()

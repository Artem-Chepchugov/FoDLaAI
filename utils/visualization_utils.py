import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import os
import torch

# Создаём папку для графиков, если нет
os.makedirs("plots", exist_ok=True)


def plot_loss_curves(loss_dict, dataset_name):
    """
    Строит и сохраняет графики кривых обучения (потеря на эпохах) для нескольких моделей.

    Параметры:
    - loss_dict: dict, ключ — имя модели, значение — список значений loss по эпохам
    - dataset_name: str, имя датасета (для подписи и имени файла)
    """
    plt.figure(figsize=(8, 6))
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curves - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    filename = f"plots/loss_curves_{dataset_name}.png"
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved loss curves plot: {filename}")


def plot_confusion_matrix(y_true, y_pred, classes, filename):
    """
    Строит и сохраняет матрицу ошибок.

    Параметры:
    - y_true: list или numpy array, истинные метки
    - y_pred: list или numpy array, предсказанные метки
    - classes: list, имена классов
    - filename: str, путь для сохранения изображения
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved confusion matrix plot: {filename}")


def plot_activation_maps(activation_tensor, layer_name, filename_prefix, max_maps=8):
    """
    Визуализация активаций сверточного слоя.

    Параметры:
    - activation_tensor: torch.Tensor, форма [batch_size, channels, H, W], активации слоя
    - layer_name: str, имя слоя (для заголовка)
    - filename_prefix: str, путь и префикс для сохранения файлов
    - max_maps: int, сколько каналов визуализировать (максимум)
    """
    activation = activation_tensor.detach().cpu()
    batch_size, channels, h, w = activation.shape
    n_maps = min(channels, max_maps)

    plt.figure(figsize=(n_maps * 2, 2))
    for i in range(n_maps):
        plt.subplot(1, n_maps, i + 1)
        plt.imshow(activation[0, i, :, :], cmap="viridis")
        plt.axis("off")
        plt.title(f"Map {i}")
    plt.suptitle(f"Activation Maps: {layer_name}")
    plt.tight_layout()
    filename = f"{filename_prefix}_activation_maps.png"
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved activation maps plot: {filename}")


def plot_feature_maps(feature_maps, layer_name, filename_prefix, max_maps=8):
    """
    Аналогично plot_activation_maps, для визуализации feature maps после любого слоя.
    """
    plot_activation_maps(feature_maps, layer_name, filename_prefix, max_maps)


def plot_gradient_histogram(grads, model_name, filename_prefix):
    """
    Визуализация распределения градиентов по слоям модели (анализ vanishing/exploding gradients).

    Параметры:
    - grads: list или numpy array, список средних абсолютных значений градиентов параметров
    - model_name: str, имя модели (для заголовка)
    - filename_prefix: str, путь и префикс для сохранения файла
    """
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(grads)), grads)
    plt.xlabel("Layer Index")
    plt.ylabel("Average Gradient Magnitude")
    plt.title(f"Gradient Flow Histogram - {model_name}")
    plt.tight_layout()
    filename = f"{filename_prefix}_gradient_histogram.png"
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved gradient histogram plot: {filename}")


def plot_attention_map(attention_map, layer_name, filename_prefix):
    """
    Визуализация attention карты (задание 3.1).

    Параметры:
    - attention_map: torch.Tensor, форма [H, W], карта внимания
    - layer_name: str, имя слоя
    - filename_prefix: str, путь и префикс для сохранения
    """
    attn = attention_map.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(attn, cmap="hot")
    plt.colorbar()
    plt.title(f"Attention Map - {layer_name}")
    plt.tight_layout()
    filename = f"{filename_prefix}_attention_map.png"
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved attention map plot: {filename}")

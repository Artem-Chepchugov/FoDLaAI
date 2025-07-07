import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
import logging
import matplotlib.pyplot as plt
from utils.visualization_utils import (
    plot_loss_curves,
    plot_confusion_matrix,
    plot_activation_maps,
    plot_attention_map,
    plot_gradient_histogram,
)
from utils.training_utils import train_model, evaluate_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Кастомный сверточный слой с дополнительной логикой
class CustomConvFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        # обычная сверточная операция с ReLU
        output = F.conv2d(input, weight, bias, stride=stride, padding=padding)
        output_relu = F.relu(output)
        ctx.output_relu = output_relu
        return output_relu

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        # Градиент ReLU
        grad_relu = grad_output.clone()
        grad_relu[ctx.output_relu <= 0] = 0

        # Градиенты параметров свертки
        grad_input = torch.nn.grad.conv2d_input(
            input.shape, weight, grad_relu, stride=stride, padding=padding
        )
        grad_weight = torch.nn.grad.conv2d_weight(
            input, weight.shape, grad_relu, stride=stride, padding=padding
        )
        grad_bias = grad_relu.sum(dim=[0, 2, 3]) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None


class CustomConvLayer(nn.Module):
    """
    Обёртка над CustomConvFunction, как nn.Module с параметрами.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return CustomConvFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding
        )


# Attention механизм для CNN
class SimpleSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map = self.sigmoid(
            self.conv1(x)
        )  # карта внимания [B,1,H,W], значения в [0,1]
        out = x * attn_map  # масштабирование признаков по пространству
        self.attention_map = attn_map.detach()
        return out


# Кастомная функция активации: ParametricSwish
class ParametricSwish(nn.Module):
    """
    Parametric Swish: x * sigmoid(alpha * x), alpha — обучаемый параметр
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)


# Кастомный pooling слой: MixedPooling (сумма max + avg)
class MixedPooling(nn.Module):
    """
    Смешанный pooling: max pooling + average pooling, сумма результатов.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.max_pool(x) + self.avg_pool(x)


# Residual блоки


class BasicResidualBlock(nn.Module):
    """
    Базовый Residual блок с 2 сверточными слоями 3x3
    """

    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleneckResidualBlock(nn.Module):
    """
    Bottleneck Residual блок (1x1 -> 3x3 -> 1x1) для уменьшения/увеличения каналов
    """

    def __init__(self, in_channels, bottleneck_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, 3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class WideResidualBlock(nn.Module):
    """
    Wide Residual блок: расширенный канал, простая схема 3x3 -> 3x3 с увеличенной шириной
    """

    def __init__(self, in_channels, width_factor=2, stride=1):
        super().__init__()
        out_channels = in_channels * width_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# Модели для теста кастомных слоев и residual блоков


class CustomCNNWithCustomLayers(nn.Module):
    """
    CNN с кастомными слоями из 3.1 для теста
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.custom_conv = CustomConvLayer(in_channels, 16, kernel_size=3, padding=1)
        self.attention = SimpleSpatialAttention(16)
        self.custom_pool = MixedPooling(kernel_size=2, stride=2)
        self.activation = ParametricSwish()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.custom_conv(x)
        x = self.attention(x)
        x = self.custom_pool(x)
        x = self.activation(x)
        return self.classifier(x)


class CNNWithResidualBlocks(nn.Module):
    """
    Модель для сравнения трёх вариантов residual блоков
    """

    def __init__(self, block_type, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        if block_type == "basic":
            self.res_block1 = BasicResidualBlock(self.in_channels, self.in_channels)
            self.res_block2 = BasicResidualBlock(self.in_channels, self.in_channels)
        elif block_type == "bottleneck":
            self.res_block1 = BottleneckResidualBlock(
                self.in_channels, bottleneck_channels=8, out_channels=self.in_channels
            )
            self.res_block2 = BottleneckResidualBlock(
                self.in_channels, bottleneck_channels=8, out_channels=self.in_channels
            )
        elif block_type == "wide":
            self.res_block1 = WideResidualBlock(self.in_channels, width_factor=2)
            self.res_block2 = WideResidualBlock(self.in_channels * 2, width_factor=2)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        out_channels = (
            self.res_block2.bn2.num_features
            if block_type != "bottleneck"
            else self.res_block2.bn3.num_features
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


# Тестирование и сравнение


def test_custom_layers():
    logging.info("Testing custom layers...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Тест кастомного сверточного слоя
    x = torch.randn(2, 3, 8, 8).to(device)
    custom_conv = CustomConvLayer(3, 6, kernel_size=3, padding=1).to(device)
    out = custom_conv(x)
    logging.info(f"CustomConvLayer output shape: {out.shape}")

    # Тест attention
    attention = SimpleSpatialAttention(6).to(device)
    out_att = attention(out)
    logging.info(f"Attention output shape: {out_att.shape}")

    # Тест активации
    activation = ParametricSwish().to(device)
    out_act = activation(out_att)
    logging.info(f"ParametricSwish output shape: {out_act.shape}")

    # Тест кастомного pooling
    pooling = MixedPooling(2).to(device)
    out_pool = pooling(out_act)
    logging.info(f"MixedPooling output shape: {out_pool.shape}")

    logging.info("Custom layers tests passed.\n")


def compare_residual_blocks(train_loader, test_loader, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for block_type in ["basic", "bottleneck", "wide"]:
        logging.info(f"Training model with {block_type} residual blocks...")
        model = CNNWithResidualBlocks(block_type, in_channels=1).to(device)

        train_time, losses, grads = train_model(
            model, train_loader, device, epochs=epochs, analyze_gradients=True
        )
        accuracy, infer_time, preds, targets = evaluate_model(
            model, test_loader, device
        )

        results[block_type] = {
            "model": model,
            "train_time": train_time,
            "losses": losses,
            "accuracy": accuracy,
            "infer_time": infer_time,
            "avg_gradients": grads,
            "params_count": sum(p.numel() for p in model.parameters()),
        }
        logging.info(
            f"{block_type} block: train_time={train_time:.2f}s, accuracy={accuracy:.4f}, params={results[block_type]['params_count']}"
        )

    # Визуализация
    loss_dict = {k: v["losses"] for k, v in results.items()}
    plot_loss_curves(loss_dict, "ResidualBlocks")

    for k, v in results.items():
        plot_gradient_histogram(v["avg_gradients"], k, f"plots/{k}_resblock")

    return results


def main_experiment(train_loader, test_loader, epochs=5):
    logging.info("Starting experiments with custom layers and residual blocks...")

    # 3.1 Тестируем кастомные слои отдельно
    test_custom_layers()

    # 3.2 Сравниваем residual блоки
    resblock_results = compare_residual_blocks(train_loader, test_loader, epochs=epochs)

    # Пример работы кастомной модели с кастомными слоями (MNIST-like)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNNWithCustomLayers(in_channels=1, num_classes=10).to(device)
    train_time, losses, grads = train_model(
        model, train_loader, device, epochs=epochs, analyze_gradients=True
    )
    accuracy, infer_time, preds, targets = evaluate_model(model, test_loader, device)

    logging.info(
        f"CustomCNNWithCustomLayers: train_time={train_time:.2f}s, accuracy={accuracy:.4f}, params={sum(p.numel() for p in model.parameters())}"
    )

    plot_loss_curves({"CustomCNNWithCustomLayers": losses}, "CustomCNNWithCustomLayers")
    plot_gradient_histogram(grads, "CustomCNNWithCustomLayers", "plots/custom_cnn")

    # Визуализация attention map из модели
    if hasattr(model.attention, "attention_map"):
        plot_attention_map(
            model.attention.attention_map[0, 0], "Attention", "plots/custom_cnn"
        )

    return resblock_results


if __name__ == "__main__":
    # Для запуска нужен подготовленный train_loader и test_loader
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    results = main_experiment(train_loader, test_loader, epochs=3)

import torch.nn as nn


class FullyConnectedNet(nn.Module):
    """Полносвязная сеть с 3 скрытыми слоями для MNIST и CIFAR-10."""

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_model(model, train_loader, device, epochs=5, analyze_gradients=False):
    """
    Обучение модели.

    Параметры:
    - model: torch.nn.Module, обучаемая модель
    - train_loader: DataLoader, загрузчик обучающих данных
    - device: torch.device, устройство для вычислений (CPU/GPU)
    - epochs: int, число эпох обучения
    - analyze_gradients: bool, анализировать ли градиенты после последнего backward

    Возвращает:
    - train_time: float, время обучения в секундах
    - losses: list, список средних потерь по эпохам
    - avg_gradients: list, средние градиенты по слоям (если analyze_gradients=True), иначе None
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    losses = []
    avg_gradients = None

    for epoch in range(epochs):
        total_loss = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if analyze_gradients and epoch == epochs - 1 and i == len(train_loader) - 1:
                grads = []
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grads.append(param.grad.abs().mean().item())
                avg_gradients = grads
                logging.info("Gradient analysis completed.")

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        losses.append(avg_loss)

    train_time = time.time() - start_time
    return train_time, losses, avg_gradients


def evaluate_model(model, test_loader, device):
    """
    Оценка модели на тестовом наборе.

    Параметры:
    - model: torch.nn.Module, обученная модель
    - test_loader: DataLoader, загрузчик тестовых данных
    - device: torch.device, устройство для вычислений

    Возвращает:
    - accuracy: float, точность модели
    - infer_time: float, время инференса в секундах
    - all_preds: list, предсказания модели
    - all_targets: list, истинные метки
    """
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    infer_time = time.time() - start_time
    accuracy = correct / total
    logging.info(f"Evaluation accuracy: {accuracy:.4f}")
    return accuracy, infer_time, all_preds, all_targets

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Классы Dataset
class RegressionDataset(Dataset):
    """
    Dataset для задачи регрессии.

    Args:
        X (array-like): Признаки.
        y (array-like): Целевая переменная.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationDataset(Dataset):
    """
    Dataset для задачи классификации.

    Args:
        X (array-like): Признаки.
        y (array-like): Метки классов.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Модель линейной регрессии
class LinearRegressionModel(nn.Module):
    """
    Линейная регрессия на основе nn.Linear.

    Args:
        in_features (int): Количество входных признаков.
    """

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


# Модель логистической регрессии (мультиклассовая)
class LogisticRegressionModel(nn.Module):
    """
    Логистическая регрессия с поддержкой многоклассовой классификации.

    Args:
        in_features (int): Количество входных признаков.
        num_classes (int): Количество классов.
    """

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


# Функция обучения линейной регрессии
def train_linear_model(
    model,
    dataloader,
    lr=0.01,
    epochs=100,
    l1_lambda=0.0,
    l2_lambda=0.0,
    early_stopping_delta=1e-4,
    early_stopping_patience=5,
):
    """
    Обучает модель линейной регрессии с L1/L2 регуляризацией и early stopping.

    Args:
        model (nn.Module): Модель линейной регрессии.
        dataloader (DataLoader): Загрузчик обучающего датасета.
        lr (float): Скорость обучения.
        epochs (int): Максимальное количество эпох обучения.
        l1_lambda (float): Коэффициент L1 регуляризации.
        l2_lambda (float): Коэффициент L2 регуляризации.
        early_stopping_delta (float): Минимальное улучшение для сброса счётчика.
        early_stopping_patience (int): Кол-во эпох без улучшения до остановки.

    Returns:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    counter = 0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Добавление регуляризаций
            l1_penalty = sum(
                torch.sum(torch.abs(param)) for param in model.parameters()
            )
            l2_penalty = sum(torch.sum(param**2) for param in model.parameters())
            loss += l1_lambda * l1_penalty + l2_lambda * l2_penalty

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # Early stopping
        if best_loss - avg_loss > early_stopping_delta:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= early_stopping_patience:
                logging.info("Early stopping triggered")
                break


# Функция обучения логистической регрессии
def train_logistic_model(model, dataloader, lr=0.01, epochs=100):
    """
    Обучает модель логистической регрессии для задачи многоклассовой классификации.

    Args:
        model (nn.Module): Модель логистической регрессии.
        dataloader (DataLoader): Загрузчик обучающего датасета.
        lr (float): Скорость обучения.
        epochs (int): Количество эпох обучения.

    Returns:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch}: loss={avg_loss:.4f}")


# Оценка логистической модели
def evaluate_logistic_model(model, dataloader, num_classes):
    """
    Печатает метрики классификации и строит confusion matrix.

    Args:
        model (nn.Module): Обученная модель.
        dataloader (DataLoader): Загрузчик тестового датасета.
        num_classes (int): Количество классов в задаче.

    Returns:
        None
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(batch_y.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("ROC-AUC cannot be computed (possible missing classes in batch).")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()


# Тест обучения моделей
def test_linear_model():
    """Запускает тест обучения линейной регрессии."""
    X, y = make_regression(n_samples=300, n_features=4, noise=0.2)
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LinearRegressionModel(in_features=4)
    train_linear_model(
        model, dataloader, lr=0.1, epochs=100, l1_lambda=1e-4, l2_lambda=1e-4
    )
    logging.info("Тест линейной регрессии пройден")


def test_logistic_model():
    """Запускает тест обучения логистической регрессии."""
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
    )
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LogisticRegressionModel(in_features=4, num_classes=3)
    train_logistic_model(model, dataloader, lr=0.1, epochs=100)
    evaluate_logistic_model(model, dataloader, num_classes=3)
    logging.info("Тест логистической регрессии пройден")


if __name__ == "__main__":
    test_linear_model()
    test_logistic_model()

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Кастомный CSV Dataset
class CSVDataset(Dataset):
    """
    Кастомный Dataset для загрузки CSV-файлов с предобработкой.
    Поддерживает как регрессию, так и классификацию.

    Args:
        filepath (str): Путь к CSV-файлу.
        target_column (str): Название целевого столбца.
        task_type (str): 'regression' или 'classification'.
        drop_columns (list, optional): Список столбцов для удаления.
        csv_sep (str): Разделитель в CSV-файле.
    """

    def __init__(
        self,
        filepath,
        target_column,
        task_type="regression",
        drop_columns=None,
        csv_sep=",",
    ):
        self.df = pd.read_csv(filepath, sep=csv_sep)
        self.target_column = target_column
        self.task_type = task_type

        if drop_columns:
            self.df.drop(columns=drop_columns, inplace=True)

        self.df.dropna(inplace=True)

        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]

        self.numerical_cols = self.X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        self.transformer = ColumnTransformer(
            [
                ("num", StandardScaler(), self.numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
            ]
        )

        self.X_processed = self.transformer.fit_transform(self.X)

        if self.task_type == "regression":
            self.y_processed = self.y.values.astype(np.float32).reshape(-1, 1)
        elif self.task_type == "classification":
            label_encoder = LabelEncoder()
            self.y_processed = label_encoder.fit_transform(self.y).astype(np.int64)
        else:
            raise ValueError("task_type должен быть 'regression' или 'classification'")

        self.X_tensor = torch.tensor(
            (
                self.X_processed.toarray()
                if hasattr(self.X_processed, "toarray")
                else self.X_processed
            ),
            dtype=torch.float32,
        )
        self.y_tensor = torch.tensor(
            self.y_processed,
            dtype=torch.float32 if self.task_type == "regression" else torch.long,
        )

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]


# Модели
class LinearRegressionModel(nn.Module):
    """
    Простая линейная регрессия на основе одного слоя Linear.

    Args:
        in_features (int): Количество входных признаков.
    """

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegressionModel(nn.Module):
    """
    Модель логистической регрессии для многоклассовой классификации.

    Args:
        in_features (int): Количество входных признаков.
        num_classes (int): Количество классов.
    """

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


# Тренировка и оценка
def train_model(model, dataloader, task_type="regression", lr=0.01, epochs=50):
    """
    Обучает модель с использованием указанного критерия и логирования.

    Args:
        model (nn.Module): Обучаемая модель.
        dataloader (DataLoader): Загрузчик данных.
        task_type (str): 'regression' или 'classification'.
        lr (float): Скорость обучения.
        epochs (int): Количество эпох.

    Returns:
        None
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            if task_type == "regression":
                loss = criterion(outputs, y_batch)
            else:
                loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")


def evaluate_model(model, dataloader, task_type="regression", num_classes=None):
    """
    Оценивает обученную модель по соответствующим метрикам.

    Args:
        model (nn.Module): Обученная модель.
        dataloader (DataLoader): Загрузчик данных.
        task_type (str): 'regression' или 'classification'.
        num_classes (int, optional): Число классов (для классификации).

    Returns:
        None
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            if task_type == "regression":
                y_true.extend(y_batch.squeeze().tolist())
                y_pred.extend(outputs.squeeze().tolist())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_true.extend(y_batch.tolist())
                y_pred.extend(preds.tolist())

    if task_type == "regression":
        print("MSE:", mean_squared_error(y_true, y_pred))
        print("R2:", r2_score(y_true, y_pred))
    else:
        print("Classification Report:")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("plots/confusion_matrix_dataset.png")
        plt.close()


# Запуск эксперимента
def run_experiment(file_path, target_column, task_type, drop_columns=None, csv_sep=","):
    """
    Запускает полный цикл эксперимента: загрузка, обучение, оценка.

    Args:
        file_path (str): Путь к CSV-файлу.
        target_column (str): Название целевого столбца.
        task_type (str): 'regression' или 'classification'.
        drop_columns (list, optional): Столбцы, которые нужно отбросить.
        csv_sep (str): Разделитель в файле.

    Returns:
        None
    """
    dataset = CSVDataset(file_path, target_column, task_type, drop_columns, csv_sep)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = dataset.X_tensor.shape[1]
    if task_type == "regression":
        model = LinearRegressionModel(input_size)
    else:
        num_classes = len(torch.unique(dataset.y_tensor))
        model = LogisticRegressionModel(input_size, num_classes)

    train_model(model, dataloader, task_type=task_type, lr=0.01, epochs=50)
    evaluate_model(model, dataloader, task_type=task_type)


# Основной блок
if __name__ == "__main__":
    # Классификация: Titanic
    run_experiment(
        file_path="data/titanic.csv",
        target_column="survived",
        task_type="classification",
        drop_columns=["name"],
    )

    # Регрессия: Wine Quality
    run_experiment(
        file_path="data/winequality-red.csv",
        target_column="quality",
        task_type="regression",
        csv_sep=";",
    )

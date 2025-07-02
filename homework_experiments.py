import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_wine_data(filepath, sep=";"):
    """
    Загружает данные из CSV-файла с вином.

    Args:
        filepath (str): Путь к CSV-файлу.
        sep (str): Разделитель в файле.

    Returns:
        DataFrame: Загруженный DataFrame.
    """
    df = pd.read_csv(filepath, sep=sep)
    return df


def preprocess_data(
    df, target_column="quality", poly_degree=None, interaction_only=False
):
    """
    Выполняет предобработку и создание новых признаков (опционально).

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        target_column (str): Название целевой переменной.
        poly_degree (int): Степень полиномиальных признаков.
        interaction_only (bool): Только взаимодействия между признаками.

    Returns:
        TensorDataset: Обработанные данные.
        int: Размер входных признаков.
        float: R2 score базовой модели.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column].values.astype(np.float32).reshape(-1, 1)

    # Стандартизация
    X_std = (X - X.mean()) / X.std()
    baseline_input_size = X_std.shape[1]

    # Базовая модель
    X_tensor = torch.tensor(X_std.values, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    baseline_model = nn.Linear(baseline_input_size, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(baseline_model.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        output = baseline_model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    baseline_preds = baseline_model(X_test).detach().numpy()
    baseline_r2 = r2_score(y_test.numpy(), baseline_preds)

    # Feature engineering
    if poly_degree:
        poly = PolynomialFeatures(
            degree=poly_degree, interaction_only=interaction_only, include_bias=False
        )
        X_poly = poly.fit_transform(X_std)
        X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
        X_poly_df["row_mean"] = X_poly_df.mean(axis=1)
        X_poly_df["row_std"] = X_poly_df.std(axis=1)
        X_tensor = torch.tensor(X_poly_df.values, dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X_std.values, dtype=torch.float32)

    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset, X_tensor.shape[1], baseline_r2


def train_and_evaluate(model, dataloader):
    """
    Обучает модель и вычисляет R^2-оценку.

    Args:
        model (nn.Module): Модель PyTorch.
        dataloader (DataLoader): Загрузчик данных.

    Returns:
        float: R^2 score.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(100):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            predictions.extend(outputs.squeeze().tolist())
            targets.extend(y_batch.squeeze().tolist())

    return r2_score(targets, predictions)


def run_experiment(filepath):
    """
    Выполняет серию экспериментов по обучению модели с разными гиперпараметрами.

    Args:
        filepath (str): Путь к датасету.

    Returns:
        None
    """
    df = load_wine_data(filepath)
    results = []

    learning_rates = [0.01, 0.001]
    batch_sizes = [16, 32]
    optimizers = {"SGD": optim.SGD, "Adam": optim.Adam, "RMSprop": optim.RMSprop}

    for lr in learning_rates:
        for bs in batch_sizes:
            for opt_name, opt_func in optimizers.items():
                dataset, input_size, _ = preprocess_data(df)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
                model = nn.Linear(input_size, 1)
                optimizer = opt_func(model.parameters(), lr=lr)
                criterion = nn.MSELoss()
                for _ in range(50):
                    for X_batch, y_batch in dataloader:
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                r2 = train_and_evaluate(model, dataloader)
                results.append((lr, bs, opt_name, r2))
                logging.info(f"lr={lr}, batch_size={bs}, opt={opt_name}, R2={r2:.4f}")

    # Визуализация результатов
    plot_hyperparam_results(results)


def plot_hyperparam_results(results):
    """
    Строит bar-график для сравнения гиперпараметров.

    Args:
        results (List[Tuple[float, int, str, float]]): Список результатов.

    Returns:
        None
    """
    labels = [f"lr={lr}\nbs={bs}\n{opt}" for lr, bs, opt, _ in results]
    scores = [r2 for _, _, _, r2 in results]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, scores, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R2 Score")
    plt.title("Сравнение гиперпараметров")
    plt.tight_layout()
    plt.savefig("plots/hyperparam_comparison.png")
    plt.close()


def plot_feature_engineering_results(results):
    """
    Строит график для сравнения качества моделей с различными признаками.

    Args:
        results (List[Tuple[str, float]]): Название подхода и R2-оценка.

    Returns:
        None
    """
    names = [name for name, _ in results]
    scores = [score for _, score in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, scores, color="lightgreen")
    plt.ylabel("R2 Score")
    plt.title("Влияние Feature Engineering на качество модели")
    plt.tight_layout()
    plt.savefig("plots/feature_engineering_comparison.png")
    plt.close()


if __name__ == "__main__":
    df = load_wine_data("data/winequality-red.csv")
    baseline_dataset, input_size, baseline_r2 = preprocess_data(df)
    dataloader = DataLoader(baseline_dataset, batch_size=32, shuffle=True)
    baseline_model = nn.Linear(input_size, 1)
    baseline_r2 = train_and_evaluate(baseline_model, dataloader)

    poly_dataset, _, poly_r2 = preprocess_data(
        df, poly_degree=2, interaction_only=False
    )
    poly_loader = DataLoader(poly_dataset, batch_size=32, shuffle=True)
    poly_model = nn.Linear(poly_dataset.tensors[0].shape[1], 1)
    poly_r2 = train_and_evaluate(poly_model, poly_loader)

    inter_dataset, _, inter_r2 = preprocess_data(
        df, poly_degree=2, interaction_only=True
    )
    inter_loader = DataLoader(inter_dataset, batch_size=32, shuffle=True)
    inter_model = nn.Linear(inter_dataset.tensors[0].shape[1], 1)
    inter_r2 = train_and_evaluate(inter_model, inter_loader)

    results = [
        ("Baseline", baseline_r2),
        ("Poly Features", poly_r2),
        ("Interactions Only", inter_r2),
    ]

    plot_feature_engineering_results(results)

    run_experiment("data/winequality-red.csv")

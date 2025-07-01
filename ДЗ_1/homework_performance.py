# Задание 3: Сравнение производительности CPU vs CUDA
import torch
import time

torch.manual_seed(42868594)

# 3.1 Подготовка данных
# Матрица размером 64 x 1024 x 1024 со случайными числами:
random_matrix_64x1024x1024 = torch.rand(64, 1024, 1024)
# Матрица размером 128 x 512 x 512 со случайными числами:
random_matrix_128x512x512 = torch.rand(128, 512, 512)
# Матрица размером 256 x 256 x 256 со случайными числами:
random_matrix_256x256x256 = torch.rand(256, 256, 256)


# 3.2 Функция измерения времени
# Функции для измерения времени выполнения операций:
def measure_cpu_time(func, *args):
    """
    Измеряет время выполнения операции на CPU.

    Args:
        func (callable): Функция, которую нужно выполнить.
        *args: Аргументы, передаваемые в функцию.

    Returns:
        float: Время выполнения операции в миллисекундах.

    Raises:
        TypeError: Если аргумент не является тензором.
        ValueError: Если размерность тензора некорректна.
    """
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            raise TypeError("Аргументы должны быть объектами torch.Tensor.")
        if arg.dim() != 3:
            raise ValueError(f"Ожидалась размерность 3, получено: {arg.dim()}")

    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000


def measure_cuda_time(func, *args):
    """
    Измеряет время выполнения операции на GPU (CUDA).

    Args:
        func (callable): Функция, которую нужно выполнить.
        *args: Аргументы, передаваемые в функцию.

    Returns:
        float: Время выполнения операции в миллисекундах.

    Raises:
        TypeError: Если аргумент не является тензором.
        ValueError: Если размерность тензора некорректна.
    """
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            raise TypeError("Аргументы должны быть объектами torch.Tensor.")
        if arg.dim() != 3:
            raise ValueError(f"Ожидалась размерность 3, получено: {arg.dim()}")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    func(*args)
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


# 3.3 Сравнение операций
# Сравнение времекни выполнения операций на CPU и CUDA:
operations = {
    "Матричное умножение": lambda x: torch.matmul(x, x.transpose(-1, -2)),
    "Сложение": lambda x: x + x,
    "Поэлементное умножение": lambda x: x * x,
    "Транспонирование": lambda x: x.transpose(-1, -2),
    "Суммирование": lambda x: x.sum(),
}

matrix_info = [
    ("64x1024x1024", random_matrix_64x1024x1024),
    ("128x512x512", random_matrix_128x512x512),
    ("256x256x256", random_matrix_256x256x256),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nУстройство: {device}\n")

for label, matrix in matrix_info:
    print(f"\nРазмер тензора: {label}")
    print(f"{'Операция':<22} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение'}")
    print("-" * 60)
    for name, op in operations.items():
        cpu_time = measure_cpu_time(op, matrix)

        if torch.cuda.is_available():
            matrix_cuda = matrix.to(device)
            gpu_time = measure_cuda_time(op, matrix_cuda)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            print(
                f"{name:<22} | {cpu_time:>9.2f} | {gpu_time:>9.2f} | {speedup:>8.2f}x"
            )
        else:
            print(f"{name:<22} | {cpu_time:>9.2f} | {'—':>9} | {'—'}")

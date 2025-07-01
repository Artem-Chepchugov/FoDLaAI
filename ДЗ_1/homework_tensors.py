# Задание 1: Создание и манипуляции с тензорами
import torch

torch.manual_seed(42868594)

# 1.1 Создание тензоров
# Тензор размером 3x4, заполненный случайными числами от 0 до 1:
random_tensor_3x4 = torch.rand(3, 4)
print(
    f"Тензор размером 3x4, заполненный случайными числами от 0 до 1: \n{random_tensor_3x4}\n"
)

# Тензор размером 2x3x4, заполненный нулями:
zero_tensor_2x3x4 = torch.zeros(2, 3, 4)
print(f"Тензор размером 2x3x4, заполненный нулями: \n{zero_tensor_2x3x4}\n")

# Тензор размером 5x5, заполненный единицами:
ones_tensor_5x5 = torch.ones(5, 5)
print(f"Тензор размером 5x5, заполненный единицами: \n{ones_tensor_5x5}\n")

# Тензор размером 4x4 с числами от 0 до 15 с использованием reshape:
classic_tensor = torch.arange(16)
print(f"Тензор 1x16 без преобразования при помощи reshape: \n{classic_tensor}\n")

if classic_tensor.numel() == 16:
    reshaped_tensor_4x4 = torch.reshape(classic_tensor, (4, 4))
    print(
        f"Тензор размером 4x4 с числами от 0 до 15 с использованием reshape: \n{reshaped_tensor_4x4}\n"
    )
else:
    print("Ошибка: неверное количество элементов для reshape в тензор размером 4x4.")

# 1.2 Операции с тензорами
A = torch.arange(12).reshape(3, 4)
B = torch.arange(12).reshape(4, 3)
print(f"Исходная матрица A: \n{A}\n")
print(f"Исходная матрица B: \n{B}\n")

# Транспонирование тензора A:
transposed_A = torch.transpose(A, 0, 1)
print(f"Транспонированная матрица A: \n{transposed_A}\n")

# Матричное умножение A и B:
matrix_product_AB = torch.matmul(A, B)
print(f"Результат матричного умножения A и B: \n{matrix_product_AB}\n")

# Поэлементное умножение A и транспонированного B:
transposed_B = torch.transpose(B, 0, 1)
A_elementwise_mult_BT = A * transposed_B
print(
    f"Результат поэлементного умножения A и транспонированного B: \n{A_elementwise_mult_BT}\n"
)

# Сумма всех элементов тензора A:
sum_of_elements_A = torch.sum(A)
print(f"Сумма всех элементов тензора A: \n{sum_of_elements_A}\n")

# 1.3 Индексация и срезы
tensor_5x5x5 = torch.arange(125).reshape(5, 5, 5)
print(f"Исходный тензор размером 5x5x5: \n{tensor_5x5x5}\n")

# Извлечение первой строки:
first_row_of_tensor = tensor_5x5x5[:1, :1, :]
print(f"Первая строка: \n{first_row_of_tensor}\n")

# Извлечение последнего столбца:
last_column_of_tensor = tensor_5x5x5[-1:, :, -1:]
print(f"Последний столбец: \n{last_column_of_tensor}\n")

# Извлечение подматрицы размером 2x2 из центра тензора:
central_submatrix_2x2 = tensor_5x5x5[2, 1:3, 1:3]
print(f"Подматрица размером 2x2 из центра тензора: \n{central_submatrix_2x2}\n")

# Извлечение всех элементов с четными индексами:
even_indexed_elements = tensor_5x5x5[:, :, ::2]
print(f"Все элементы с четными индексами: \n{even_indexed_elements}\n")

# 1.4 Работа с формами
tensor_24_elements = torch.arange(24)
print(f"Тензор 1x24 без преобразования при помощи reshape: \n{tensor_24_elements}\n")


def safe_reshape(tensor, shape):
    """
    Безопасно преобразует тензор в заданную форму.

    Проверяет, совпадает ли общее количество элементов в тензоре с произведением размеров новой формы.
    Если совпадает, возвращает reshaped тензор.
    Если нет — выводит сообщение об ошибке и возвращает None.

    Args:
        tensor (torch.Tensor): Исходный тензор.
        shape (tuple или list): Желаемая форма для преобразования.

    Returns:
        torch.Tensor или None: Преобразованный тензор или None при ошибке.
    """
    if tensor.numel() == torch.tensor(shape).prod().item():
        return torch.reshape(tensor, shape)
    else:
        print(
            f"Ошибка: нельзя преобразовать тензор размером {tensor.numel()} элементов в форму {shape}."
        )
        return None


reshaped_to_2x12 = safe_reshape(tensor_24_elements, (2, 12))
if reshaped_to_2x12 is not None:
    print(
        f"Преобразованный тензор размером 24 элемента в форму 2x12: \n{reshaped_to_2x12}\n"
    )

reshaped_to_3x8 = safe_reshape(tensor_24_elements, (3, 8))
if reshaped_to_3x8 is not None:
    print(
        f"Преобразованный тензор размером 24 элемента в форму 3x8: \n{reshaped_to_3x8}\n"
    )

reshaped_to_4x6 = safe_reshape(tensor_24_elements, (4, 6))
if reshaped_to_4x6 is not None:
    print(
        f"Преобразованный тензор размером 24 элемента в форму 4x6: \n{reshaped_to_4x6}\n"
    )

reshaped_to_2x3x4 = safe_reshape(tensor_24_elements, (2, 3, 4))
if reshaped_to_2x3x4 is not None:
    print(
        f"Преобразованный тензор размером 24 элемента в форму 2x3x4: \n{reshaped_to_2x3x4}\n"
    )

reshaped_to_2x2x2x3 = safe_reshape(tensor_24_elements, (2, 2, 2, 3))
if reshaped_to_2x2x2x3 is not None:
    print(
        f"Преобразованный тензор размером 24 элемента в форму 2x2x2x3: \n{reshaped_to_2x2x2x3}\n"
    )

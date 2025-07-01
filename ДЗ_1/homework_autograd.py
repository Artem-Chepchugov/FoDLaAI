# Задание 2: Автоматическое дифференцирование
import torch

torch.manual_seed(42868594)

# 2.1 Простые вычисления с градиентами
# Тензоры x, y, z с requires_grad=True:
x1 = torch.randn(2, 2, requires_grad=True)
y1 = torch.randn(2, 2, requires_grad=True)
z1 = torch.randn(2, 2, requires_grad=True)

# Проверка, что все тензоры одинаковой формы для операции
assert x1.shape == y1.shape == z1.shape, "Тензоры x, y, z должны иметь одинаковую форму"

print(f"Тензор x с requires_grad=True: \n{x1}\n")
print(f"Тензор y с requires_grad=True: \n{y1}\n")
print(f"Тензор z с requires_grad=True: \n{z1}\n")

# Вычисление функции f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z:
f1 = x1**2 + y1**2 + z1**2 + 2 * x1 * y1 * z1
print(f"Функция f(x,y,z) = \n{f1}\n")
f1.sum().backward()

# Градиенты по всем переменным:
print(f"Градиент x: \n{x1.grad}\n")
print(f"Градиент y: \n{y1.grad}\n")
print(f"Градиент z: \n{z1.grad}\n")

# 2.2 Градиент функции потерь
# Функция MSE = (1/n) * Σ(y_pred - y_true)^2, где y_pred = w * x + b (лин. ф-ция):
x2 = torch.randn(2, 2)
y2_true = torch.randn(2, 2)

# Проверка размерностей входных данных
assert x2.shape == y2_true.shape, "x и y_true должны иметь одинаковую форму"

w2 = torch.randn(1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

y2_pred = w2 * x2 + b2
mse = ((y2_pred - y2_true) ** 2).mean()
mse.backward()

# Градиенты по w и b:
print(f"Градиент по w: \n{w2.grad}\n")
print(f"Градиент по b: \n{b2.grad}\n")

# 2.3 Цепное правило
# Составная функция f(x) = sin(x^2 + 1):
x3 = torch.randn(2, 2, requires_grad=True)
f3 = torch.sin(x3**2 + 1)
f3.sum().backward(retain_graph=True)

# Градиент df/dx:
print(f"Градиент df/dx: \n{x3.grad}\n")

# Проверка через torch.autograd.grad:
grad3 = torch.autograd.grad(f3.sum(), x3)[0]
print(f"Проверка через torch.autograd.grad: \n{grad3}\n")

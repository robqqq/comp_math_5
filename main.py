import string
import sys
import time
import sympy
import numpy as np
import matplotlib.pyplot as plt
import csv
from sympy.core.sympify import SympifyError

X_SYMPY = sympy.symbols("x")
DELTA = 1E-6


def input_choice():
    input_choice_prompt = "\nВыберите способ ввода или введите 0 для выхода:\n" \
                          "\tНабор точек (1)\n" \
                          "\tНабор точек из файла (2)\n" \
                          "\tФункция (3)\n" \
                          "\tВыход (0)\n" \
                          "> "
    try:
        choice = input(input_choice_prompt)
    except EOFError:
        return 0
    return choice.strip()


def input_values():
    print()
    print("Введите пары значений (x, y), разделенные пробелом, для завершения ввода, введите 'stop':")
    values = []
    count = 0
    while True:
        try:
            s = input("> ")
            if s == "stop":
                if count < 2:
                    print("Количество точек должно быть >= 2, продолжайте ввод", file=sys.stderr)
                    time.sleep(0.1)
                    continue
                else:
                    break
            x, y = map(float, s.split(" "))
            values.append((x, y))
            count += 1
        except ValueError:
            print("Некорректная точка, повторите ввод:", file=sys.stderr)
            time.sleep(0.1)
    return values


def input_file():
    path = input("Введите путь к файлу: ")
    values = []
    try:
        with open(path) as file:
            reader = csv.DictReader(file)
            for line in reader:
                values.append((float(line["x"]), float(line["y"])))
    except KeyError:
        print("Файл должен быть в формате CSV, с хедером 'x,y'", file=sys.stderr)
        time.sleep(0.1)
        return None
    except ValueError:
        print("Значения X и Y должны быть числами", file=sys.stderr)
        time.sleep(0.1)
        return None
    if len(values) < 2:
        print("Количество точек должно быть >= 2", file=sys.stderr)
        time.sleep(0.1)
        return None
    return values


def input_function():
    print("Введите функцию:")
    while True:
        try:
            func = input("> ")
            float(sympy.sympify(func).evalf(subs={X_SYMPY: 1}))
            break
        except (SympifyError, TypeError):
            print("Некорректная функция, повторите ввод:", file=sys.stderr)
            time.sleep(0.1)
    print("Введите количество узлов:")
    n = 0
    while True:
        try:
            n = int(input("> "))
            if n < 2:
                print("Количество точек должно быть >= 2, повторите ввод")
                time.sleep(0.1)
            else:
                break
        except ValueError:
            print("Некорректное значение, повторите ввод:", file=sys.stderr)
            time.sleep(0.1)
    print("Введите границы отрезка интерполяции")
    while True:
        try:
            a, b = map(float, input("> ").split(" "))
            break
        except ValueError:
            print("Некорректная точка, повторите ввод:", file=sys.stderr)
            time.sleep(0.1)
    if a > b:
        a, b = b, a
    values = []
    h = (b - a) / (n - 1)
    for i in range(n):
        values.append((a, sympy.sympify(func).evalf(subs={X_SYMPY: a})))
        a += h
    return values, func


def input_x():
    print("Введите значение x для расчета значения:")
    while True:
        try:
            x = float(input("> "))
            break
        except ValueError:
            print("Некорректное значение, повторите ввод:", file=sys.stderr)
            time.sleep(0.1)
    return x


def interpolate_lagrange(x, values):
    result = 0
    for i in range(len(values)):
        upper = 1
        lower = 1
        for j in range(len(values)):
            if i != j:
                upper *= x - values[j][0]
                lower *= values[i][0] - values[j][0]
        result += values[i][1] * upper / lower
    return result


def interpolate_gaussian(x, values):
    h = values[1][0] - values[0][0]
    for i in range(1, len(values) - 1):
        t = values[i + 1][0] - values[i][0]
        if np.abs(h - t) > DELTA:
            return None
    n = len(values)
    if n % 2 == 0:
        n -= 1
    middle = n // 2
    x0 = values[middle][0]

    diff_matrix = [[0 for __ in range(len(values))] for _ in range(len(values))]
    for i in range(len(values)):
        diff_matrix[i][0] = values[i][1]
    rows = len(values) - 1
    for j in range(1, len(values)):
        for i in range(rows):
            diff_matrix[i][j] = diff_matrix[i + 1][j - 1] - diff_matrix[i][j - 1]
        rows -= 1

    t = (x - x0) / h
    sum = 0
    if x > x0:
        for i in range(n):
            i1 = i // 2
            s = 1
            m = 1
            if i % 2 == 0:
                for j in range(-i1 + 1, i1 + 1):
                    s *= (t - j) / m
                    m += 1
            else:
                for j in range(-i1, i1 + 1):
                    s *= (t - j) / m
                    m += 1

            s *= diff_matrix[middle - i1][i]
            sum += s
    else:
        for i in range(n):
            i1 = i // 2
            s = 1
            m = 1
            if i % 2 == 0:
                for j in range(-i1 + 1, i1 + 1):
                    s *= (t + j) / m
                    m += 1
            else:
                for j in range(-i1, i1 + 1):
                    s *= (t + j) / m
                    m += 1

            s *= diff_matrix[middle - i1 - i % 2][i]
            sum += s

    return sum


def main():
    while True:
        func = None
        match input_choice():
            case "0":
                break
            case "1":
                values = input_values()
            case "2":
                values = input_file()
            case "3":
                values, func = input_function()
            case _:
                print("Необходимо ввести 0, 1 или 2", file=sys.stderr)
                time.sleep(0.1)
                continue
        if values is None:
            continue
        x = input_x()
        x_values = np.array([value[0] for value in values])
        y_values = np.array([value[1] for value in values])
        y_lagrange = interpolate_lagrange(x, values)
        print("Интерполяция по Лагранжу -", y_lagrange)
        y_gaussian = interpolate_gaussian(x, values)
        plot_x = np.linspace(np.min(x_values), np.max(x_values))
        plot_lagrange = [interpolate_lagrange(x, values) for x in plot_x]
        plt.figure()
        plt.plot(plot_x, plot_lagrange, label="Многочлен Лагранжа", color="g")
        if func is not None:
            plot_func = [sympy.sympify(func).evalf(subs={X_SYMPY: x}) for x in plot_x]
            plt.plot(plot_x, plot_func, label="Заданная функция", color="b")
        if y_gaussian is not None:
            plot_gaussian = [interpolate_gaussian(x, values) for x in plot_x]
            plt.plot(plot_x, plot_gaussian, label="Многочлен Гаусса", color="r")
            plt.scatter(x, y_gaussian, label="Интерполяция по Гауссу", color="r")
            print("Интерполяция по Гауссу -", y_gaussian)
        else:
            print("Невозможно интерполировать по Гауссу, так как шаг между узлами различен")
        plt.scatter(x_values, y_values, label="Узлы интерполяции", color="b")
        plt.scatter(x, y_lagrange, label="Интерполяция по Лагранжу", color="g")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()

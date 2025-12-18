import toml
import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
import json
import sys

class SphereRCS:
    """
    Класс для расчета Эффективной Площади Рассеяния (ЭПР)
    идеально проводящей сферы.
    """
    def __init__(self, D):
        """
        :param D: Диаметр сферы (м)
        """
        self.radius = D / 2.0  # радиус сферы
        self.wavelength = None
        self.freq = None

    def compute_rcs(self, freq):
        """
        Расчет ЭПР для заданной частоты.
        :param freq: частота в Гц
        :return: ЭПР в м^2
        """
        c = 299792458.0  # скорость света
        wavelength = c / freq
        k = 2 * np.pi / wavelength  # волновое число
        kr = k * self.radius

        # Максимальный порядок ряда (достаточно для сходимости)
        N = int(np.ceil(kr + 10))

        # Инициализация суммы
        sigma = 0.0
        for n in range(1, N + 1):
            # Сферические функции Бесселя
            jn_kr = spherical_jn(n, kr)
            jn_kr_prime = spherical_jn(n, kr, derivative=True)
            yn_kr = spherical_yn(n, kr)
            yn_kr_prime = spherical_yn(n, kr, derivative=True)

            # Вычисление коэффициентов an, bn
            # Для идеально проводящей сферы:
            an = - (jn_kr) / (jn_kr + 1j * yn_kr)
            bn = - (kr * jn_kr_prime) / (kr * jn_kr_prime + 1j * kr * yn_kr_prime)

            term = ((-1) ** n) * (n + 0.5) * (bn - an)
            sigma += term

        # Формула ЭПР
        rcs = (wavelength ** 2 / np.pi) * np.abs(sigma) ** 2
        return rcs


class ResultWriter:
    """
    Класс для записи результатов в файлы разных форматов.
    """
    def __init__(self, freqs, wavelengths, rcs_values):
        self.freqs = freqs
        self.wavelengths = wavelengths
        self.rcs_values = rcs_values

    def write_json_format_3(self, filename):
        """
        Запись в JSON формата 3 (массив объектов).
        """
        data = []
        for f, lam, rcs in zip(self.freqs, self.wavelengths, self.rcs_values):
            data.append({
                "freq": float(f),
                "lambda": float(lam),
                "rcs": float(rcs)
            })
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"data": data}, f, indent=4)


# =================== Основная программа ===================

def main():
    # Параметры варианта 12
    input_file = "task_rcs_02.toml"
    variant_index = 12  # вариант 12
    output_format = 3   # формат 3 (JSON массив словарей)

    # Чтение входных данных
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = toml.load(f)
    except FileNotFoundError:
        print(f"Файл {input_file} не найден.")
        sys.exit(1)

    variant_key = f"variant_{variant_index}"
    if variant_key not in data['data']:
        print(f"Вариант {variant_index} не найден в файле.")
        sys.exit(1)

    params = data['data'][variant_key]
    D = float(params['D'])
    fmin = float(params['fmin'])
    fmax = float(params['fmax'])

    print(f"Вариант {variant_index}:")
    print(f"  Диаметр сферы D = {D} м")
    print(f"  Диапазон частот: от {fmin} Гц до {fmax} Гц")

    # Создание объекта для расчета ЭПР
    sphere = SphereRCS(D)

    # Генерация частот (линейная шкала)
    num_points = 500
    freqs = np.linspace(fmin, fmax, num_points)
    wavelengths = 299792458.0 / freqs

    # Расчет ЭПР для каждой частоты
    rcs_values = []
    for f in freqs:
        rcs = sphere.compute_rcs(f)
        rcs_values.append(rcs)

    # Запись результатов
    writer = ResultWriter(freqs, wavelengths, rcs_values)
    output_file = f"rcs_result_variant_{variant_index}.json"
    writer.write_json_format_3(output_file)
    print(f"Результаты сохранены в {output_file}")

    # Построение графика (линейный масштаб)
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, rcs_values, 'b-', linewidth=2)
    plt.xlabel('Частота (Гц)', fontsize=14)
    plt.ylabel('ЭПР (м²)', fontsize=14)
    plt.title(f'ЭПР идеально проводящей сферы (D={D} м)', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"rcs_plot_variant_{variant_index}.png", dpi=300)
    plt.show()

    # Вывод первых 20 строк результата
    print("\nПервые 20 строк результата:")
    print("Частота (Гц)\t\tДлина волны (м)\t\tЭПР (м²)")
    print("-" * 70)
    for i in range(min(20, len(freqs))):
        print(f"{freqs[i]:.6e}\t{wavelengths[i]:.6e}\t{rcs_values[i]:.6e}")


if __name__ == "__main__":
    main()
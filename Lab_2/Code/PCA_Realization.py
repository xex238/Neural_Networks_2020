# Рисование красивой матрицы кореляций
import seaborn as sn

# matplotlib libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# sklearn libraries
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA # Алгоритм (метод главных компонент)

# additional libraries
import numpy as np # Работа с матрицами
import pandas as pd # Работа с датафреймами + матрица корреляций
import csv
import math

class PCA_Realization:
    data = None # Входные данные
    dataset_name = None # Имя датасета

    # 1) Надо сохранить
    corr = None # Матрица корреляции
    corr_sum = None # Построчная сумма значений по модулю матрицы корреляции
    # 2) Надо сохранить
    low_corr_list = [] # Список признаков со слабой корреляцией

    data_scaled = None # Отмасштабированные (стандартизованные) данные

    pca = None # Реализация метода главных компонент

    variance = None # Значения дисперсии главных компонент
    cov = None # Значения ковариации главных компонент
    values = None # Собственные значения главных компонент
    vectors = None # Собственные вектора главных компонент
    singular_values = None # Сингулярные значения (особые значения, соответствующие каждому из выбранных компонентов)

    sum = 0 # Сумма нагрузок главных компонент (сумма дисперсий)

    components_Kaizer = None # Число выбранных главных компонент по критерию Кайзера
    components_broken_stick = None # Число выбранных главных компонент по критерию сломанной стрости
    components_scree = None # Число выбранных главных компонент по критерию каменистой осыпи

    PCA_matrix = None # Матрица уменьшенной размерности, полученная после применения метода главных компонент
    factor_load_matrix = None # Матрица факторной нагрузки

    sum_load_matrix = None # Ранжированный список с суммарной нагрузкой

    remains_list = None # Список остатков
    remains_mean = None # Средний остаток по списку

    # 3) Надо сохранить
    result_components = None # Результирующее число выбранных главных компонент

    # Стандартный конструктор класса принимает входные данные
    def __init__(self, data, dataset_name=""):
        self.data = data
        self.dataset_name = dataset_name

    # 2) Построение матрицы корреляций и составление списка признаков со слабой корреляцией
    def Correlation(self):
        # Получение матрицы корреляций
        self.corr = self.data.corr()
        # corr_.style.background_gradient(cmap = 'coolwarm') # Для вывода "цветных" значений (не для консоли)

        # Получение списка столбцов со слабой корреляцией
        for i in range(self.corr.shape[0]):
            for j in range(i):
                if(abs(self.corr.iloc[i][j]) < 1e-2):
                    self.low_corr_list.append([self.corr.columns[i], self.corr.index[j]])

        print("Список столбцов со слабой корреляцией (от меня):")
        for i in self.low_corr_list:
            print(i)
        print()

        low_corr_file = open("Results\\" + self.dataset_name + "_low_corr_list.txt", 'w')
        low_corr_file.write("Список столбцов со слабой корреляцией:\n")
        for i in self.low_corr_list:
            low_corr_file.write(str(i) + '\n')
        low_corr_file.close()

        # Отрисовка матрицы корреляции
        f1 = plt.figure("Correlation matrix")
        sn.heatmap(self.corr, annot = True)

        # Сохранение в файл матрицы корреляции
        plt.savefig("Results\\" + self.dataset_name + "_Correlation_matrix.jpeg")

    # 3) Подготовка данных для метода главных компонент на основе составленного списка
    def Data_preparing(self):
        # Масштабирование (стандартизация). Предобработка данных, после которой каждый признак имеет среднее 0 и дисперсию 1
        self.data_scaled = preprocessing.scale(self.data)

    # 4) Реализация метода главных компонент
    def PCA_Realization(self):
        # Создание и "обучение" метода главных компонент (с количеством главных компонент равных количеству столбцов)
        self.pca = PCA(n_components = self.data_scaled.shape[1])
        self.pca.fit(self.data_scaled)

        # Получаем значения описываемой дисперсии
        self.variance = self.pca.explained_variance_ratio_
        # Получаем значения ковариации
        self.cov = self.pca.get_covariance()
        # Получаем собственные значения
        self.values = self.pca.explained_variance_
        # Получаем собственные вектора
        self.vectors = self.pca.components_

        # Получение суммы нагрузок (суммы дисперсий)
        print("Коэффициенты дисперсии различных компонент:")
        for i in range(self.data_scaled.shape[1]):
            self.sum = self.sum + self.variance[i]
            print("Компонента ", i + 1, ": ", round(self.variance[i], 4))

        print()
        print("Сумма коэффициентов дисперсий = ", round(self.sum, 4))
        print()

    # Реализация критерия Кайзера (выбор только тех факторов, у которых собственные значения больше 1)
    def Kaizer_critery(self):
        #f1 = plt.figure("Критерий Кайзера")
        #plt.plot(range(0, self.data_scaled.shape[1]), np.sort(self.pca.explained_variance_)[::-1][:])

        values1_plus = None
        values1_minus = None
        for i in range(len(self.values)):
            if(self.values[i] < 1):
                values1_minus = i - 1
                values1_plus = i
                break

        #plt.title("Критерий Кайзера", size = 15)
        #plt.axhline(1, ls = '--', color = 'grey')
        #plt.axvline(values1_minus, ls = '--', color = 'grey')
        #plt.axvline(values1_plus, ls = '--', color = 'grey')

        #plt.savefig("Results\\" + self.dataset_name + "_Caizer_critery.jpeg")

        self.components_Kaizer = values1_plus
        print("Исходя из критерия Кайзера, необходимо оставить ", values1_plus, " главных компонент")
        print()

    # Реализация критерия сломанной трости
    def Broken_stick_critery(self):
        broken_stick = []
        for i in range(len(self.variance)):
            sum = 0
            for j in range(len(self.variance) - i):
                sum = sum + (1 / (j + 1))
            broken_stick.append(sum / len(self.variance))

        print("Критерий сломанной трости - Отклонение")
        for i in range(len(broken_stick)):
            print(round(broken_stick[i], 4), " - ", round(self.variance[i], 4))
        print()

        for i in range(len(broken_stick)):
            if(broken_stick[i] > self.variance[i]):
                print("Исходя из критерия сломанной трости, необходимо оставить ", i, " главных компонент")
                print()
                self.components_broken_stick = i
                break

    # Реализация критерия каменистой осыпи (поиск места, где убывание собственных значений слева направо максимально замедляется)
    def Scree_critery(self):
        # Правило каменистой осыпи
        f2 = plt.figure("Критерий каменистой осыпи")
        plt.plot([i + 1 for i in range(len(self.variance))], self.variance)
        plt.scatter([i + 1 for i in range(len(self.variance))], self.variance, marker="o")
        plt.xlabel("Principal component number")
        plt.ylabel("Variation")

        #plt.savefig("Results\\" + self.dataset_name + "_Scree_critery.jpeg")        

        plt.show()

        self.components_scree = input("Введите количество главных компонент, которые необходимо оставить согласно критерию каменистой осыпи: ")
        print()

    # 5) Реализация критериев выбора числа главных компонент (Кайзера, сломанной трости, каменистой осыпи)
    def Criteria_Realization(self):
        # Реализация критерия Кайзера (выбор только тех факторов, у которых собственные значения больше 1)
        self.Kaizer_critery()

        # Реализация критерия сломанной трости
        self.Broken_stick_critery()

        # Реализация критерия каменистой осыпи
        self.Scree_critery()

        f1 = plt.figure("Критерии выбора главных компонент")
        plt.plot([i + 1 for i in range(len(self.variance))], self.variance)
        plt.scatter([i + 1 for i in range(len(self.variance))], self.variance, marker="o")
        plt.title("Критерии выбора главных компонент", size = 18)

        plt.axvline(self.components_broken_stick, ls = '--', color = 'red')
        plt.axvline(self.components_Kaizer, ls = '-.', color = 'blue')
        plt.axvline(int(self.components_scree), ls = ':', color = 'green')

        plt.savefig("Results\\" + self.dataset_name + "_Criteries.jpeg") 

        plt.show() 

    # 7) Построение матрицы факторной нагрузки для всех значимых главных компонент
    def Get_factor_load_matrix(self):
        self.pca = PCA(n_components = self.result_components)
        self.pca.fit(self.data_scaled)

        # Получаем значения описываемой дисперсии
        self.variance = self.pca.explained_variance_ratio_
        # Получаем значения ковариации
        self.cov = self.pca.get_covariance()
        # Получаем собственные значения
        self.values = self.pca.explained_variance_
        # Получаем собственные вектора
        self.vectors = self.pca.components_

        # Получение результирующей матрицы после применения метода главных компонент
        self.PCA_matrix = self.pca.transform(self.data_scaled) # Тип матрицы - numpy array
        #print("PCA_matrix = \n", self.PCA_matrix)

        columns = []
        for i in range(self.result_components):
            columns.append("comp_" + str(i))
        #print("columns = \n", columns)

        # Получение матрицы факторной нагрузки
        general_matrix = self.data.join(pd.DataFrame(data = self.PCA_matrix, columns = columns))
        #print("general matrix = \n", general_matrix)

        general_matrix_corr = general_matrix.corr()
        #print("factor load matrix = \n", general_matrix_corr)

        self.factor_load_matrix = general_matrix_corr.iloc[:self.corr.shape[1], self.corr.shape[1]:]
        print("factor load matrix = \n", self.factor_load_matrix)
        print()

        writer = pd.ExcelWriter("Results\\" + self.dataset_name + "_Factor_Load_Matrix.xlsx", engine = 'xlsxwriter')
        self.factor_load_matrix.to_excel(writer, "Factor_load_matrix")
        writer.save()
        writer.close()

    # 8) Определение суммарной нагрузки для признаков. Составление ранжированного списка признаков (по убыванию нагрузки)
    def Sum_load(self):
        sum_load_matrix = []
        for i in range(self.factor_load_matrix.shape[0]):
            sum = 0
            for j in range(self.factor_load_matrix.shape[1]):
                sum = sum + self.factor_load_matrix.iloc[i][j]
            sum_load_matrix.append(sum)

        self.sum_load_matrix = pd.Series(sum_load_matrix, index = self.factor_load_matrix.index)

        self.sum_load_matrix = self.sum_load_matrix.sort_values(ascending=False)

        print("Отсортированный список признаков с суммарной нагрузкой:\n", self.sum_load_matrix)
        print()

        writer = pd.ExcelWriter("Results\\" + self.dataset_name + "_sort_sum_load_matrix.xlsx", engine = 'xlsxwriter')
        self.sum_load_matrix.to_excel(writer, "Sort_sum_load_matrix")
        writer.save()
        writer.close()

    # 9) Анализ остатков
    def Residue_analysis(self):
        # Копирование матрицы корреляций по значению
        reduc = self.corr.copy()
        # Замена элементов главной диагонали на значения nan
        np.fill_diagonal(reduc.values, np.nan)
        reduc.max()
        # Замена элементов главной диагонали на максимальное значение столбца (получение редуцированной матрицы)
        for col in reduc.columns:
          reduc[col][col] = reduc[col].max()

        sum = 0
        self.remains_list = pd.Series(index = self.factor_load_matrix.columns)
        for col in self.factor_load_matrix.columns:
            # Скалярное произведение по каждому столбцу матрицы нагрузок
            reproduced = np.matrix(self.factor_load_matrix[col]).dot(np.matrix(self.factor_load_matrix[col].transpose()).reshape(self.factor_load_matrix.shape[0], 1))
            # Поиск остатков путём вычитания полученного значения из редуцированной матрицы
            remains = np.matrix(reduc) - reproduced
            self.remains_list[col] = remains.mean()
            sum = sum + remains.mean()
        self.remains_mean = sum / (self.factor_load_matrix.shape[0])

        print("Остатки:")
        for i in range(len(self.remains_list)):
            print(self.remains_list.index[i], ": ", round(self.remains_list[i], 4))
        print("Средний остаток: ", round(self.remains_mean, 4))

        self.remains_list['mean'] = self.remains_mean

        writer = pd.ExcelWriter("Results\\" + self.dataset_name + "_remains_list.xlsx", engine = 'xlsxwriter')
        self.remains_list.to_excel(writer, "Remains_list")
        writer.save()
        writer.close()

    # Запуск всех шагов реализации метода главных компонент
    def Start_PCA(self):
        # 2) Построение матрицы корреляций и составление списка признаков со слабой корреляцией
        self.Correlation()

        # 3) Подготовка данных для метода главных компонент на основе составленного списка
        self.Data_preparing()

        # 4) Реализация метода главных компонент
        self.PCA_Realization()

        # 5) Реализация критериев выбора числа главных компонент (Кайзера, сломанной трости, каменистой осыпи)
        self.Criteria_Realization()

        # 6) Получение результирующего значения количества главных компонент, исходя из полученных значений (среднее значение)
        self.result_components = round((self.components_broken_stick + self.components_Kaizer + int(self.components_scree)) / 3)
        print("Итоговое значение количества оставляемых главных компонент: ", self.result_components)
        print()

        count_of_components_file = open("Results\\" + self.dataset_name + "_criterias_components.txt", 'w')
        count_of_components_file.write("Исходя из критерия Кайзера, необходимо оставить " + str(self.components_Kaizer) + " главных компонент.\n")
        count_of_components_file.write("Исходя из критерия сломанной трости, необходимо оставить " + str(self.components_broken_stick) + " главных компонент.\n")
        count_of_components_file.write("Исходя из критерия каменистой осыпи, необходимо оставить " + str(self.components_scree) + " главных компонент.\n")
        count_of_components_file.write("Итоговое значение количества оставляемых главных компонент: " + str(self.result_components))
        count_of_components_file.close()

        # 7) Построение матрицы факторной нагрузки для всех значимых главных компонент
        self.Get_factor_load_matrix()

        # 8) Определение суммарной нагрузки для признаков. Составление ранжированного списка признаков (по убыванию нагрузки)
        self.Sum_load()

        # 9) Анализ остатков
        self.Residue_analysis()
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd

import json

class Dataset:
    path = None # Путь к датасету
    dataset_name = None # Имя датасета
    metadata_path = None # Путь к метаданным
    data = None # Сами данные

    X = None # Признаки
    Y = None # Классы
    Y_numpy = None # Классы в типе numpy array (необходим при вычислении метрик кластерного анализа)

    count_of_X = None # Количество столбцов с признаками
    count_of_Y = None # Количество столбцов с классами

    count_of_classes = None # Количество классов

    X_train = None # X обучающие
    X_test = None # Х тестирующие
    Y_train = None # Y обучающие
    Y_test = None # Y тестирующие

    # Стандартный конструктор класса (принимает на вход путь к датасету)
    def __init__(self, path, dataset_name = "", metadata_path = None):
        self.path = path
        self.dataset_name = dataset_name
        self.metadata_path = metadata_path

    # Обработка датасета
    # (столбцы Х из датасета, столбцы Y из датасета, перемешать данные?, доля обучающей выборки)
    def Preprocessing_data(self, shuffle_ = True, split = 0.7):
        self.data = pd.read_csv(self.path) # Открытие датасета

        # Перемешивание данных
        if(shuffle_):
            self.data = shuffle(self.data)

        # Удаление строк с пустыми значениями
        self.data = self.data.dropna()

        # Замена строковых данных на категориальные
        category_columns = self.data.select_dtypes(include = ['object'])
        for column in category_columns.columns:
            self.data[column] = self.data[column].astype('category')
            self.data[column] = self.data[column].cat.codes
            self.data[column] = self.data[column].astype('category')

        print(self.dataset_name)
        print("data = \n", self.data)
        print("columns = \n", self.data.columns)

        # Из дополнительного файла получение X- и Y-значений, а также разделение на обучающую и тестовую выборки
        if (self.metadata_path is not None):
            json_data = None
            with open(self.metadata_path) as write_file:
                json_data = json.load(write_file)

                columns_list = []
                for key, value in json_data.items():
                    columns_list.append([key, value])
                columns_list.sort()

                self.count_of_X = len(columns_list[0][1])
                self.count_of_Y = len(columns_list[1][1])

                print("count_of_X = ", self.count_of_X)
                print("count_of_y = ", self.count_of_Y)

                for i in columns_list[0][1]:
                    self.X = pd.concat([self.X, self.data[i]], axis = 1)

                for i in columns_list[1][1]:
                    self.Y = pd.concat([self.Y, self.data[i]], axis = 1)

                print("X = \n", self.X)
                print("Y = \n", self.Y)

                self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, train_size = split, random_state = 1, shuffle = shuffle_)

        if(self.count_of_Y == 1):
            self.count_of_classes = len(self.Y.iloc[:, 0].unique())
            print("count_of_classes = ", self.count_of_classes)

            self.Y_numpy = self.Y.to_numpy().transpose()[0]
        else:
            print("Количество столбцов с классами больше 1! Возможны проблемы при вычислении метрик!")
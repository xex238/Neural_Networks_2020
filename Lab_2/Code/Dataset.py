from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd

class Dataset:
    path = None # Путь к датасету
    dataset_name = None # Имя датасета
    metadata_path = None # Путь к метаданным
    data = None # Сами данные

    X = None # Признаки
    Y = None # Классы

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

        if (self.metadata_path is not None):
            metadata = []
            with open(self.metadata_path) as metadata_file:
                for line in metadata_file:
                    metadata.append(line)

            for i in X_columns:
                self.X = pd.concat([self.X, self.data.iloc[:, i:i+1]], axis = 1)

            for i in Y_columns:
                self.Y = pd.concat([self.Y, self.data.iloc[:, i:i+1]], axis = 1)

            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = shuffle_, random_state = 1)
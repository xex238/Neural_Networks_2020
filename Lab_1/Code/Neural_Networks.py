# matplotlib libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# sklearn libraries
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# additional libraries
import numpy as np
import pandas as pd
import csv

class Neural_Networks:
    # Разделённый на выборки датасет
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

    # Виды обучения
    log_regression = None
    perceptron = None
    mlp = None

    # Результаты обучения
    log_regression_result = None
    perceptron_result = None
    mlp_result = None

    # Метрики
    lg_metrics = None
    clf_metrics = None
    mlp_metrics = None

    # Конструктор - разделение датасета на обучающую и тестовую выборки
    def __init__(self, X, Y, part):
        self.X_train = X.iloc[:round(len(X) * part), :]
        self.X_test = X.iloc[round(len(X) * part):len(X), :]
        self.Y_train = Y.iloc[:round(len(Y) * part), :]
        self.Y_test = Y.iloc[round(len(Y) * part):len(Y), :]

        #print(self.X_train)
        #print(self.X_test)
        #print(self.Y_train)
        #print(self.Y_test)

        print(type(self.X_train))
        print(type(self.X_test))
        print(type(self.Y_train))
        print(type(self.Y_test))

    # Отрисова обучающего и тестового множеств
    def Draw_train_test(self, dataset_name):
        fig1, axs1 = plt.subplots(2, 1,  figsize=(16, 10), sharey=True)
        fig1.suptitle('Predictions')

        print("type(axs1) = ", type(axs1))

        axs1[0].scatter(self.X_train.iloc[:, 0], self.X_train.iloc[:, 1], c = self.Y_train.to_numpy(), label = 'Train data')
        axs1[1].scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c = self.Y_test.to_numpy(), label = 'Test data')

        axs1[0].legend(loc = 0)
        axs1[1].legend(loc = 0)

        #plt.show()
        plt.savefig("Results\\Figures\\train_test_" + dataset_name + ".jpeg")

    # Отрисовка полученных значений после классификации
    def Draw(self, dataset_name):
        fig1, axs1 = plt.subplots(2, 2,  figsize=(16, 10), sharey=True)
        fig1.suptitle('Predictions')

        Y_test_matrix = self.Y_test.to_numpy()

        print("type(axs1) = ", type(axs1))
        print("type(self.log_regression_result) = ", type(self.log_regression_result))

        axs1[0][0].scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c = Y_test_matrix, label = 'Start classification')
        axs1[0][1].scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c = self.log_regression_result, label = 'Logistic regression')
        axs1[1][0].scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c = self.perceptron_result, label = 'Perceptron')
        axs1[1][1].scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c = self.mlp_result, label = 'MLP')

        axs1[0][0].legend(loc = 0)
        axs1[0][1].legend(loc = 0)
        axs1[1][0].legend(loc = 0)
        axs1[1][1].legend(loc = 0)

        #plt.show()
        plt.savefig("Results\Figures\\" + dataset_name + "_" + "classifications_result.jpeg")

    # Реализация линейной регрессии
    def Logistic_regression_realization(self):
        self.log_regression = LogisticRegression()

        # Обучение
        self.log_regression.fit(self.X_train, self.Y_train)

        # Прогнозирование
        self.log_regression_result = self.log_regression.predict(self.X_test)
        #print(self.log_regression_result)

    # Реализация однослойного персептрона
    def Perceptron_realization(self):
        self.perceptron = Perceptron(tol = 1e-3, random_state = 0)

        # Обучение
        self.perceptron.fit(self.X_train, self.Y_train)

        # Прогнозирование
        self.perceptron_result = self.perceptron.predict(self.X_test)
        #print(self.perceptron_result)

    # Реализация многослойного персептрона
    def MLP_realization(self):
        self.mlp = MLPClassifier(hidden_layer_sizes = (4), max_iter = 1000)

        # Обучение
        self.mlp.fit(self.X_train, self.Y_train)

        # Прогнозирование
        self.mlp_result = self.mlp.predict(self.X_test)
        #print(self.mlp_result)

    # Реализация методов обучения
    def Learning(self):
        self.Logistic_regression_realization()
        self.Perceptron_realization()
        self.MLP_realization()

    # Подсчёт метрик для линейного классификатора
    def Get_linear_metrix(self):
        self.lg_metrics = precision_recall_fscore_support(self.Y_test, self.log_regression_result, average = 'macro') + (accuracy_score(self.Y_test, self.log_regression_result), )
        print('Logistic regression\n\tPrecision: {}\n\tRecall: {}\n\tF-score: {}\n\tAccuracy: {}'.format(
            self.lg_metrics[0], self.lg_metrics[1], self.lg_metrics[2], self.lg_metrics[4]))

    # Подсчёт метрик для однослойного персептрона
    def Get_perceptron_metrix(self):
        self.clf_metrics = precision_recall_fscore_support(self.Y_test, self.perceptron_result, average = 'macro') + (accuracy_score(self.Y_test, self.perceptron_result), )
        print('Perceptron\n\tPrecision: {}\n\tRecall: {}\n\tF-score: {}\n\tAccuracy: {}'.format(
            self.clf_metrics[0], self.clf_metrics[1], self.clf_metrics[2], self.clf_metrics[4]))

    # Подсчёт метрик для многослойного персептрона
    def Get_MLP_metrix(self):
        self.mlp_metrics = precision_recall_fscore_support(self.Y_test, self.mlp_result, average = 'macro') + (accuracy_score(self.Y_test, self.mlp_result), )
        print('MLP\n\tPrecision: {}\n\tRecall: {}\n\tF-score: {}\n\tAccuracy: {}'.format(
            self.mlp_metrics[0], self.mlp_metrics[1], self.mlp_metrics[2], self.mlp_metrics[4]))

    # Подсчёт метрик
    def Get_metrix(self):
        metrix = []
        self.Get_linear_metrix()
        self.Get_perceptron_metrix()
        self.Get_MLP_metrix()

    # Отрисовка результатов работы выбранного метода обучения
    def Draw_results(self, learning_type, dataset_name):
        x_min, x_max = self.X_test.iloc[:, 0].min() - 1, self.X_test.iloc[:, 0].max() + 1
        y_min, y_max = self.X_test.iloc[:, 1].min() - 1, self.X_test.iloc[:, 1].max() + 1

        # Получение координатной сетки двумерного пространства (arange - равномерное распределение данных)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # ravel - сжатие матрицы до массива, c_ - объединение в матрицу
        X_plot = np.c_[xx.ravel(), yy.ravel()]

        Z = None
        if(learning_type == "Logistic_regression"):
            Z = self.log_regression.predict(X_plot)
        if(learning_type == "Perceptron"):
            Z = self.perceptron.predict(X_plot)
        if(learning_type == "MLP"):
            Z = self.mlp.predict(X_plot)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize = (20, 10))
        plt.contourf(xx, yy, Z, alpha = 0.3, cmap = 'Spectral')
        colors = self.Y_test.to_numpy()
        plt.scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], c = colors, s = 70)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(learning_type)

        #plt.show()
        plt.savefig("Results\Figures\\" + dataset_name + "_" + learning_type + ".jpeg")

    # Отрисовка результатов работы всех методов обучения
    def Draw_all_results(self, dataset_name):
        self.Draw_results("Logistic_regression", dataset_name)
        self.Draw_results("Perceptron", dataset_name)
        self.Draw_results("MLP", dataset_name)
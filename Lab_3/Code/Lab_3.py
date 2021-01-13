# Методы кластерного анализа
from sklearn.cluster import k_means
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# Импорт показателей качества разделения (импорт метрик)
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import f1_score

# Метод для генерации точек
from sklearn.datasets import make_blobs

# Импорт модуля для визуализации
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# Модули для хранения и обработки данных
import numpy as np
import pandas as pd

# Модуль для чтени/генерации датасетов с расширением csv
import csv

# Модуль для вычислений
import math

# Мои классы
import Dataset
import Klaster_Analysis

# Метод генерации точек для датасета с сохранением полученных значений в файл
def Generate_points(Separability, count_of_points, file_name):
    # Separation - разделимость классов (1.05 для линейно разделимых, 2 для пересечения 10-20%, 3.5 для пересечения 50-70%)
    # Separability = 3.5

    # Генерация выборки
    # X_blobs - точки (двумерный массив), y_blobs - классы (массив)
    (X_blobs, y_blobs) = make_blobs(n_samples = count_of_points, n_features = 2, centers = 4, cluster_std = Separability, random_state = 20)

    # Объединение точек и классов для записи в csv файл
    result = []
    line = ["X", "Y", "Class"]
    result.append(line)
    for i in range(len(X_blobs)):
        line = []
        line.append(X_blobs[i][0])
        line.append(X_blobs[i][1])
        line.append(y_blobs[i])
        result.append(line)

    print(result)

    ## Запись полученных значений в csv файл
    #full_file_name = 'C:\Datasets\Points\\' + file_name + ".csv"
    #print(full_file_name)
    #myFile = open(full_file_name, 'w', newline='')
    #writer = csv.writer(myFile)
    #writer.writerows(result)
    #myFile.close()

    #print("Writing complete!")

    # Вывод сгенерированной выборки на экран
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharey=True)
    fig.suptitle('Сгенерированная выборка, коэф. разделения {}'.format(Separability))
    axs.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)

    patch1 = mpatches.Patch(color='yellow', label='Class1')
    patch2 = mpatches.Patch(color='purple', label='Class2')
    patch3 = mpatches.Patch(color='darkcyan', label='Class3')
    patch4 = mpatches.Patch(color='lightgreen', label='Class4')

    axs.legend(loc=2, handles=[patch1, patch2, patch3, patch4])

    plt.savefig("Results\\" + file_name + ".jpeg")

    plt.show()

# -----
# 0) Генерация двумерных данных (точек) для датасета
#Generate_points(3.5, 1000, 'points_50_70.csv')
#Generate_points(2, 1000, 'points_10_20.csv')
#Generate_points(1.05, 1000, 'points_linear.csv')
#Generate_points(0.005, 1000, "points_extra_linear")

# -----
# 1) Определение путей к датасетам и метаданным
# Эталонные датасеты
red_wine_quality_dataset_path = "C:\Datasets\Red_Wine_Quality\winequality-red.csv"
breast_cancer_dataset_path = "C:\Datasets\Breast_Cancer_Dataset\data.csv"

# Самостоятельно сгенерированные датасеты
points_50_70_dataset_path = "C:\Datasets\Points\points_50_70.csv"
points_10_20_dataset_path = "C:\Datasets\Points\points_10_20.csv"
points_linear_dataset_path = "C:\Datasets\Points\points_linear.csv"
points_exta_linear_dataset_path = "C:\Datasets\Points\points_extra_linear.csv" # Ещё не сгенерирован

# Пути к метаданным
red_wine_quality_metadata_path = "C:\Datasets\Red_Wine_Quality\Red_Wine_Quality_metadata.json"
breast_cancer_metadata_path = "C:\Datasets\Breast_Cancer_Dataset\Breast_Cancer_metadata.json"
points_metadata_path = "C:\Datasets\Points\Points_metadata.json"

# -----
# 2) Предобработка данных
red_wine_quality_data = Dataset.Dataset(red_wine_quality_dataset_path, dataset_name="red_wine_quality", metadata_path=red_wine_quality_metadata_path)
red_wine_quality_data.Preprocessing_data(shuffle_=False)

breast_cancer_data = Dataset.Dataset(breast_cancer_dataset_path, dataset_name="breast_cancer", metadata_path=breast_cancer_metadata_path)
breast_cancer_data.Preprocessing_data(shuffle_=False)

points_50_70_data = Dataset.Dataset(points_50_70_dataset_path, dataset_name="points_50_70", metadata_path=points_metadata_path)
points_50_70_data.Preprocessing_data(shuffle_=False)

points_10_20_data = Dataset.Dataset(points_10_20_dataset_path, dataset_name="points_10_20", metadata_path=points_metadata_path)
points_10_20_data.Preprocessing_data(shuffle_=False)

points_linear_data = Dataset.Dataset(points_linear_dataset_path, dataset_name="points_linear", metadata_path=points_metadata_path)
points_linear_data.Preprocessing_data(shuffle_=False)

points_exta_linear_data = Dataset.Dataset(points_exta_linear_dataset_path, dataset_name="points_exta_linear", metadata_path=points_metadata_path)
points_exta_linear_data.Preprocessing_data(shuffle_=False)

# -----
# 3) Реализация методов кластерного анализа

metrix = pd.Series([])

# Самостоятельно сгенерированные датасеты
points_extra_linear_klaster_analysis = Klaster_Analysis.Klaster_Analysis(points_exta_linear_data)
points_extra_linear_klaster_analysis.Methods_computing()
metrix['points_extra_linear']=points_extra_linear_klaster_analysis.metrics

points_linear_klaster_analysis = Klaster_Analysis.Klaster_Analysis(points_linear_data)
points_linear_klaster_analysis.Methods_computing()
metrix['points_linear']=points_linear_klaster_analysis.metrics

points_10_20_klaster_analysis = Klaster_Analysis.Klaster_Analysis(points_10_20_data)
points_10_20_klaster_analysis.Methods_computing()
metrix['points_10_20']=points_10_20_klaster_analysis.metrics

points_50_70_klaster_analysis = Klaster_Analysis.Klaster_Analysis(points_50_70_data)
points_50_70_klaster_analysis.Methods_computing()
metrix['points_50_70']=points_50_70_klaster_analysis.metrics

# Эталонные датасеты
red_wine_quality_klaster_analysis = Klaster_Analysis.Klaster_Analysis(red_wine_quality_data, draw_results=False)
red_wine_quality_klaster_analysis.Methods_computing()
metrix['red_wine_quality']=red_wine_quality_klaster_analysis.metrics

breast_cancer_klaster_analysis = Klaster_Analysis.Klaster_Analysis(breast_cancer_data, draw_results=False)
breast_cancer_klaster_analysis.Methods_computing()
metrix['breast_cancer']=breast_cancer_klaster_analysis.metrics

#print(points_extra_linear_klaster_analysis.metrics)
#print()
#print(points_linear_klaster_analysis.metrics)
#print()
#print(points_10_20_klaster_analysis.metrics)
#print()
#print(points_50_70_klaster_analysis.metrics)
#print()

#print(red_wine_quality_klaster_analysis.metrics)
#print()
#print(breast_cancer_klaster_analysis.metrics)

#print(metrix['points_10_20'])
#print(metrix['points_50_70'])
#print(metrix['red_wine_quality'])

# Сохранение полученных метрик в файл
writer = pd.ExcelWriter("Results\\Factor_Load_Matrix.xlsx", engine = 'xlsxwriter')
points_extra_linear_klaster_analysis.metrics.to_excel(writer, "points_extra_linear_metrix")
points_linear_klaster_analysis.metrics.to_excel(writer, "points_linear_metrix")
points_10_20_klaster_analysis.metrics.to_excel(writer, "points_10_20_metrix")
points_50_70_klaster_analysis.metrics.to_excel(writer, "points_50_70_metrix")
red_wine_quality_klaster_analysis.metrics.to_excel(writer, "red_wine_quality_metrix")
breast_cancer_klaster_analysis.metrics.to_excel(writer, "breast_cancer_metrix")
writer.save()
writer.close()
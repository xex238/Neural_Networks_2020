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

# My functions
import Dataset
import Datasets
import PCA_Realization

# Пути к файлам
# -----
points_linear_dataset_path = "C:\Datasets\Points\points_linear.csv"

Anuran_Calls_Dataset_22_path = "C:\Datasets\Anuran_Calls_Dataset_22\Frogs_MFCCs.csv"
Diabetic_Dataset_55_path = "C:\Datasets\Diabetes_Dataset_55\diabetic_data.csv"
DetectingMalwareUsingAHybridApproach_Datasets_1087_path = "C:\Datasets\DetectingMalwareUsingAHybridApproach_Datasets_1087\staDynVxHeaven2698Lab.csv"
# -----

# Предобработка датасетов
# -----
## Датасет точек
#points_linear_data = Dataset.Dataset(points_linear_dataset_path, dataset_name="points_linear")
#points_linear_data.Preprocessing_data(True)

## Датасет с 22-мя признаками
#Anuran_Calls_Dataset_22 = Dataset.Dataset(Anuran_Calls_Dataset_22_path, dataset_name="Anuran_Calls_Dataset_22")
#Anuran_Calls_Dataset_22.Preprocessing_data(shuffle_ = False)

## Датасет с 55-ю признаками
#Diabetic_Dataset_55 = Dataset.Dataset(Diabetic_Dataset_55_path, dataset_name="Diabetic_Dataset_55")
#Diabetic_Dataset_55.Preprocessing_data(shuffle_ = False)

# Датасет с 1087-ю признаками
DetectingMalwareUsingAHybridApproach_Datasets_1087 = Dataset.Dataset(DetectingMalwareUsingAHybridApproach_Datasets_1087_path, dataset_name="DetectingMalwareUsingAHybridApproach_Datasets_1087")
DetectingMalwareUsingAHybridApproach_Datasets_1087.Preprocessing_data(shuffle_ = False)
# -----

# Реализация метода главных компонент
# -----
#pca = PCA_Realization.PCA_Realization(points_linear_data.data) # Работает
#pca.Start_PCA()

#pca_22 = PCA_Realization.PCA_Realization(Anuran_Calls_Dataset_22.data, dataset_name="Anuran_Calls_Dataset_22")
#pca_22.Start_PCA()

#pca_55 = PCA_Realization.PCA_Realization(Diabetic_Dataset_55.data, dataset_name="Diabetic_Dataset_55")
#pca_55.Start_PCA()

pca_1087 = PCA_Realization.PCA_Realization(DetectingMalwareUsingAHybridApproach_Datasets_1087.data, dataset_name="DetectingMalwareUsingAHybridApproach_Datasets_1087")
pca_1087.Start_PCA()
# -----
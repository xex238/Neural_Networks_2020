# Методы кластерного анализа
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# Импорт показателей качества разделения (импорт метрик для кластеризации)
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

# Подбор по сетке для поиска лучших значений для DBSCAN
from sklearn.model_selection import GridSearchCV

# Импорт модуля для визуализации
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# Модуль для хранения и обработки данных
import pandas as pd

# Мои классы
import Dataset

class Klaster_Analysis:
    data = None # Данные датасета

    # Переменные для реализации k-means
    kmeans = None # Экземпляр класса k_means
    kmeans_predict = None # Прогнозы для тестовой выборки метода k_means

    # Переменные для реализации AgglomerativeClustering
    agg_clustering = None # Экземпляр класса AgglomerativeClustering
    agg_clustering_predict = None # Прогнозы для тестовой выборки выборки метода AgglomerativeClustering

    # Переменные для реализации метода DBSCAN
    dbscan = None # Экземпляр класса DBSCAN
    dbscan_predict = None # Прогнозы для тестовой выборки метода DBSCAN

    # Переменные для реализации метода KMedoids
    kmedoids = None # Экземпляр класса KMedoids
    kmedoids_predict = None # Прогнозы для тестовой выборки метода KMedoids

    metrics = None # DataFrame с рассчитываемыми метриками

    draw_results = True # Проводить ли отрисовку результатов

    # Стандартный конструктор датасета (принятие датасета в качестве параметра)
    def __init__(self, data, draw_results=True):
        self.data = data
        self.metrics = pd.DataFrame(columns = ['adjusted_rand', 'completeness', 'fowlkes_mallows', 'homogeneity'])
        self.draw_results = draw_results

    # Визуализация разделения для двумерных данных
    def Draw_results(self, method_name, predict):
        fig1, axs1 = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
        fig1.suptitle("Результат работы метода " + method_name)
        axs1[0].scatter(self.data.X.iloc[:, 0], self.data.X.iloc[:, 1], c=self.data.Y_numpy, label = 'Start points')
        axs1[1].scatter(self.data.X.iloc[:, 0], self.data.X.iloc[:, 1], c=predict, label = 'k_means classification')

        axs1[0].legend(loc = 0)
        axs1[1].legend(loc = 1)

        plt.savefig("Results\\" + self.data.dataset_name + "_" + method_name + "_klasterization_result.jpeg")

        #plt.show()

    # Реализация расчёта метрик
    def Compute_metrix(self, clusterization_name, predict):
        self.metrics.loc[clusterization_name, 'adjusted_rand'] = adjusted_rand_score(self.data.Y_numpy, predict)
        self.metrics.loc[clusterization_name, 'completeness'] = completeness_score(self.data.Y_numpy, predict)
        self.metrics.loc[clusterization_name, 'fowlkes_mallows'] = fowlkes_mallows_score(self.data.Y_numpy, predict)
        self.metrics.loc[clusterization_name, 'homogeneity'] = homogeneity_score(self.data.Y_numpy, predict)

        print(clusterization_name + " clustering metrics for " + self.data.dataset_name + " :")
        print("adjusted_rand_score: ", self.metrics.loc[clusterization_name, 'adjusted_rand'])
        print("kmeans_completeness_score: ", self.metrics.loc[clusterization_name, 'completeness'])
        print("fowlkes_mallows_score: ", self.metrics.loc[clusterization_name, 'fowlkes_mallows'])
        print("homogeneity_score: ", self.metrics.loc[clusterization_name, 'homogeneity'])
        print()

    # Реализация метода k-means
    def KMeans_realization(self):
        self.kmeans = KMeans(n_clusters=self.data.count_of_classes)
        self.kmeans.fit(self.data.X)
        self.kmeans_predict = self.kmeans.predict(self.data.X)

        self.Compute_metrix('K_means', self.kmeans_predict)

        if(self.draw_results):
            self.Draw_results("K_means", self.kmeans_predict)

    # Реализация агломеративного метода кластеризации
    def Agglomerative_Clustering_realization(self):
        # Агломеративный метод с евклидовой метрикой
        self.agg_clustering = AgglomerativeClustering(n_clusters=self.data.count_of_classes, affinity='euclidean')
        self.agg_clustering.fit(self.data.X)
        self.agg_clustering_predict = self.agg_clustering.labels_

        self.Compute_metrix('Agglomerative_Clustering_euclidean', self.agg_clustering_predict)

        if(self.draw_results):
            self.Draw_results("Agglomerative_Clustering_euclidean", self.agg_clustering_predict)

        # Агломеративный метод с манхеттенской метрикой
        self.agg_clustering = AgglomerativeClustering(n_clusters=self.data.count_of_classes, affinity='manhattan', linkage='complete') # linkage='single'
        self.agg_clustering.fit(self.data.X)
        self.agg_clustering_predict = self.agg_clustering.labels_

        self.Compute_metrix('Agglomerative_Clustering_manhattan', self.agg_clustering_predict)

        if(self.draw_results):
            self.Draw_results("Agglomerative_Clustering_manhattan", self.agg_clustering_predict)

    # Реализация метода DBSCAN
    def DBSCAN_realization(self):
        self.dbscan = DBSCAN(metric='euclidean', eps=0.7, min_samples=4)
        self.dbscan.fit(self.data.X)
        self.dbscan_predict = self.dbscan.labels_

        self.Compute_metrix('DBSCAN_euclidean', self.dbscan_predict)

        if(self.draw_results):
            self.Draw_results("DBSCAN_euclidean", self.dbscan_predict)

        self.dbscan = DBSCAN(metric='manhattan', eps=0.7, min_samples=4)
        self.dbscan.fit(self.data.X)
        self.dbscan_predict = self.dbscan.labels_

        self.Compute_metrix('DBSCAN_manhattan', self.dbscan_predict)

        if(self.draw_results):
            self.Draw_results("DBSCAN_manhattan", self.dbscan_predict)

    # Реализация метода KMedoids
    def KMedoids_realization(self):
        self.kmedoids = KMedoids(n_clusters=self.data.count_of_classes, metric='euclidean')
        self.kmedoids.fit(self.data.X)
        self.kmedoids_predict = self.kmedoids.predict(self.data.X)

        self.Compute_metrix('KMedoids_euclidean', self.kmedoids_predict)

        if(self.draw_results):
            self.Draw_results("KMedoids_euclidean", self.kmedoids_predict)

        self.kmedoids = KMedoids(n_clusters=self.data.count_of_classes, metric='manhattan')
        self.kmedoids.fit(self.data.X)
        self.kmedoids_predict = self.kmedoids.predict(self.data.X)

        self.Compute_metrix('KMedoids_manhattan', self.kmedoids_predict)

        if(self.draw_results):
            self.Draw_results("KMedoids_manhattan", self.kmedoids_predict)

    # Запуск всех методов кластеризации
    def Methods_computing(self):
        self.KMeans_realization()
        self.Agglomerative_Clustering_realization()
        self.DBSCAN_realization()
        self.KMedoids_realization()
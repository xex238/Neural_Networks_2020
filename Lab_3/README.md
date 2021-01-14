# Результаты работы алгоритмов кластеризации на различных датасетах

## Датасет "points_linear" (группы расположены близко или касаются друг друга)

### Алгоритм KMeans

![points_linear_K_means_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_linear_K_means_klasterization_result.jpeg?raw=true)

### Метод AgglomerativeClustering

#### Евклидова метрика

![points_linear_Agglomerative_Clustering_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_linear_Agglomerative_Clustering_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_linear_Agglomerative_Clustering_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_linear_Agglomerative_Clustering_manhattan_klasterization_result.jpeg?raw=true)

### Метод DBSCAN

#### Евклидова метрика

![points_linear_DBSCAN_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_linear_DBSCAN_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_linear_DBSCAN_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_linear_DBSCAN_manhattan_klasterization_result.jpeg?raw=true)

### Метод KMedoids

#### Евклидова метрика

![points_linear_KMedoids_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_linear_KMedoids_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_linear_KMedoids_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_linear_KMedoids_manhattan_klasterization_result.jpeg?raw=true)

### Значения метрик

|                    | adjusted_rand | completeness | fowlkes_mallows | homogeneity |
| ------------------ | :-----------: | :----------: | :-------------: | :---------: |
| K_means            |     1,000     |    1,000     |      1,000      |    1,000    |
| AC_euclidean       |     1,000     |    1,000     |      1,000      |    1,000    |
| AC_manhattan       |     0,864     |    0,901     |      0,898      |    0,893    |
| DBSCAN_euclidean   |     0,964     |    0,920     |      0,973      |    0,976    |
| DBSCAN_manhattan   |     0,932     |    0,868     |      0,949      |    0,955    |
| KMedoids_euclidean |     0,626     |    0,840     |      0,742      |    0,737    |
| KMedoids_manhattan |     0,997     |    0,995     |      0,998      |    0,995    |

## Датасет "points_10_20" (средняя площадь пересечения классов 10-20%)

### Алгоритм KMeans

![points_10_20_K_means_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_10_20_K_means_klasterization_result.jpeg?raw=true)

### Метод AgglomerativeClustering

#### Евклидова метрика

![points_10_20_Agglomerative_Clustering_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_10_20_Agglomerative_Clustering_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_10_20_Agglomerative_Clustering_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_10_20_Agglomerative_Clustering_manhattan_klasterization_result.jpeg?raw=true)

### Метод DBSCAN

#### Евклидова метрика

![points_10_20_DBSCAN_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_10_20_DBSCAN_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_10_20_DBSCAN_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_10_20_DBSCAN_manhattan_klasterization_result.jpeg?raw=true)

### Метод KMedoids

#### Евклидова метрика

![points_10_20_KMedoids_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_10_20_KMedoids_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_10_20_KMedoids_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_10_20_KMedoids_manhattan_klasterization_result.jpeg?raw=true)

### Значения метрик

|                    | adjusted_rand | completeness | fowlkes_mallows | homogeneity |
| ------------------ | :-----------: | :----------: | :-------------: | :---------: |
| K_means            |     0,863     |    0,837     |      0,897      |    0,837    |
| AC_euclidean       |     0,842     |    0,835     |      0,882      |    0,832    |
| AC_manhattan       |     0,764     |    0,768     |      0,823      |    0,766    |
| DBSCAN_euclidean   |     0,593     |    0,675     |      0,711      |    0,643    |
| DBSCAN_manhattan   |     0,490     |    0,524     |      0,619      |    0,624    |
| KMedoids_euclidean |     0,859     |    0,831     |      0,894      |    0,831    |
| KMedoids_manhattan |     0,544     |    0,694     |      0,680      |    0,619    |

## Датасет "points_50_70" (средняя площадь пересечения классов 50-70%)

### Алгоритм KMeans

![points_50_70_K_means_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_50_70_K_means_klasterization_result.jpeg?raw=true)

### Алгоритм AgglomerativeClustering

#### Евклидова метрика

![points_50_70_Agglomerative_Clustering_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_50_70_Agglomerative_Clustering_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_50_70_Agglomerative_Clustering_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_50_70_Agglomerative_Clustering_manhattan_klasterization_result.jpeg?raw=true)

### Метод DBSCAN

#### Евклидова метрика

![points_50_70_DBSCAN_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_50_70_DBSCAN_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_50_70_DBSCAN_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_50_70_DBSCAN_manhattan_klasterization_result.jpeg?raw=true)

### Метод KMedoids

#### Евклидова метрика

![points_50_70_KMedoids_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_50_70_KMedoids_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_50_70_KMedoids_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_50_70_KMedoids_manhattan_klasterization_result.jpeg?raw=true)

### Значения метрик

|                    | adjusted_rand | completeness | fowlkes_mallows | homogeneity |
| ------------------ | :-----------: | :----------: | :-------------: | :---------: |
| K_means            |     0,524     |    0,531     |      0,643      |    0,530    |
| AC_euclidean       |     0,460     |    0,490     |      0,596      |    0,485    |
| AC_manhattan       |     0,203     |    0,339     |      0,430      |    0,305    |
| DBSCAN_euclidean   |     0,013     |    0,151     |      0,330      |    0,168    |
| DBSCAN_manhattan   |     0,047     |    0,188     |      0,235      |    0,390    |
| KMedoids_euclidean |     0,352     |    0,425     |      0,519      |    0,415    |
| KMedoids_manhattan |     0,396     |    0,470     |      0,551      |    0,459    |

## Датасет "points_extra_linear" (с расстоянием между группами в $$10^3$$ раз больше, чем диаметр группы)

### Алгоритм KMeans

![points_exta_linear_K_means_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_exta_linear_K_means_klasterization_result.jpeg?raw=true)

### Алгоритм AgglomerativeClustering

#### Евклидова метрика

![points_exta_linear_Agglomerative_Clustering_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_exta_linear_Agglomerative_Clustering_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_exta_linear_Agglomerative_Clustering_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_exta_linear_Agglomerative_Clustering_manhattan_klasterization_result.jpeg?raw=true)

### Метод DBSCAN

#### Евклидова метрика

![points_exta_linear_DBSCAN_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_exta_linear_DBSCAN_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_exta_linear_DBSCAN_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_exta_linear_DBSCAN_manhattan_klasterization_result.jpeg?raw=true)

### Метод KMedoids

#### Евклидова метрика

![points_exta_linear_KMedoids_euclidean_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_exta_linear_KMedoids_euclidean_klasterization_result.jpeg?raw=true)

#### Манхеттенская метрика

![points_exta_linear_KMedoids_manhattan_klasterization_result.jpeg](https://github.com/xex238/Neural_Networks_2020/blob/main/Lab_3/Results/points_exta_linear_KMedoids_manhattan_klasterization_result.jpeg?raw=true)

### Значения метрик

|                    | adjusted_rand | completeness | fowlkes_mallows | homogeneity |
| ------------------ | :-----------: | :----------: | :-------------: | :---------: |
| K_means            |     1,000     |    1,000     |      1,000      |    1,000    |
| AC_euclidean       |     1,000     |    1,000     |      1,000      |    1,000    |
| AC_manhattan       |     1,000     |    1,000     |      1,000      |    1,000    |
| DBSCAN_euclidean   |     1,000     |    1,000     |      1,000      |    1,000    |
| DBSCAN_manhattan   |     1,000     |    1,000     |      1,000      |    1,000    |
| KMedoids_euclidean |     1,000     |    1,000     |      1,000      |    1,000    |
| KMedoids_manhattan |     0,630     |    0,858     |      0,746      |    0,750    |

## Датасет Red Wine Quality

### Значения метрик

|                    | adjusted_rand | completeness | fowlkes_mallows | homogeneity |
| ------------------ | :-----------: | :----------: | :-------------: | :---------: |
| K_means            |    -0,005     |    0,038     |      0,305      |    0,046    |
| AC_euclidean       |    -0,009     |    0,040     |      0,315      |    0,047    |
| AC_manhattan       |    -0,018     |    0,050     |      0,406      |    0,042    |
| DBSCAN_euclidean   |    -0,002     |    0,117     |      0,581      |    0,018    |
| DBSCAN_manhattan   |     0,000     |    0,144     |      0,588      |    0,013    |
| KMedoids_euclidean |     0,015     |    0,030     |      0,260      |    0,045    |
| KMedoids_manhattan |     0,007     |    0,033     |      0,269      |    0,048    |

## Датасет Breast Cancer

### Значения метрик

|                    | adjusted_rand | completeness | fowlkes_mallows | homogeneity |
| ------------------ | :-----------: | :----------: | :-------------: | :---------: |
| K_means            |     0,003     |    0,003     |      0,716      |    0,000    |
| AC_euclidean       |     0,003     |    0,003     |      0,716      |    0,000    |
| AC_manhattan       |     0,003     |    0,003     |      0,716      |    0,000    |
| DBSCAN_euclidean   |     0,000     |    1,000     |      0,729      |    0,000    |
| DBSCAN_manhattan   |     0,000     |    1,000     |      0,729      |    0,000    |
| KMedoids_euclidean |     0,005     |    0,001     |      0,549      |    0,001    |
| KMedoids_manhattan |     0,005     |    0,001     |      0,549      |    0,001    |
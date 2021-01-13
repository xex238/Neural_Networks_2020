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

# My functions
import Datasets
import Neural_Networks

# Метод генерации точек для датасета с сохранением полученных значений в файл
def Generate_points(Separability, count_of_points, file_name):
    # Separation - разделимость классов (1.05 для линейно разделимых, 2 для пересечения 10-20%, 3.5 для пересечения 50-70%)
    # Separability = 3.5

    # randomInt - случайное число из указанного диапазона. Если в датасете 400 объектов, а randomInt равен 136,
    # то первые 136 элементов будут отнесены к обучающей выборке, следующие 136 к валидационной, а оставшиеся к тестовой.
    randomInt = np.random.randint(136, 156)
    print('dataset random separator:', randomInt)

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

    # Запись полученных значений в csv файл
    full_file_name = 'C:\Datasets\Points\\' + file_name
    print(full_file_name)
    myFile = open(full_file_name, 'w', newline='')
    writer = csv.writer(myFile)
    writer.writerows(result)
    myFile.close()

    print("Writing complete!")

    # Вывод сгенерированной выборки на экран
    fig, axs = plt.subplots(1, 1, figsize=(12, 4), sharey=True)
    fig.suptitle('Сгенерированная выборка, коэф. разделения {}'.format(Separability))
    axs.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)

    patch1 = mpatches.Patch(color='yellow', label='Class1')
    patch2 = mpatches.Patch(color='purple', label='Class2')
    patch3 = mpatches.Patch(color='darkcyan', label='Class3')
    patch4 = mpatches.Patch(color='lightgreen', label='Class4')

    axs.legend(loc=2, handles=[patch1, patch2, patch3, patch4])

    plt.show()

# Обучение на одном наборе данных
def Learning(X, Y, dataset_name, results_drawing):
    metrix_df = pd.DataFrame(columns = ['learning_type', 'Accuracy', 'Precision', 'Recall', 'F_score'])
    counter = 0

    NN_points = Neural_Networks.Neural_Networks(X, Y, 0.7)

    if(results_drawing):
        NN_points.Draw_train_test(dataset_name)

    NN_points.Learning() # Обучение по всем методам

    if(results_drawing):
        NN_points.Draw(dataset_name) # Отрисовка результатов классификации
    NN_points.Get_metrix() # Вывод полученных метрик по всем типам обучения

    metrix_df.loc[counter] = [dataset_name, '', '', '', '']
    counter = counter + 1
    metrix_df.loc[counter] = ['Linear_classifier', NN_points.lg_metrics[4], NN_points.lg_metrics[0], NN_points.lg_metrics[1], NN_points.lg_metrics[2]]
    counter = counter + 1
    metrix_df.loc[counter] = ['Perceptron', NN_points.clf_metrics[4], NN_points.clf_metrics[0], NN_points.clf_metrics[1], NN_points.clf_metrics[2]]
    counter = counter + 1
    metrix_df.loc[counter] = ['MLP', NN_points.mlp_metrics[4], NN_points.mlp_metrics[0], NN_points.mlp_metrics[1], NN_points.mlp_metrics[2]]
    counter = counter + 1

    print(metrix_df)

    if(results_drawing):
        NN_points.Draw_all_results(dataset_name) # Отрисовка результатов классификации по всем типам обучения

    return metrix_df

## Генерация двумерных данных (точек) для датасета
#Generate_points(3.5, 1000, 'points_50_70.csv')
#Generate_points(2, 1000, 'points_10_20.csv')
#Generate_points(1.05, 1000, 'points_linear.csv')

datasets = Datasets.Datasets(10)
datasets.Preprocessing_datasets()

result_metrix_df = pd.DataFrame(columns = ['learning_type', 'Accuracy', 'Precision', 'Recall', 'F_score'])

result_metrix_df = result_metrix_df.append(Learning(datasets.X_PL_data, datasets.Y_PL_data, "points_linear", True), ignore_index = True)
result_metrix_df = result_metrix_df.append(Learning(datasets.X_P_10_20_data, datasets.Y_P_10_20_data, "points_10_20", True), ignore_index=True)
result_metrix_df = result_metrix_df.append(Learning(datasets.X_P_50_70_data, datasets.Y_P_50_70_data, "points_50_70", True), ignore_index=True)

result_metrix_df = result_metrix_df.append(Learning(datasets.X_SPE_data, datasets.Y_SPE_data.iloc[:, :1], "SPE_dataset", False), ignore_index = True)
result_metrix_df = result_metrix_df.append(Learning(datasets.X_CG_data, datasets.Y_CG_data, "CG_dataset", False), ignore_index = True)
#result_metrix_df = result_metrix_df.append(Learning(datasets.X_VGS_data, datasets.Y_VGS_data.iloc[:, :1], "VGS_dataset", False), ignore_index = True) # Не работает
result_metrix_df = result_metrix_df.append(Learning(datasets.X_RWQ_data, datasets.Y_RWQ_data, "RWQ_dataset", False), ignore_index = True)
result_metrix_df = result_metrix_df.append(Learning(datasets.X_iris_data, datasets.Y_iris_data, "iris_dataset", False), ignore_index = True)

writer = pd.ExcelWriter('Results\metrix.xlsx', engine = 'xlsxwriter')
result_metrix_df.to_excel(writer, 'Metrix')
writer.save()
writer.close()
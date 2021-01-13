from sklearn.utils import shuffle

import pandas as pd

class Datasets:
    # Тип - DataFrame
    students_performance_in_exams_data = pd.read_csv("C:\Datasets\Students_Performance_In_Exams\StudentsPerformance.csv")
    chess_game_data = pd.read_csv("C:\Datasets\Chess_Game_Dataset\games.csv")
    video_game_sales_data = pd.read_csv("C:\Datasets\Video_Game_Sales\\vgsales.csv")
    red_wine_quality_data = pd.read_csv("C:\Datasets\Red_Wine_Quality\winequality-red.csv")
    iris_data = pd.read_csv("C:\Datasets\Iris\iris.csv")

    points_50_70_data = pd.read_csv("C:\Datasets\Points\points_50_70.csv")
    points_10_20_data = pd.read_csv("C:\Datasets\Points\points_10_20.csv")
    points_linear_data = pd.read_csv("C:\Datasets\Points\points_linear.csv")
    
    X_SPE_data = [] # students_performance_in_exams_data
    Y_SPE_data = []
    X_CG_data = [] # chess_game_data
    Y_CG_data = []
    X_VGS_data = [] # video_game_sales_data
    Y_VGS_data = []
    X_RWQ_data = [] # red_wine_quality_data
    Y_RWQ_data = []
    X_iris_data =[] # iris_data
    Y_iris_data = []

    X_P_50_70_data = [] # points_50_70_data
    Y_P_50_70_data = []
    X_P_10_20_data = [] # points_10_20_data
    Y_P_10_20_data = []
    X_PL_data = [] # points_linear_data
    Y_PL_data = []

    number = 0

    def __init__(self, value):
        number = value
    
    # Замена строковых значений на числовые
    def Category_change(self, df):
        for i in df.columns:
            df[i] = df[i].astype('category')
            df[i] = df[i].cat.codes
            df[i] = df[i].astype('category')

        #print(df)
        #print(df.dtypes)

        return df

    # Обработка входного датасета students_performance_in_exams_data
    def Preprocessing_SPE(self):
        # Выделение признаков и классов
        self.X_SPE_data = self.students_performance_in_exams_data.iloc[:, 0:5]
        self.Y_SPE_data = self.students_performance_in_exams_data.iloc[:, 5:8]

        # Замена строковых значений на числовые
        self.X_SPE_data = self.Category_change(self.X_SPE_data)

    # Обработка входного датасета chess_game_data
    def Preprocessing_CG(self):
        # Выделение признаков и классов
        self.X_CG_data = self.chess_game_data.iloc[:, :6]
        self.X_CG_data = pd.concat([self.X_CG_data, self.chess_game_data.iloc[:, 7:16]], axis = 1)
        
        self.Y_CG_data = self.chess_game_data.iloc[:, 6:7]

        # Замена строковых значений на числовые
        self.X_CG_data = self.Category_change(self.X_CG_data)
        self.Y_CG_data = self.Category_change(self.Y_CG_data)

    # Обработка входного датасета video_game_sales_data
    def Preprocessing_VGS(self):
        #print(self.video_game_sales_data)
        #print(self.video_game_sales_data.dtypes)
        #print(type(self.video_game_sales_data))

        # Выделение признаков и классов
        self.X_VGS_data = self.video_game_sales_data.iloc[:, :6]
        self.Y_VGS_data = self.video_game_sales_data.iloc[:, 6:11]

        # Замена строковых значений на числовые
        self.X_VGS_data = self.Category_change(self.X_VGS_data)

    # Обработка входного датасета red_wine_quality_data
    def Preprocessing_RWQ(self):
        #print(self.red_wine_quality_data)

        # Выделение признаков и классов
        self.X_RWQ_data = self.red_wine_quality_data.iloc[:, :11]
        self.Y_RWQ_data = self.red_wine_quality_data.iloc[:, 11:12]

        #print(self.X_RWQ_data)
        #print(self.Y_RWQ_data)

    # Обработка входного датасета iris_data
    def Preprocessing_iris(self):
        #print(self.iris_data)

        # Выделение признаков и классов
        self.X_iris_data = self.iris_data.iloc[:, :4]
        self.Y_iris_data = self.iris_data.iloc[:, 4:5]

        #print(self.X_iris_data)
        #print(self.Y_iris_data)

        self.Y_iris_data = self.Category_change(self.Y_iris_data)

        #print(self.Y_iris_data)

    # Обработка входного сгенерированного датасета
    def Preprocessing_points(self):
        # Выделение признаков и классов
        self.X_P_50_70_data = self.points_50_70_data.iloc[:, :2]
        self.Y_P_50_70_data = self.points_50_70_data.iloc[:, 2:3]

        self.X_P_10_20_data = self.points_10_20_data.iloc[:, :2]
        self.Y_P_10_20_data = self.points_10_20_data.iloc[:, 2:3]

        self.X_PL_data = self.points_linear_data.iloc[:, :2]
        self.Y_PL_data = self.points_linear_data.iloc[:, 2:3]

        #print(self.X_P_50_70_data)
        #print(self.Y_P_50_70_data)

        #print(self.X_P_10_20_data)
        #print(self.Y_P_10_20_data)

        #print(self.X_PL_data)
        #print(self.Y_PL_data)

    # Перемешивание данных во входных датасетах
    def Shuffle_datasets(self):
        self.students_performance_in_exams_data = shuffle(self.students_performance_in_exams_data)
        self.chess_game_data = shuffle(self.chess_game_data)
        self.video_game_sales_data = shuffle(self.video_game_sales_data)
        self.red_wine_quality_data = shuffle(self.red_wine_quality_data)
        self.iris_data = shuffle(self.iris_data)
        self.points_50_70_data = shuffle(self.points_50_70_data)
        self.points_10_20_data = shuffle(self.points_10_20_data)
        self.points_linear_data = shuffle(self.points_linear_data)

    # Обработка входных датасетов
    # 1) Разделение данных на признаки и на классы
    # 2) Замена строковых значений на категориальные
    # 3) Перемешивание данных
    def Preprocessing_datasets(self):
        self.Shuffle_datasets()

        self.Preprocessing_SPE()
        self.Preprocessing_CG()
        self.Preprocessing_VGS()
        self.Preprocessing_RWQ()
        self.Preprocessing_iris()
        self.Preprocessing_points()
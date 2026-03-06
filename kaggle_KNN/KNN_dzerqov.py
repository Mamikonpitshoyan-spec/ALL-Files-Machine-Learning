import numpy as np
class KNeighborsClassifier:
    def __init__(self,k=5):
        self.k = k

    def fit_pred(X_train,y_train,X_test):
        y_pred = []











# import numpy as np

# class KNeighborsClassifier:

#     def __init__(self, k=5):
#         self.k = k

#     def fit_predict(self, X_train, y_train, X_test):
#         predictions = []

#         for x_test in X_test:
#             # 1. считаем расстояния от x_test до всех X_train
#             distances = np.linalg.norm(X_train - x_test, axis=1)

#             # 2. индексы ближайших соседей
#             indices = np.argsort(distances)[:self.k]

#             # 3. классы k соседей
#             k_labels = y_train[indices]

#             # 4. голосование
#             pred = np.bincount(k_labels).argmax()
#             predictions.append(pred)

#         return np.array(predictions)

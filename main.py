import numpy as np
class KNN:
    def __init__(self, k = 3, task = 'classification'):
        self.k = k
        self.task = task
    def euclidean_distance(self, a, b):
        self.a = a
        self.b = b
        return np.sqrt(np.sum((a-b)**2, axis =1))

    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y
    def predicted(self,X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)
    def _predict_single(self,x):
        distances = self.euclidean_distance(x,self.X_train)
        k_indices = np.argsort(distances)[:self.k]

        nn = [self.X_train[i].tolist() for i in k_indices]
        print('Nearest_neighbours: ', nn)

        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        print('Labels for Nearest_neighbours: ', k_nearest_labels)

        if self.task == 'classification':
            return self._majority_vote(k_nearest_labels)
        elif self.task == 'regression':
            return self._average(k_nearest_labels)
    def _majority_vote(self,labels):
        return np.argmax(np.bincount(labels))
    def _average(self,values):
        return np.mean(values)

if __name__ == "__main__":
    X_train = np.array([[0,0],[1,1],[2,2],[3,3]])

    y_train = np.array([0,0,1,1])

    X_test = np.array([[0.5, 0.5]])

    knn = KNN(k=3, task='classification')
    knn.fit(X_train, y_train)
    y_pred = knn.predicted(X_test)

    print("Predicted labels:", y_pred)


if __name__ == '__main__':
    X_train = np.array([[0,0],[1,1],[2,2],[3,3]])

    y_train = np.array([0,0,1,1])

    X_test = np.array([[2,2]])

    knn = KNN(k=3, task='regression')
    knn.fit(X_train, y_train)
    y_pred = knn.predicted(X_test)

    print('Predicted labels:', y_pred)

if __name__ == '__main__':
    X_train = np.array([[0,0],[1,1],[2,2],[3,3]])

    y_train = np.array([0,0,2,2])

    X_test = np.array([[3,3]])

    knn = KNN(k=3, task='regression')
    knn.fit(X_train, y_train)
    y_pred = knn.predicted(X_test)

    print('Predicted labels:', y_pred)



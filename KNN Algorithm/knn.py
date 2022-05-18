import numpy as np
from download_mnist import load
import time


class KNNClassifier:
    # finding L2 (euclidean) distance between two numpy arrays and returning their distances
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.square(point1 - point2))))

    # calls euclidean_distance function by iterating through elements of train_data
    # and stores the respective euclidean distance and labels into a list called nearest_distances
    def nearest_neighbors(self, train_data, test_row, train_labels, num_neighbors):
        nearest_distances = []
        for i in range(len(train_data)):
            distances = self.euclidean_distance(test_row, train_data[i])
            nearest_distances.append((train_labels[i], distances))
        # sorts the distances in ascending order
        nearest_distances.sort(key=lambda tup: tup[1])
        neighbors = []
        # storing nearest distance compared to k nearest neighbors
        for i in range(num_neighbors):
            neighbors.append(nearest_distances[i][0])
        return neighbors

    def predict_classification(self, train_data, test_data, train_labels, num_neighbors):
        output_labels = []
        for test_row in test_data:
            # calls the nearest neighbors function by passing each element of test data
            k_neighbors = self.nearest_neighbors(train_data, test_row, train_labels, num_neighbors)
            # classifying the test data by counting maximum number of labels around it
            prediction = max(set(k_neighbors), key=k_neighbors.count)
            output_labels.append(prediction)
        return output_labels


def plot_knn_classification():
    # classify using kNN
    # x_train = np.load('../x_train.npy')
    # y_train = np.load('../y_train.npy')
    # x_test = np.load('../x_test.npy')
    # y_test = np.load('../y_test.npy')
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.reshape(60000, 28, 28)
    x_test = x_test.reshape(10000, 28, 28)
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)

    start_time = time.time()  # calculating execution time
    knn_obj = KNNClassifier()  # object of class
    outputlabels = knn_obj.predict_classification(train_data=x_train,
                                                  test_data=x_test[0:40],
                                                  train_labels=y_train,
                                                  num_neighbors=2)
    # calculating the accuracy by comparing with the known test sata
    result = y_test[0:40] - outputlabels
    result = (1 - np.count_nonzero(result) / len(outputlabels))
    print( " classification of 40 images with above algorithm has below result:")
    print("---classification accuracy for knn on mnist: %s ---" % result)
    print("---execution time: %s seconds ---" % (time.time() - start_time))


plot_knn_classification()

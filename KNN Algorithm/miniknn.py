import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


class KNNClassifier:
    # finding L2 (euclidean) distance between two numpy arrays and returning their distances
    def euclidean_distance(self, point1, point2):
        distance = np.sum((np.square(point1 - point2)))
        return np.sqrt(distance)

    # calls euclidean_distance function by iterating through elements of train_data
    # and stores the respective euclidean distance and labels into a list called nearest_distances
    def nearest_neighbors(self, train_data, test_row, train_labels, num_neighbors):
        nearest_distances = []
        for i in range(len(train_data)):
            line_dist = self.euclidean_distance(test_row, train_data[i])
            nearest_distances.append((train_labels[i], line_dist))
        # sorts the distances in ascending order
        nearest_distances.sort(key=lambda tup: tup[1])
        neighbors = []
        # storing nearest distance compared to k nearest neighbors
        for i in range(num_neighbors):
            neighbors.append(nearest_distances[i][0])
        return neighbors

    def predict_classification(self, train_data, test_data, train_labels, num_neighbors):
        output_labels = []
        # calls the nearest neighbors function by passing each element of test data
        for test_row in test_data:
            k_neighbors = self.nearest_neighbors(train_data, test_row, train_labels, num_neighbors)
            # classifying the test data by counting maximum number of labels around it
            prediction = max(set(k_neighbors), key=k_neighbors.count)
            output_labels.append(prediction)
        return output_labels


def plot_knn_classification():
    mini_train = np.load('C:\\Users\\rlohi\PycharmProjects\\pythonProject\\'
                         'Introduction to Deep Learning\\knn_minitrain.npy')
    mini_train_label = np.load('C:\\Users\\rlohi\PycharmProjects\\pythonProject\\'
                               'Introduction to Deep Learning\\knn_minitrain_label.npy')
    print("Training Data\n", mini_train)
    # randomly generate test data
    mini_test = np.random.randint(20, size=20)
    mini_test = mini_test.reshape(10, 2)
    k_value = np.random.randint(3, 10)
    print("Testing Data\n",mini_test)

    knn_obj = KNNClassifier()  # object of the class to call its function
    # passing the input to the funtcion
    outputlabels = knn_obj.predict_classification(train_data=mini_train,
                                                  test_data=mini_test,
                                                  train_labels=mini_train_label,
                                                  num_neighbors=k_value)

    # plotting the train data in circle shape
    train_x = mini_train[:, 0]
    train_y = mini_train[:, 1]

    fig = plt.figure()
    plt.scatter(train_x[np.where(mini_train_label == 0)], train_y[np.where(mini_train_label == 0)], color='red')
    plt.scatter(train_x[np.where(mini_train_label == 1)], train_y[np.where(mini_train_label == 1)], color='blue')
    plt.scatter(train_x[np.where(mini_train_label == 2)], train_y[np.where(mini_train_label == 2)], color='yellow')
    plt.scatter(train_x[np.where(mini_train_label == 3)], train_y[np.where(mini_train_label == 3)], color='black')

    test_x = mini_test[:, 0]
    test_y = mini_test[:, 1]

    # plotting the classified test data in triangle shape
    outputlabels = np.array(outputlabels)
    print("Classification of Test Data into classes 1 to 4 respectively : ",outputlabels)
    plt.scatter(test_x[np.where(outputlabels == 0)], test_y[np.where(outputlabels == 0)], marker='^', color='red')
    plt.scatter(test_x[np.where(outputlabels == 1)], test_y[np.where(outputlabels == 1)], marker='^', color='blue')
    plt.scatter(test_x[np.where(outputlabels == 2)], test_y[np.where(outputlabels == 2)], marker='^', color='yellow')
    plt.scatter(test_x[np.where(outputlabels == 3)], test_y[np.where(outputlabels == 3)], marker='^', color='black')

    # save diagram as png file
    plt.savefig("miniknn.png")


# calling the main function
plot_knn_classification()

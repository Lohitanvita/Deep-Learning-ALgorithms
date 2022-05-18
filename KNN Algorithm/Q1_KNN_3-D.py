from math import sqrt


# finding L2 (euclidean) distance between two arrays and returning their distances
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)


# calls euclidean_distance function by iterating through elements of train_data
# and stores the respective euclidean distance into a list called nearest_distances
def nearest_neighbors(train_data, test_data, num_neighbors):
    nearest_distances = []
    for train_row in train_data:
        line_dist = euclidean_distance(test_data, train_row)
        nearest_distances.append((train_row, line_dist))
    # sorts the distances in ascending order
    nearest_distances.sort(key=lambda tup: tup[1])
    neighbors = []
    # storing nearest distance compared to k nearest neighbors
    for i in range(num_neighbors):
        neighbors.append(nearest_distances[i][0])
    return neighbors


# calls nearest_neighbors function by passing all the test data, train data and k-value
# and classifies the test data into nearest labeled class
def predict_classification(train_data, test_data, num_neighbors):
    k_neighbors = nearest_neighbors(train_data, test_data, num_neighbors)
    result = [row[-1] for row in k_neighbors]
    prediction = max(set(result), key=result.count)
    return prediction


# driver function
if __name__ == '__main__':
    dataset = [[0, 1, 0, "ClassA"], [0, 1, 1, "ClassA"], [1, 2, 1, "ClassA"], [1, 2, 0, "ClassA"],
               [1, 2, 2, "ClassB"], [2, 2, 2, "ClassB"], [1, 2, -1, "ClassB"], [2, 2, 3, "ClassB"],
               [-1, -1, -1, "ClassC"], [0, -1, -2, "ClassC"], [0, -1, 1, "ClassC"], [-1, -2, 1, "ClassC"]]
    test = [1, 0, 1]
    k_value1 = 1
    print("The given test data  with k value as 1 "
          "is classified into: ", predict_classification(dataset, test, k_value1))
    k_value2 = 2
    print("The given test data  with k value as 2 "
          "is classified into: ", predict_classification(dataset, test, k_value2))
    k_value3 = 3
    print("The given test data  with k value as 3 "
          "is classified into: ", predict_classification(dataset, test, k_value3))

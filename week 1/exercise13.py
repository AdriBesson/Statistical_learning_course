import os
import numpy as np
import utils as utils
import matplotlib.pyplot as plt

# File used in the exercise
class_train_file = os.path.join(os.getcwd(), 'data', 'simclass1_train.csv')
class_test_file = os.path.join(os.getcwd(), 'data', 'simclass1_test.csv')

# Load the csv files
class_train = np.genfromtxt(class_train_file, delimiter=';')
class_test = np.genfromtxt(class_test_file, delimiter=';')
label_train = class_train[1:,0]
x_train = class_train[1:,1]
y_train = class_train[1:,2]
label_test = class_test[1:,0]
x_test = class_test[1:,1]
y_test = class_test[1:,2]

# Neighbour sizes
list_neighbour_size = range(1,30)

# Output variables
label_nn_train = []
label_nn_test = []
train_loss = []
test_loss = []

# Local variable used in the loop
it = 0

# KNN estimation for all the neighbour sizes
for neighbour_size in list_neighbour_size:
    label_nn_train.append(utils.knn_estimate(x_train, x_train, y_train, neighbour_size=neighbour_size))
    label_nn_test.append(utils.knn_estimate(x_test, x_test, y_test, neighbour_size=neighbour_size))
    train_loss.append(utils.compute_classification_error(label_nn_train[it], label_train))
    test_loss.append(utils.compute_classification_error(label_nn_test[it], label_test))
    it+=1

# Plot train and test errors
plt.figure()
plt.subplot(121)
plt.plot(list_neighbour_size, train_loss)
plt.ylabel('Loss')
plt.xlabel('Number of neighbours')
plt.title('Train loss vs. Nb. of neighbours')
plt.subplot(122)
plt.plot(list_neighbour_size, test_loss)
plt.ylabel('Loss')
plt.xlabel('Number of neighbours')
plt.title('Test loss vs. Nb. of neighbours')
plt.show()

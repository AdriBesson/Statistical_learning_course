import os
import numpy as np
import utils as utils
import matplotlib.pyplot as plt

# File used in the exercise
reg_train_file = os.path.join(os.getcwd(), 'data', 'simreg1_train.csv')
reg_test_file = os.path.join(os.getcwd(), 'data', 'simreg1_test.csv')

# Load the csv files
reg_train = np.genfromtxt(reg_train_file, delimiter=';')
reg_test = np.genfromtxt(reg_test_file, delimiter=';')
x_train = reg_train[1:,0]
y_train = reg_train[1:,1]
x_test = reg_test[1:,0]
y_test = reg_test[1:,1]

# Output variables
y_train_pred = []
y_test_pred = []
x_train_reg_list = [np.ones(np.shape(x_train))]
x_test_reg_list = [np.ones(np.shape(x_test))]
train_loss = []
test_loss = []

# List of model orders
list_model_orders = range(1,10)

# Loop over the orders of the regrssion model
for k in list_model_orders:
    # Build the regression model
    x_train_reg_list.append(x_train**k)
    x_test_reg_list.append(x_test**k)
    x_train_reg = np.asarray(x_train_reg_list)
    x_test_reg = np.asarray(x_test_reg_list)

    # Compute the LS estimate
    beta_train = utils.compute_my_ls_estimate(np.transpose(x_train_reg), y_train)
    beta_test = utils.compute_my_ls_estimate(np.transpose(x_test_reg), y_test)

    # Compute the prediction
    y_train_pred.append(np.matmul(np.transpose(x_train_reg),beta_train))
    y_test_pred.append(np.matmul(np.transpose(x_test_reg),beta_test))

    # Calculate the training error
    train_loss.append(utils.compute_l2_error(y_train, y_train_pred[k-1]))

    # Empirical test error on the test set
    test_loss.append(utils.compute_l2_error(y_test, y_test_pred[k-1]))

# Plots
plt.figure()
plt.subplot(221)
plt.scatter(x_train,y_train, color='orange', marker='.', linewidth=1, label='data')
plt.plot(x_train, y_train_pred[0], color='blue', label='linear model')
plt.title('Linear fitting - Training loss = {:.2f}'.format(train_loss[0]))
plt.legend(loc='upper left', frameon=False)
plt.ylabel('y')
plt.xlabel('x')

plt.subplot(222)
plt.scatter(x_test,y_test, color='orange', marker='.', linewidth=1, label='data')
plt.plot(x_test, y_test_pred[0], color='blue', label='linear model')
plt.title('Linear fitting - Test loss = {:.2f}'.format(test_loss[0]))
plt.legend(loc='upper left', frameon=False)
plt.ylabel('y')
plt.xlabel('x')

plt.subplot(223)
plt.plot(list_model_orders, train_loss)
plt.ylabel('Loss')
plt.xlabel('Model order')
plt.title('Train loss vs. model order')

plt.subplot(224)
plt.plot(list_model_orders, test_loss)
plt.ylabel('Loss')
plt.xlabel('Model order')
plt.title('Test loss vs. model order')

plt.show()

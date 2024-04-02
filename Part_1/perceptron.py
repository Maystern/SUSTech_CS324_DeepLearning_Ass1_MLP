import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    
    @staticmethod
    def sign(x):
        return np.where(x >= 0, 1, -1)

    def __init__(self, n_inputs, max_epochs=10, learning_rate=1e-3):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias), shape = (1, n_inputs + 1).
        """
        self.n_inputs = n_inputs;
        self.max_epochs = max_epochs;
        self.learning_rate = learning_rate
        self.weights = np.zeros((1, n_inputs + 1))
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
            shape = (batch_size, feature)
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        result = self.sign(np.dot(self.weights, np.transpose(input_vec)))
        return result
        
        
    def train(self, train_inputs, train_labels, test_inputs, test_labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        
        self.train_accs = []
        self.test_accs = []
        self.train_losses = []
        self.test_losses = []
        
        
        for _ in range(self.max_epochs): 
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            print(f"---- epoch#{_} ----")
            tmp = np.vstack(train_inputs)
            new_column = np.ones((tmp.shape[0], 1))
            tmp = np.hstack((tmp, new_column))
            output = self.forward(tmp)
            diff = np.where(output != train_labels)[1]
            if len(diff) == 0:
                train_acc = np.sum((output == train_labels)[0]) / len(train_labels)
                train_loss = np.sum(tmp[diff, :])

                tmp = np.vstack(test_inputs)
                new_column = np.ones((tmp.shape[0], 1))
                tmp = np.hstack((tmp, new_column))
                output = self.forward(tmp)
                diff = np.where(output != test_labels)[1]
                test_acc = np.sum((output == test_labels)[0]) / len(test_labels)
                test_loss = np.sum(tmp[diff, :])
                print(f"train_acc: {train_acc}, test_acc: {test_acc}")
                print(f"train_loss: {train_loss}, test_loss: {test_loss}")
                self.train_accs.append(train_acc)
                self.test_accs.append(test_acc)
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)
                break
            train_acc = np.sum((output == train_labels)[0]) / len(train_labels)
            train_loss = np.sum(tmp[diff, :])
            grad = -np.sum(np.expand_dims(train_labels[diff], axis=1) * tmp[diff, :], axis=0)
            self.weights = self.weights - self.learning_rate * grad
            
            tmp = np.vstack(test_inputs)
            new_column = np.ones((tmp.shape[0], 1))
            tmp = np.hstack((tmp, new_column))
            output = self.forward(tmp)
            diff = np.where(output != test_labels)[1]
            test_acc = np.sum((output == test_labels)[0]) / len(test_labels)
            test_loss = np.sum(tmp[diff, :])
            
            print(f"train_acc: {train_acc}, test_acc: {test_acc}")
            print(f"train_loss: {train_loss}, test_loss: {test_loss}")
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)


"""
Generate a dataset of points in R2. To do this, define two Gaussian distributions and sample 100 points from each.
Your dataset should then contain a total of 200 points, 100 from each distribution. Keep 80 points per distribution
as the training (160 in total), 20 for the test (40 in total).
"""

# Generate a dataset of points in R2
np.random.seed(0)
# Define two Gaussian distributions
# mean1 = [0, 0]
# mean1 = [5, 5]; name = "55"
# mean1 = [4, 4]; name = "44"
# mean1 = [3, 3]; name = "33"
# mean1 = [2, 2]; name = "22"
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]; name = "conv_1_300"
# cov1 = [[0.5, 0], [0, 0.5]]; name = "conv_0.5"
# cov1 = [[2, 0], [0, 2]]; name = "conv_2"
mean2 = [2, 2]
cov2 = [[1, 0], [0, 1]]
# Sample 100 points from each
x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
# Keep 80 points per distribution as the training (160 in total)
training_inputs = np.vstack((np.hstack((x1[:80], x2[:80])), np.hstack((y1[:80], y2[:80])))).T
labels = np.hstack((np.ones(80), -np.ones(80)))
# 20 for the test (40 in total)
test_inputs = np.vstack((np.hstack((x1[80:], x2[80:])), np.hstack((y1[80:], y2[80:])))).T
test_labels = np.hstack((np.ones(20), -np.ones(20)))

# 可视化


plt.scatter(x1[:80], y1[:80], c='b', marker='o', label='train_distribution1')
plt.scatter(x1[80:], y1[80:], c='b', marker='x', label='test_distribution1')
plt.scatter(x2[:80], y2[:80], c='g', marker='o', label='train_distribution2')
plt.scatter(x2[80:], y2[80:], c='g', marker='x', label='test_distribution2')

# Initialize the perceptron
p = Perceptron(n_inputs=2, max_epochs=500, learning_rate=0.01)
# Train the perceptron
p.train(training_inputs, labels, test_inputs, test_labels)
# 绘制 decision boundary
weight = p.weights[0]
x = np.linspace(0, 6, 100)
y = -weight[0] / weight[1] * x - weight[2] / weight[1]
plt.plot(x, y, 'r--', label='decision boundary')

plt.legend(loc='upper left')
plt.savefig("decision_boundary_" + name + ".png")


# Plot the training and test accuracy
plt.clf()
plt.plot(p.train_accs, label='train_acc')
plt.plot(p.test_accs, label='test_acc')
plt.title('Perceptron accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.savefig("acc_" + name + ".png")


# plot the training and test loss
plt.clf()
plt.plot(p.train_losses, label='train_loss')
plt.plot(p.test_losses, label='test_loss')
plt.title('Perceptron loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.savefig("loss_" + name + ".png")


min_loss = min(p.test_losses)
max_acc = max(p.test_accs)
print(max_acc, min_loss)
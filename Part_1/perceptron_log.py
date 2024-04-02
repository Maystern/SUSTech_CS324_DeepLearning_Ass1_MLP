import numpy as np

class Perceptron(object):
    
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def __init__(self, n_inputs, max_epochs=10, learning_rate=1e-3, batch_size = 1):
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
        self.batch_size = batch_size
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
            shape = (batch_size, feature)
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        new_column = np.ones((input_vec.shape[0], 1))
        new_input_vec = np.hstack((input_vec, new_column))
        result = self.sigmoid(np.dot(self.weights, np.transpose(new_input_vec)))
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
            train_acc = 0
            index = np.arange(len(train_inputs))
            np.random.shuffle(index)
            train_inputs = train_inputs[index]
            train_labels = train_labels[index]
            for i in range(0, len(train_inputs), self.batch_size):
                if i + self.batch_size > len(train_inputs):
                    input_vec = train_inputs[i:]
                    label = train_labels[i:]
                else:
                    input_vec = train_inputs[i:i+self.batch_size]
                    label = train_labels[i:i+self.batch_size]
                
                output = self.forward(input_vec)
                train_acc += np.sum((output > 0.5).astype(int) == label)
                new_column = np.ones((input_vec.shape[0], 1))
                new_input_vec = np.hstack((input_vec, new_column))
                tmp = -(label / output - (1 - label) / (1 - output)) * output * (1 - output) * np.transpose(new_input_vec)

                gradient = np.sum(tmp, axis=1) / input_vec.shape[0]
                self.weights -= self.learning_rate * gradient
                
            test_acc = 0
            for i in range(0, len(test_inputs), self.batch_size):
                if i + self.batch_size > len(test_inputs):
                    input_vec = test_inputs[i:]
                    label = test_labels[i:]
                else:
                    input_vec = test_inputs[i:i+self.batch_size]
                    label = test_labels[i:i+self.batch_size]
                output = self.forward(input_vec)
                test_acc += np.sum((output > 0.5).astype(int) == label)
        
            train_acc = train_acc / len(train_inputs)
            test_acc = test_acc / len(test_labels)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            print(f"epoch: {_}, train_acc: {train_acc}, test_acc: {test_acc}")
            
"""
Generate a dataset of points in R2. To do this, define two Gaussian distributions and sample 100 points from each.
Your dataset should then contain a total of 200 points, 100 from each distribution. Keep 80 points per distribution
as the training (160 in total), 20 for the test (40 in total).
"""

# Generate a dataset of points in R2
np.random.seed(0)
# Define two Gaussian distributions
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]
mean2 = [2, 2]
cov2 = [[1, 0], [0, 1]]
# Sample 100 points from each
x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
# Keep 80 points per distribution as the training (160 in total)
training_inputs = np.vstack((np.hstack((x1[:80], x2[:80])), np.hstack((y1[:80], y2[:80])))).T
labels = np.hstack((np.ones(80), np.zeros(80)))
# 20 for the test (40 in total)
test_inputs = np.vstack((np.hstack((x1[80:], x2[80:])), np.hstack((y1[80:], y2[80:])))).T
test_labels = np.hstack((np.ones(20), np.zeros(20)))

# 可视化
import matplotlib.pyplot as plt
plt.scatter(x1[:80], y1[:80], c='b', marker='o', label='train_distribution1')
plt.scatter(x1[80:], y1[80:], c='b', marker='x', label='test_distribution1')
plt.scatter(x2[:80], y2[:80], c='g', marker='o', label='train_distribution2')
plt.scatter(x2[80:], y2[80:], c='g', marker='x', label='test_distribution2')

# plt.savefig('distribution.png')

# Initialize the perceptron
p = Perceptron(n_inputs=2, max_epochs=500, learning_rate=0.1, batch_size=len(training_inputs))
# Train the perceptron
p.train(training_inputs, labels, test_inputs, test_labels)


weight = p.weights[0]
x = np.linspace(0, 5, 100)
y = -weight[0] / weight[1] * x - weight[2] / weight[1]
# 绘制虚线，红色
plt.plot(x, y, 'r--', label='decision boundary')
plt.legend(loc='lower right')
plt.savefig('decision_boundary_log.png')


# Plot the training and test accuracy
plt.clf()
plt.plot(p.train_accs, label='train_acc')
plt.plot(p.test_accs, label='test_acc')
plt.title('Perceptron accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.savefig('acc_log.png')

# Test the perceptron
# result = p.forward(test_inputs)
# print("result: ", result)
# print("test_labels: ", test_labels)
# acc = np.sum((result > 0.5).astype(int) == test_labels)
# print("acc: ", acc / len(test_labels))

# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.model_selection import GridSearchCV

# class PerceptronClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, n_inputs, max_epochs=10, learning_rate=1e-3, batch_size=1):
#         self.n_inputs = n_inputs
#         self.max_epochs = max_epochs
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.model = None

#     def fit(self, X, y):
#         self.model = Perceptron(self.n_inputs, self.max_epochs, self.learning_rate, self.batch_size)
#         self.model.train(X, y, X, y)
#         return self

#     def predict(self, X):
#         return (self.model.forward(X) > 0.5).astype(int)

# # 设置要搜索的超参数范围
# param_grid = {
#     'max_epochs': [10, 20, 30],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'batch_size': [1, 10, 20]
# }

# # 创建PerceptronClassifier对象
# perceptron = PerceptronClassifier(n_inputs=2)

# # 使用GridSearchCV搜索最佳超参数
# grid_search = GridSearchCV(estimator=perceptron, param_grid=param_grid, cv=3)
# grid_search.fit(training_inputs, labels)

# # 打印最佳超参数组合
# print("Best hyperparameters:", grid_search.best_params_)

# # 使用最佳模型进行预测
# best_model = grid_search.best_estimator_
# predictions = best_model.predict(test_inputs)
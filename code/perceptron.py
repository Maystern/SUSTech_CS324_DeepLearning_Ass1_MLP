import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    
    @staticmethod
    def sign(x):
        return np.where(x >= 0, 1, -1)

    def __init__(self, n_inputs, max_epochs=10, learning_rate=1e-3, print_flag = True):
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
        self.print_flag = print_flag
        
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
            if self.print_flag:
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
                if self.print_flag:
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

            if self.print_flag:
                print(f"train_acc: {train_acc}, test_acc: {test_acc}")
                print(f"train_loss: {train_loss}, test_loss: {test_loss}")

            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
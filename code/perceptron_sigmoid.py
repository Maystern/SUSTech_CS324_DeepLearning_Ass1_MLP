import numpy as np

class PerceptronSigmoid(object):
    
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def __init__(self, n_inputs, max_epochs=10, learning_rate=1e-3, batch_size = 1, print_flag = True):
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
            if self.print_flag:
                print(f"epoch: {_}, train_acc: {train_acc}, test_acc: {test_acc}")

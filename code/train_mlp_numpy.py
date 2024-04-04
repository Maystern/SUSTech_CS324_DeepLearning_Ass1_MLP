import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from modules import * 
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

seed = 0

# Default constants
# DNN_HIDDEN_UNITS_DEFAULT = '20'
# LEARNING_RATE_DEFAULT = 2e-2
# MAX_EPOCHS_DEFAULT = 35000 # adjust if you use batch or not
# EVAL_FREQ_DEFAULT = 10

DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

train_losses = []
train_accs = []
test_losses = []
test_accs = []
epochs = []
name = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    accuracy = np.mean(predicted_classes == true_classes) * 100
    return accuracy

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size=800, noise = 0.1, print_flag = True):
    global train_losses
    global train_accs
    global test_losses
    global test_accs
    global epochs

    train_losses.clear()
    train_accs.clear()
    test_losses.clear()
    test_accs.clear()
    epochs.clear()

    # writer = SummaryWriter()
    global name
    # Load your data here
    X, y = make_moons(n_samples=1000, noise=noise, random_state=seed)
    # X, y = make_moons(n_samples=1000, noise=0.1, random_state=seed); name = "_sigma_0.1_"
    # X, y = make_moons(n_samples=1000, noise=0.05, random_state=seed); name = "_sigma_0.05_"
    
    
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_labels = np.eye(2)[train_labels]
    test_labels = np.eye(2)[test_labels]
    
    # Initialize your MLP model and loss function
    hidden_units = [int(x) for x in dnn_hidden_units.split(',')]
    mlp = MLP(n_inputs=train_data.shape[1], n_hidden=hidden_units, n_classes=train_labels.shape[1])
    loss_fn = CrossEntropy()

    for step in range(max_steps):
        # Shuffle the training data and labels
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        # Mini-batch training
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Forward pass
            predictions = mlp.forward(batch_data)

            # Compute loss
            loss = loss_fn.forward(predictions, batch_labels)

            # Backward pass (compute gradients)
            dout = loss_fn.backward(predictions, batch_labels)
            mlp.backward(dout)

            # Update weights
            for layer in mlp.layers:
                if isinstance(layer, Linear):
                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']

        # Evaluate the model on the test set
        if step % eval_freq == 0 or step == max_steps - 1:
            train_predictions = mlp.forward(train_data)
            train_loss = loss_fn.forward(train_predictions, train_labels)
            train_accuracy = accuracy(train_predictions, train_labels)

            test_predictions = mlp.forward(test_data)
            test_loss = loss_fn.forward(test_predictions, test_labels)
            test_accuracy = accuracy(test_predictions, test_labels)


            # writer.add_scalar('Loss/train', train_loss, step)
            # writer.add_scalar('Accuracy/train', train_accuracy, step)
            # writer.add_scalar('Loss/test', test_loss, step)
            # writer.add_scalar('Accuracy/test', test_accuracy, step)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_accuracy)
            test_accs.append(test_accuracy)

            epochs.append(step)
            
            if print_flag:
                print(f"Step: {step}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    print(min(test_losses), max(test_accs))
    print("Training complete!")
    # writer.close()

def plot_acc():
    return epochs, train_accs, test_accs

def plot_loss():
    return epochs, train_losses, test_losses



def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS = parser.parse_known_args()[0]

    bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 800]

    for b in bs:
        print(b)
        train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, b)

    

if __name__ == '__main__':
    # random.seed(seed)
    np.random.seed(seed)
    main()

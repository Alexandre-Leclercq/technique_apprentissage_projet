import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# We don't follow the PEP8 convention with the variables X_train and X_test as we wanted to use an uppercase
# with the X to remember that it is a matrix.


class CNN:
    DATA_TYPE = torch.float32

    def __init__(self, num_epochs):

        self._device = torch.device('cpu')
        if torch.backends.cuda.is_built():
            self._device = torch.device('cuda')

        self.loss_function = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.model = None
        self._optimizer = None

    def set_model(self, num_channels: int, img_size: int, output_size: int):
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=img_size, kernel_size=(5,5), padding=1, dtype=self.DATA_TYPE),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=img_size, out_channels=img_size, kernel_size=(5, 5), padding=1, dtype=self.DATA_TYPE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Flatten(),

            nn.Linear(6272, 512, dtype=self.DATA_TYPE),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, output_size, bias=True, dtype=self.DATA_TYPE),
        )
        self.model.to(device=self._device)

    def reset_model_params(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def set_optimizer(self, optim='SGD', lr=1e-3, reg=1e-4):
        if self.model is None:
            raise ValueError('the model is not defined')
        if optim == 'SGD':
            self._optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.5, weight_decay=reg)
        elif optim == 'ADAM':
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)

    def training(self, train_loader, test_loader, verbose=True, tol=1e-4):
        if self._optimizer is None:
            raise ValueError('optimizer is not defined')

        self.reset_model_params()

        accuracy_train = []
        loss_train = []
        accuracy_test = []
        loss_test = []
        for epoch in range(self.num_epochs):
            accuracy = 0
            loss_epoch = 0

            # in one epoch we iterate on all the data
            for (x_sample, t_sample) in train_loader:
                y = self.model(x_sample)
                loss = self.loss_function(y, t_sample)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                loss_epoch += loss.item()
                accuracy += (torch.argmax(y) == torch.argmax(t_sample)).item()
            accuracy = accuracy / len(train_loader)
            loss_train.append(loss_epoch / len(train_loader))
            accuracy_train.append(accuracy)

            loss_test.append(0)
            accuracy_test.append(0)
            for (x_sample, t_sample) in test_loader:
                y_test = self.model(x_sample)
                loss = self.loss_function(y_test, t_sample)
                loss_test[-1] += loss.item()
                accuracy_test[-1] += (torch.argmax(y_test) == torch.argmax(t_sample)).item()
            loss_test[-1] = loss_test[-1] / len(test_loader)
            accuracy_test[-1] = accuracy_test[-1] / len(test_loader)

            if verbose:
                print("Epoch {}/{}, Loss: {:.5f}, Accuracy: {:.2f}%".format(epoch + 1, self.num_epochs, loss_train[-1],
                                                                            accuracy_train[-1] * 100))

            # End the training if the algorithm has converged sufficiently. It is the tolerance of the algorithm.
            if len(loss_train) >= 2 and abs(loss_train[-1] - loss_train[-2]) <= tol:
                break

        return loss_train, accuracy_train, loss_test, accuracy_test


def plot_training(loss_train, accuracy_train, loss_test, accuracy_test):
    """
    Plot the training plot for the data_train and data_test. The function is based on the plot_curves function
    used in the TP4 of IFT712.
    """
    xdata = np.arange(1, len(loss_train) + 1)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel('Loss')
    plt.plot(xdata, loss_train, label='training')
    plt.plot(xdata, loss_test, label='validation')
    plt.xticks(xdata)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylabel('Accuracy')
    plt.plot(xdata, accuracy_train, label='training')
    plt.plot(xdata, accuracy_test, label='validation')
    plt.xticks(xdata)
    plt.legend()
    plt.show(block=False)
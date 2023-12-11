import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils.customDataset import CustomDataset
from torch.utils.data import DataLoader

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

    def set_model(self, output_size: int, activation_function='relu'):
        if activation_function == 'relu':
            activ = nn.ReLU()
        elif activation_function == 'prelu':
            activ = nn.PReLU()
        else:
            raise ValueError('Enter a valid activation function. \'relu\' or \'prelu\'.')

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            activ,

            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            activ,

            nn.Flatten(),

            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            activ,
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            activ,
            nn.Linear(512, output_size)
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

    def training(self, train_loader, test_loader, verbose=True, tol=5e-4):
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

    def k_fold_cross_validation(self, X_train_filenames, t_train, img_size: int, optim='SGD'):
        lr_choices = [1e-4, 1e-3, 1e-2]
        reg_choices = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        K = 5
        kf = KFold(n_splits=K, shuffle=True)

        best_accu = -1
        best_params = None
        accuracy_historic = np.zeros((len(lr_choices), len(reg_choices)))

        for i, lr in enumerate(lr_choices):
            for j, reg in enumerate(reg_choices):
                print("testing parameters: learning_rate = {:.0e},    regularization = {:.0e}".format(lr, reg))
                params = (lr, reg)
                val_accu = 0

                for k, (k_train_indice, k_val_indice) in enumerate(kf.split(X_train_filenames)):
                    dataset_train = CustomDataset('../data/images', filenames=X_train_filenames[k_train_indice],
                                                  targets=t_train[k_train_indice], img_size=img_size)
                    dataset_validation = CustomDataset('../data/images', filenames=X_train_filenames[k_val_indice],
                                                  targets=t_train[k_val_indice], img_size=img_size)

                    train_dataloader = DataLoader(dataset_train, batch_size=1)
                    validation_dataloader = DataLoader(dataset_validation, batch_size=1)
                    self.set_optimizer(optim=optim, lr=lr, reg=reg)
                    curves = self.training(train_dataloader, validation_dataloader, verbose=False)
                    _, _, _, accuracy_validation = curves
                    print('K = {}, accuracy: {:.3f}'.format(k, accuracy_validation[-1]))
                    val_accu += accuracy_validation[-1]

                val_accu = val_accu / K
                accuracy_historic[i][j] = val_accu
                if val_accu > best_accu:
                    print('Best val accuracy: {:.3f} | lr: {:.0e} | l2_reg: {:.0e}'.format(val_accu, lr, reg))
                    best_accu = val_accu
                    best_params = params
        return best_params, accuracy_historic


def plot_training(loss_train, accuracy_train, loss_test, accuracy_test, label_test_val: str):
    """
    Plot the training plot for the data_train and data_test. The function is based on the plot_curves function
    used in the TP4 of IFT712.
    The label_set_val allow us to change the label regarding if we provide data_test or data_validation
    """
    xdata = np.arange(1, len(loss_train) + 1)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel('Loss')
    plt.plot(xdata, loss_train, label='training')
    plt.plot(xdata, loss_test, label=label_test_val)
    plt.xticks(xdata)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylabel('Accuracy')
    plt.plot(xdata, accuracy_train, label='training')
    plt.plot(xdata, accuracy_test, label=label_test_val)
    plt.xticks(xdata)
    plt.legend()
    plt.show(block=False)
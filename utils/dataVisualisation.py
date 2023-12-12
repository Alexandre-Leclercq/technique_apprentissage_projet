import matplotlib.pyplot as plt
import numpy as np


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, axis):
    # grid_param_1 is the first parameter to vary
    # grid_param_2 is the second parameter to vary

    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    if axis == 0:
        for idx, val in enumerate(grid_param_1):
            ax.semilogx(grid_param_2, scores_mean[:, idx], '-o', label=name_param_1 + ': ' + str(val))
            ax.set_xlabel(name_param_2, fontsize=10)
    elif axis == 1:
        for idx, val in enumerate(grid_param_2):
            ax.semilogx(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))
            ax.set_xlabel(name_param_1, fontsize=10)
    else:
        print("unknown axis")

    ax.set_title("Grid Search Accuracies", fontsize=10, fontweight='bold')
    ax.set_ylabel('CV Average Accuracy', fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    ax.grid('on')


def plot_grid_search_single_param(cv_results, grid_param, name_param, figsize=(10, 6)):
    # Get Test Scores Mean for each grid search
    scores_mean = cv_results['mean_test_score']

    # Plot the 2D curve
    plt.figure(figsize=figsize)
    plt.plot(grid_param, scores_mean, '-o')

    # Add labels and a title
    plt.xlabel(name_param)
    plt.ylabel('CV Average Accuracy')
    plt.title("Grid Search Accuracies", fontsize=12, fontweight='bold')

    plt.grid(True)
    plt.show()


def plot_margin1_margin2(X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel('Feature Margin1')
    plt.ylabel('Feature Margin2')
    plt.show()


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

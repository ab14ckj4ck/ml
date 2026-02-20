from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train
    #       Transform X_train using the PCA object.
    #       Print the explained variance ratio of the PCA object.
    #       Return both the transformed data and the PCA object.

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    return X_train_pca, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons and hidden layers.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Train MLPClassifier with different number of layers/neurons.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.

    H = [(2, ), (8, ), (64, ), (128, ), (256, ), (1024, ), (128, 256, 128)]

    best_model_val = None
    best_model_train = None
    best_model = None

    for hidden_layer in H:
        network = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=100, solver='adam', random_state=1)
        network.fit(X_train, y_train)
        train_acc = network.score(X_train, y_train)
        val_acc = network.score(X_val, y_val)
        print(f'Hidden layer: {hidden_layer}, Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}')
        print(f'Training loss: {network.loss_:.4f}')

        if best_model_val is None:
            best_model_val = val_acc
            best_model_train = train_acc
            best_model = network

        elif val_acc > best_model_val:
            old_diff = abs(best_model_val - best_model_train)
            new_diff = abs(val_acc - train_acc)

            if new_diff <= old_diff + 0.1: # Allow for a small margin of error in overfitting
                best_model_val = val_acc
                best_model_train = train_acc
                best_model = network

    return best_model


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.

    H = [(2,), (8,), (64,), (128,), (256,), (1024,), (128, 256, 128)]

    best_model_val = None
    best_model_train = None
    best_model = None
    best_model_case = None

    for i in range(3):
        if (i == 0):
            print("\nalpha = 0.1")

        elif (i == 1):
            print("\nearly_stopping = True")

        elif (i == 2):
            print("\nalpha = 0.1, early_stopping = True")

        for hidden_layer in H:
            network = None
            if (i == 0):
                network = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=100, solver='adam', random_state=1, alpha=0.1)
            elif (i == 1):
                network = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=100, solver='adam', random_state=1, early_stopping=True)
            elif (i == 2):
                network = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=100, solver='adam', random_state=1, alpha=0.1, early_stopping=True)

            network.fit(X_train, y_train)
            train_acc = network.score(X_train, y_train)
            val_acc = network.score(X_val, y_val)
            print(f'Hidden layer: {hidden_layer}, Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}')
            print(f'Training loss: {network.loss_:.4f}')

            if best_model_val is None:
                best_model_val = val_acc
                best_model_train = train_acc
                best_model = network
                if (i == 0):
                    best_model_case = 'a'
                elif (i == 1):
                    best_model_case = 'b'
                elif (i == 2):
                    best_model_case = 'c'

            elif val_acc > best_model_val:
                old_diff = abs(best_model_val - best_model_train)
                new_diff = abs(val_acc - train_acc)

                if new_diff <= old_diff + 0.1:  # Allow for a small margin of error in overfitting
                    best_model_val = val_acc
                    best_model_train = train_acc
                    best_model = network
                    if (i == 0):
                        best_model_case = 'a'
                    elif (i == 1):
                        best_model_case = 'b'
                    elif (i == 2):
                        best_model_case = 'c'

    print(f'\nChosen model: Train: {best_model_train:.4f}, Validation: {best_model_val:.4f}, Loss: {best_model.loss_:.4f}, Best model case: {best_model_case}')

    return best_model

def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.
    plt.plot(nn.loss_curve_, color='red')
    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.

    predictions = nn.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    final_accuracy = nn.score(X_test, y_test)
    print(f'\nFinal accuracy: {final_accuracy:.4f}')
    print(f'\nClass_report:\n {class_report}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    param_dic = {
        'alpha' : [0.0, 0.1, 1.0],
        'batch_size' : [32, 512],
        'hidden_layer_sizes': [(128,), (256,)]
    }

    network = MLPClassifier(max_iter=100, solver='adam', random_state=42)

    grid = GridSearchCV(param_grid=param_dic, cv=5, verbose=4, estimator=network)
    grid.fit(X_train, y_train)

    print(f'Best mcv score: {grid.best_score_:.4f}')
    print(f'Best params: {grid.best_params_}')

    best_model = grid.best_estimator_

    return best_model
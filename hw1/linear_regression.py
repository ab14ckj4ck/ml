import numpy as np


def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    r"""
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    # TODO: Implement the univariate loss \mathcal{L}_U(theta) (as specified in Equation 1)

    b = float(theta[0])
    w = float(theta[1])

    y_guess = b + w * x
    errors = (y_guess - y) ** 2

    return float(np.mean(errors))


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """

    N = x.size
    assert N > 1, "There must be at least 2 points given!"
    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.1.1)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum_x = np.sum(x)
    sum_x_squared = np.sum(x * x)
    sum_x_y = np.sum(x * y)


    w = (sum_x_y - y_mean * sum_x) / (sum_x_squared - x_mean * sum_x)
    b = y_mean - w * x_mean
    return np.array([float(b), float(w)])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    r"""
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    # TODO: Implement Pearson correlation coefficient, as shown in Equation 3 (Task 1.1.2).

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sum_x_xm_y_ym = np.sum((x - x_mean) * (y - y_mean))
    sum_sqrt_x_xm_sq = np.sqrt(np.sum((x - x_mean) ** 2))
    sum_sqrt_y_ym_sq = np.sqrt(np.sum((y - y_mean) ** 2))

    pearson_r = float(sum_x_xm_y_ym / (sum_sqrt_x_xm_sq * sum_sqrt_y_ym_sq))
    return pearson_r


def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    r"""
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)
    # X  element of R^{N x (D + 1)}

    N = data.shape[0]
    bias_col = np.ones((N, 1))

    design_matrix = np.hstack((bias_col, data))
    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    r"""
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)
    predictions = X @ theta
    residuals = predictions - y
    mse = np.mean(residuals ** 2)
    return float(mse)


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.2.1). 
    # Note: Use the pinv function.

    X_transposed = X.T
    moore_pseudo_inverse = pinv(X_transposed @ X)
    X_transposed_y = X_transposed @ y

    theta = moore_pseudo_inverse @ X_transposed_y
    return theta


def compute_polynomial_design_matrix(x: np.ndarray, K: int) -> np.ndarray:
    r"""
    :param x: 1D array that represents the feature vector
    :param K: the degree of the polynomial
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the polynomial design matrix (Task 1.3.2)
    columns = [x ** d for d in range(K + 1)]
    polynomial_design_matrix = np.stack(columns, axis = 1)
    return polynomial_design_matrix


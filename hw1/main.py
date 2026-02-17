import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial, 
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations)
from linear_regression import (fit_univariate_lin_model, 
                               fit_multiple_lin_model, 
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import rastrigin, gradient_rastrigin, gradient_descent, finite_difference_gradient_approx


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, 
                    "duration": 4, "exercise_intensity": 5, 
                    "fitness_level": 6, "calories": 7}

    # After loading the data, you can for example access it like this: 
    # `smartwatch_data[:, column_to_id['hours_sleep']]`
    smartwatch_data = np.load('data/smartwatch_data.npy')

    # TODO: Implement Task 1.1.2: Find 3 pairs of features that have a linear relationship.
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found, 
    # the MSE, and plot the data points together with the linear function.
    # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.
    feature_pairs = [
        ['duration', 'calories'],
        ['duration', 'fitness_level'],
        ['fitness_level', 'calories'],
        ['hours_sleep', 'avg_pulse'],
        ['max_pulse', 'duration'],
        ['max_pulse', 'fitness_level']
    ]

    feature_values = []

    np.set_printoptions(suppress=True)

    if not use_linalg_formulation:
        for pair in feature_pairs:
            x = smartwatch_data[:, column_to_id[pair[0]]]
            y = smartwatch_data[:, column_to_id[pair[1]]]

            theta = fit_univariate_lin_model(x, y)
            loss = univariate_loss(x, y, theta)
            pearson = calculate_pearson_correlation(x, y)

            entry = (x, y, theta, loss, pearson, pair[0], pair[1])

            feature_values.append(entry)

        print("\nPairs:")
        for pair in feature_values:
            x, y, theta, loss, pearson, xlabel, ylabel = pair
            plot_scatterplot_and_line(x, y, theta, xlabel, ylabel, f'{xlabel} and {ylabel}', f'univariate_{xlabel}_and_{ylabel}')

            theta_str = f"[{theta[0]}, {theta[1]}]"
            print(f'Pair: {xlabel} and {ylabel} | θ*: {theta_str}, Loss (MSE): {loss}, Pearson r: {pearson}')

        feature_values.sort(key=lambda sorter: abs(sorter[4]))

        print('\nPairs without linear relationship:')
        for i in range(3):
            print(f'{feature_values[i][5]} and {feature_values[i][6]}: {feature_values[i][4]}')

        last_three = feature_values[-3:]
        print('\nPairs with linear relationship:')
        for pair in last_three:
            print(f'{pair[5]} and {pair[6]}: {pair[4]}')

        print('\n\n End of Task 1.1.x \n\n')



    # TODO: Implement Task 1.2.3: Multiple linear regression
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.

    else:
        y = smartwatch_data[:, column_to_id['calories']]

        features = ['duration', 'avg_pulse', 'max_pulse']

        X_raw = smartwatch_data[:, [column_to_id[f] for f in features]]

        X = compute_design_matrix(X_raw)

        theta_M = fit_multiple_lin_model(X, y)
        loss_M = multiple_loss(X, y, theta_M)

        print(f'\nMultiple Linear Regression:')
        print(f'Features: {features}')
        print(f'θ*: {theta_M}')
        print(f'Loss (MSE): {loss_M}')

        print("\n\nf_theta_M(x) = ", end=f"{theta_M[0]}")
        for i in range(1, len(theta_M)):
            print(f" + {theta_M[i]}·{column_to_id[features[i - 1]]}", end="")



    #Test run for Task 1.2.2
    x = smartwatch_data[:, column_to_id['duration']]
    y = smartwatch_data[:, column_to_id['calories']]

    X_raw = x.reshape(-1, 1)

    X = compute_design_matrix(X_raw)

    theta = fit_multiple_lin_model(X, y)
    loss = univariate_loss(x, y, theta)

    print('\nTest run for Task 1.2.2:')
    print(f'duration and calories θ*: {theta} Loss (MSE): {loss}')

    print('\n\n End of Task 1.2.x \n\n')


    # TODO: Implement Task 1.3.2: Polynomial regression
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K, 
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.
    feature_values = [
        'duration', 'calories'
    ]

    x = smartwatch_data[:, column_to_id[feature_values[0]]]
    y = smartwatch_data[:, column_to_id[feature_values[1]]]

    K = 3 #as K = len(theta) - 1 -> bias term + x¹ * w + x² * w + x³ * w + ... * x^k * w
    X_poly = compute_polynomial_design_matrix(x, K)

    theta = fit_multiple_lin_model(X_poly, y)
    loss = multiple_loss(X_poly, y, theta)

    theta_uni = fit_univariate_lin_model(x, y)
    loss_uni = univariate_loss(x, y, theta_uni)

    print(f'\nPolynomial Regression:')
    print(f'Features: {feature_values} with K={K}')
    print(f'θ*: {theta}')
    print(f'Loss (MSE): {loss}')

    print("\nComparison with Univariate Linear Regression:")
    print(f'θ*: {theta_uni}')
    print(f'Loss (MSE): {loss_uni}')

    plot_scatterplot_and_polynomial(x, y, theta, feature_values[0], feature_values[1], f'{feature_values[0]} and {feature_values[1]}', f'polynomial_{feature_values[0]}_and_{feature_values[1]}')

    print('\n\n End of Task 1.3.2 \n\n')


    # TODO: Implement Task 1.3.3: Use x_small and y_small to fit a polynomial model.
    # Find and report the smallest K that gets zero loss. Plot the data points and the polynomial function.
    x_small = smartwatch_data[:5, column_to_id['duration']]
    y_small = smartwatch_data[:5, column_to_id['calories']]

    print('\nTask 1.3.3 -- Overfitting --:')

    K = len(x_small) - 1
    X_Design = compute_polynomial_design_matrix(x_small, K)
    theta = np.linalg.solve(X_Design, y_small)

    loss = multiple_loss(X_Design, y_small, theta)

    np.set_printoptions(suppress=True)

    print(f'Found K as {K} for duration and calories')
    print(f'θ*: {theta}')
    print(f'Loss (MSE): {loss:.4f}')
    plot_scatterplot_and_polynomial(x_small, y_small, theta, 'duration', 'calories', '1.3.3 --Overfitting--', f'polynomial_duration_and_calories_small')

    print('\n\n End of Task 1.3.3 \n\n')



def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # TODO: Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-1.npy')
            create_design_matrix = create_design_matrix_dataset_1

        elif task == 2:
            # TODO: Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-2.npy')
            create_design_matrix = create_design_matrix_dataset_2

        elif task == 3:
            # TODO: Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-2-data.npy')
            y = np.load('data/targets-dataset-3.npy')
            create_design_matrix = create_design_matrix_dataset_3

        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # TODO: Split the dataset using the `train_test_split` function.
        # The parameter `random_state` should be set to 0.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # TODO: Fit the model to the data using the `fit` method of the classifier `clf`
        clf.fit(X_train, y_train)
        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test, y_test) # TODO: Use the `score` method of the classifier `clf` to calculate accuracy

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        yhat_train = clf.predict_proba(X_train) # TODO: Use the `predict_proba` method of the classifier `clf` to
                                                #  calculate the predicted probabilities on the training set
        yhat_test = clf.predict_proba(X_test)   # TODO: Use the `predict_proba` method of the classifier `clf` to
                                                #  calculate the predicted probabilities on the test set

        # TODO: Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
        loss_train, loss_test = log_loss(y_train, yhat_train[:, 1]), log_loss(y_test, yhat_test[:, 1])
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        # TODO: Print theta vector (and also the bias term). Hint: Check the attributes of the classifier
        classifier_weights, classifier_bias = clf.coef_, clf.intercept_
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(46)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    x0 = None
    y0 = None
    print(f'Starting point: {x0:.4f}, {y0:.4f}')

    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(rastrigin)
        plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0))

    # TODO: Check if gradient_rastrigin is correct at (x0, y0). 
    # To do this, print the true gradient and the numerical approximation.
    pass

    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.
    x_list, y_list, f_list = None, None, None

    # Print the point that is found after `num_iters` iterations
    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    print(f'Global optimum: f(0, 0)= {rastrigin(0, 0):.4f}')

    # Here we plot the contour of the function with the path taken by the gradient descent algorithm
    plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0), 
                    x_list=x_list, y_list=y_list)

    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    pass


def main():
    np.random.seed(46)

    task_1(use_linalg_formulation=True)
    task_2()
    #task_3(initial_plot=True)


if __name__ == '__main__':
    main()

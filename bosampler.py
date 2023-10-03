import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from scipy.stats import chi2, norm
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, KFold
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import copy
from sklearn.model_selection import GridSearchCV
import random
# Disable ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BOsampler:
    # Constructor method (initialization)
    def __init__(self, 
                 hyperparam_grid=None,
                 X=None,
                 y=None,
                 y_noise_params=None,
                 normalize_data=True,
                 cv_folds=5,
                 sample_size=5):
        self.hyperparam_grid = hyperparam_grid # dictionary of model hyperparameters, including kernels.
        self.y_noise_params = y_noise_params
        self.y_noise = None
        self.X = X # The input features of the data set provided for the sampler.
        self.y = y
        self.sample_size = sample_size
        if self.y_noise_params is not None:
            self.y_noise = np.random.normal(y_noise_params['mean'], y_noise_params['std'], size=self.y.shape)
            self.y = self.y + self.y_noise # Add noise to the target.
        self.scaler = None # The input feature scaler constructed with the X data.
        if normalize_data is True:
            self.initialize_scaler()
            self.normalize_x()
            self.cv_folds = cv_folds

    """
    Create a scaler for the X data.
    """
    def initialize_scaler(self):
        self.scaler = StandardScaler().fit(self.X)

    """
    Normalize the X data.
    """
    def normalize_x(self):
        self.X = self.scaler.transform(self.X)

    def inverted_lower_confidence_bound(y_mean=0, y_std=0, l=0.2):
        return l*y_std - y_mean
    
    def optimize_gpr_model(self, X=None, y=None):
        gpr = GaussianProcessRegressor()
        grid_search = GridSearchCV(gpr, self.hyperparam_grid, cv=self.cv_folds)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        return best_estimator
        #y_pred = best_estimator.predict(X)

    """
    Get and return a random sample from the X,y data.
    """
    def get_sample(self, sample_size=1):
        # Take a random sample without replacement
        index_list = np.arange(0, self.y.size)
        random_sample_indices = np.random.choice(index_list, sample_size, replace=False)
        random_sample_x = self.X[random_sample_indices, :]
        random_sample_y = self.y[random_sample_indices, 0]
        return random_sample_x, random_sample_y

    """
    Take a sample from the X,y data and fit a gpr to sample data.
    """
    def sample_optimize_gpr_model(self):
        random_sample_x, random_sample_y = self.get_sample(sample_size=self.sample_size)
        best_estimator = self.optimize_gpr_model(X=random_sample_x, y=random_sample_y)
        return best_estimator


    """
    Evaluate sample point utilities.
    """
    def evaluate_single_point_impact(self, x_data=None, y_data=None, nfolds=5):
        loo = LeaveOneOut()
        loo_ind = 1
        point_utility_list = np.zeros(shape=(len(y_data), 1))
        for inner_data_index, single_point_index in loo.split(x_data):
            sample_data_x, single_point_x = x_data[inner_data_index], x_data[single_point_index]
            sample_data_y, single_point_y = y_data[inner_data_index], y_data[single_point_index]
            kfcv = KFold(n_splits=nfolds, shuffle=True)
            print(f'Evaluating sample point {loo_ind}/{len(y_data)}')
            without_point_residual, with_point_residual = 0, 0
            for inner_train_index, inner_test_index in kfcv.split(sample_data_x):
                train_x, test_x = sample_data_x[inner_train_index], sample_data_x[inner_test_index]
                train_y, test_y = sample_data_y[inner_train_index], sample_data_y[inner_test_index]
                # Vertically stack the arrays
                train_x_with_new_point = np.vstack((train_x, single_point_x))
                train_y_with_new_point = np.vstack((train_y, single_point_y))
                model_without_point = self.optimize_gpr_model(X=train_x, y=train_y)
                model_with_point = self.optimize_gpr_model(X=train_x_with_new_point, y=train_y_with_new_point)
                predicted_y_without_point = model_without_point.predict(test_x)
                predicted_y_with_point = model_with_point.predict(test_x)
                without_point_residual += np.sum(np.abs(predicted_y_without_point - test_y)) / len(test_y)
                with_point_residual += np.sum(np.abs(predicted_y_with_point - test_y)) / len(test_y)
            without_point_residual = without_point_residual / nfolds
            with_point_residual = with_point_residual / nfolds
            point_utility_list[single_point_index] = with_point_residual - without_point_residual
            if point_utility_list[single_point_index] < 0:
                print(f'Point addition improved test error by: {point_utility_list[single_point_index]}')
            else:
                print(f'Point addition did not improve test error, but increased: {point_utility_list[single_point_index]}')
            loo_ind += 1
        return point_utility_list.reshape(-1, 1) # Return the utility values as 2D-matrix.
                

    def expected_improvement(x_data, surrogate_model, f_min, var_epsilon=1e-10):
        mu, sigma = surrogate_model.predict(X=x_data, return_std=True)  # Get mean and standard deviation from the surrogate model.
        # Calculate the standard score Z with a small epsilon to avoid division by zero.
        sigma[sigma == 0] = var_epsilon # In places where sigma is 0, set sigma to small value to prevent division by zero.
        Z = (f_min - mu) / sigma
        # Calculate the Expected Improvement.
        phi = norm.pdf(Z)
        Phi = norm.cdf(Z)
        ei = (f_min - mu) * Phi + sigma * phi
        return ei

    def scaled_expected_improvement(x_data, surrogate_model, f_min, var_epsilon=1e-10):
        mu, sigma = surrogate_model.predict(X=x_data, return_std=True)  # Get mean and standard deviation from the surrogate model.
        # Calculate the standard score Z with a small epsilon to avoid division by zero.
        sigma[sigma == 0] = var_epsilon # In places where sigma is 0, set sigma to small value to prevent division by zero.
        Z = (f_min - mu) / sigma
        # Calculate the Expected Improvement.
        phi = norm.pdf(Z)
        Phi = norm.cdf(Z)
        ei = (f_min - mu) * Phi + sigma * phi
        # Next EI variance
        ei2 = np.power(ei, 2)
        Z2 = np.power(Z, 2)
        sigma2 = np.power(sigma, 2)
        ei_std = np.sqrt(sigma2 * ((Z2 + 1)*Phi + Z*phi) - ei2)
        ei_std[ei_std == 0] = var_epsilon
        sei = ei / ei_std
        return sei

    def predictive_uncertainty(x_data, y_mean, y_std):
        max_std_ind = np.where(y_std == np.max(y_std))[0][0]
        return x_data[max_std_ind], y_mean[max_std_ind]


    """
    Minmax-normalization.
    """
    def acquisition_to_inclusion_probs(acquisition_values=None):
        max_acquisition_value = np.max(acquisition_values)
        min_acquisition_value = np.min(acquisition_values)
        divisor = max_acquisition_value - min_acquisition_value
        normalized_acquisition_values = None
        if divisor == 0: # If this happens, all acquisition values are same. In this case use random sampling with same prob. 1/N
            print(f'The maximum and minimum  acquisition values are equal, setting inc.probs to: {1.0/acquisition_values.size}')
            normalized_acquisition_values = np.full(shape=acquisition_values.shape, fill_value=1.0/acquisition_values.size)
        else: # divisor > 0
            normalized_acquisition_values = (acquisition_values - min_acquisition_value) / divisor # p € [0, 1]
            unique_acquisition_values = np.unique(normalized_acquisition_values)
            if unique_acquisition_values.size > 2:
                min_prob = unique_acquisition_values[0] + (unique_acquisition_values[1] - unique_acquisition_values[0]) / 2
                max_prob = unique_acquisition_values[-2] + (unique_acquisition_values[-1] - unique_acquisition_values[-2]) / 2
                normalized_acquisition_values[normalized_acquisition_values == 1] = max_prob
                normalized_acquisition_values[normalized_acquisition_values == 0] = min_prob # p € (0, 1)
            else: # This means there must be two unique values
                print(f'There are only two unique acquisition values, setting to 0.2 and 0.8.')
                normalized_acquisition_values[normalized_acquisition_values == 1] = 0.8
                normalized_acquisition_values[normalized_acquisition_values == 0] = 0.2 # p € (0, 1)
        return normalized_acquisition_values





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
        assert sample_size <= y.size and cv_folds <= y.size
        self.hyperparam_grid = hyperparam_grid # dictionary of model hyperparameters, including kernels.
        self.y_noise_params = y_noise_params
        self.y_noise = None
        self.X = self.check_2d_format(X) # The input features of the data set provided for the sampler.
        self.y = self.check_2d_format(y)
        self.sample_size = sample_size
        if self.y_noise_params is not None:
            self.y_noise = np.random.normal(y_noise_params['mean'], y_noise_params['std'], size=self.y.shape)
            self.y = self.y + self.check_2d_format(self.y_noise) # Add noise to the target.
        self.scaler = None # The input feature scaler constructed with the X data.
        if normalize_data is True:
            self.initialize_scaler()
            self.X = self.normalize_x(self.X)
        self.cv_folds = cv_folds
        self.model = None
        self.utility_model = None # The utility GPR model solved from utility values and sample x.
        self.utility_data_x = None
        self.utility_data_y = None # The estimates values from the utility function based on sample data.
        random_sample_x, random_sample_y = self.take_sample(sample_size=self.sample_size)
        self.sample_x = random_sample_x
        self.sample_y = random_sample_y



    """
    Container for making sure that we deal with 2D data. 
    """
    def check_2d_format(self, arr):
        arr = np.array(arr)
        # Check if the input is already 2D (n,m)
        if len(arr.shape) == 2:
            return arr  # Do nothing if it's already 2D
        # If it's 1D (n,), convert it to (n,1)
        elif len(arr.shape) == 1:
            return arr.reshape(-1, 1)

    """
    Create a scaler for the X data.
    """
    def initialize_scaler(self):
        self.scaler = StandardScaler().fit(self.X)

    """
    Normalize the X data.
    """
    def normalize_x(self, x):
        if self.scaler is not None:
            return self.scaler.transform(X=x)
        else:
            return None

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
    Fit GPR model to the sample data.
    """
    def fit_response_gpr_model(self):
        self.model = self.optimize_gpr_model(X=self.sample_x, y=self.sample_y)


    """
    Get and return a random sample from the X,y data and set them to object attributes immediately.
    """
    def set_sample(self, sample_size=1):
        # Take a random sample without replacement
        index_list = np.arange(0, self.y.size)
        random_sample_indices = np.random.choice(index_list, sample_size, replace=False)
        random_sample_x = self.check_2d_format(self.X[random_sample_indices, :])
        random_sample_y = self.check_2d_format(self.y[random_sample_indices, 0])
        self.sample_x = random_sample_x
        self.sample_y = random_sample_y
    

    """
    Get and return a random sample from the X,y data.
    """
    def take_sample(self, sample_size=1):
        # Take a random sample without replacement
        index_list = np.arange(0, self.y.size)
        random_sample_indices = np.random.choice(index_list, sample_size, replace=False)
        random_sample_x = self.check_2d_format(self.X[random_sample_indices, :])
        random_sample_y = self.check_2d_format(self.y[random_sample_indices, 0])
        return random_sample_x, random_sample_y

    """
    Calculate utility values and corresponding GPR model.
    """
    def estimate_utility_function(self):
        utility_values = self.evaluate_single_point_impact(x_data=self.sample_x, y_data=self.sample_y, nfolds=self.cv_folds)
        self.utility_data_x = self.sample_x
        self.utility_data_y = utility_values
        self.utility_model = self.optimize_gpr_model(X=self.utility_data_x, y=self.utility_data_y)


    """
    Evaluate sample point utilities.
    """
    def evaluate_single_point_impact(self, x_data=None, y_data=None, nfolds=5):
        x_data = self.check_2d_format(x_data)
        y_data = self.check_2d_format(y_data)
        loo = LeaveOneOut()
        #loo_ind = 1
        point_utility_list = np.zeros(shape=(len(y_data), 1))
        for loo_ind, (inner_data_index, single_point_index) in enumerate(loo.split(x_data)):
            sample_data_x, single_point_x = x_data[inner_data_index, :], x_data[single_point_index, :]
            sample_data_y, single_point_y = y_data[inner_data_index], y_data[single_point_index]
            kfcv = KFold(n_splits=nfolds, shuffle=True)
            without_point_residual, with_point_residual = 0, 0
            for inner_train_index, inner_test_index in kfcv.split(sample_data_x):
                train_x, test_x = sample_data_x[inner_train_index, :], sample_data_x[inner_test_index, :]
                train_y, test_y = sample_data_y[inner_train_index], sample_data_y[inner_test_index]
                # Vertically stack the arrays
                train_x_with_new_point = np.vstack((train_x, single_point_x))
                train_y_with_new_point = np.vstack((train_y, single_point_y))
                model_without_point = self.optimize_gpr_model(X=train_x, y=train_y)
                model_with_point = self.optimize_gpr_model(X=train_x_with_new_point, y=train_y_with_new_point)
                predicted_y_without_point = self.check_2d_format(model_without_point.predict(test_x))
                predicted_y_with_point = self.check_2d_format(model_with_point.predict(test_x))
                without_point_residual += np.mean(np.abs(predicted_y_without_point - test_y))
                with_point_residual += np.mean(np.abs(predicted_y_with_point - test_y))
            without_point_residual = without_point_residual / nfolds
            with_point_residual = with_point_residual / nfolds
            point_utility_list[single_point_index] = with_point_residual - without_point_residual
            if point_utility_list[single_point_index] < 0:
                print(f'Point {loo_ind+1}/{len(y_data)} addition improved residual by: {point_utility_list[single_point_index][0][0]}')
            else:
                print(f'Point {loo_ind+1}/{len(y_data)} addition did not improve residual, but increased: {point_utility_list[single_point_index][0][0]}')
        return self.check_2d_format(point_utility_list) # Return the utility values as 2D-matrix.
                

    def expected_improvement(self, x_data, surrogate_model, f_min, var_epsilon):
        mu, sigma = surrogate_model.predict(X=x_data, return_std=True)  # Get mean and standard deviation from the surrogate model.
        # Calculate the standard score Z with a small epsilon to avoid division by zero.
        sigma[sigma == 0] = var_epsilon # In places where sigma is 0, set sigma to small value to prevent division by zero.
        Z = (f_min - mu) / sigma
        # Calculate the Expected Improvement.
        phi = norm.pdf(Z)
        Phi = norm.cdf(Z)
        ei = (f_min - mu) * Phi + sigma * phi
        return ei

    def scaled_expected_improvement(self, x_data, surrogate_model, f_min, var_epsilon):
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
    def acquisition_to_inclusion_probs(self, acquisition_values=None):
        acquisition_values = self.check_2d_format(acquisition_values)
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
    
    
    def get_inclusion_probabilities(self, X=None, method='pu', l=0.2):
        X = self.check_2d_format(X)
        inclusion_probabilities = None
        # Check for normalization
        if self.scaler is not None: 
            X = self.scaler.transform(X)
        # Next step, we solve the acquisition values.
        if method == 'pu': # Predictive uncertainty
            #y_mean, y_std = self.model.predict(X=X, return_std=True)
            #y_mean, y_std = self.check_2d_format(y_mean), self.check_2d_format(y_std)
            _, y_std = self.model.predict(X=X, return_std=True)
            y_std = self.check_2d_format(y_std)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=y_std)
        elif method == 'ilcb':
            y_mean, y_std = self.utility_model.predict(X=X, return_std=True)
            ilcb_acquisition_values = self.inverted_lower_confidence_bound(y_mean=y_mean, y_std=y_std, l=l)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=ilcb_acquisition_values)
        elif method == 'ei':
            current_minimum_utility_value = np.min(self.utility_data_y)
            #print(f'Minimum utility value is: {min_utility_y}')
            ei_acquisition_values = self.expected_improvement(X, self.utility_model, current_minimum_utility_value, 1e-10)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=ei_acquisition_values)
        elif method == 'sei':
            current_minimum_utility_value = np.min(self.utility_data_y)
            sei_acquisition_values = self.scaled_expected_improvement(X, self.utility_model, current_minimum_utility_value, 1e-10)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=sei_acquisition_values)
        return inclusion_probabilities

    def sample_by_inclusion_probabilities(self, inclusion_probabilities=None, sample_count=1):
        # Your set of 10 numbers with selection probabilities
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        inclusion_probabilities = [0.1, 0.2, 0.05, 0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05]
        # Take a random sample of 5 numbers based on the probabilities
        sample = random.choices(numbers, weights=inclusion_probabilities, k=sample_count)






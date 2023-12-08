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
from utilities import calculate_histogram_distances, sort_by_x_values
import time
from utilities import fd_optimal_bins
# Disable ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BOsampler:
    # Constructor method (initialization)
    def __init__(self, 
                 hyperparam_grid=None, # The parameter grid for optimizing the GPR model.
                 X=None, # The input features, assumed to be of shape (n,m)
                 y=None, # The output data, assumed to be of shape (n,) or (n,1)
                 y_noise_params=None, # Noide parameter, if not-None, noise will be added to y.
                 normalize_data=True, # Should the input data be standardized?
                 cv_folds=5): # How many CV folds to use in performance evaluation.
        self.hyperparam_grid = hyperparam_grid # dictionary of model hyperparameters, including kernels.
        self.y_noise_params = y_noise_params
        self.y_noise = None
        self.X = self.check_2d_format(X) # The input features of the data set provided for the sampler.
        self.y = self.check_2d_format(y)
        if self.y_noise_params is not None:
            self.y_noise = np.random.normal(y_noise_params['mean'], y_noise_params['std'], size=self.y.shape)
            self.y = self.y + self.check_2d_format(self.y_noise) # Add noise to the target.
        self.scaler = None # The input feature scaler constructed with the X data.
        if normalize_data is True:
            self.initialize_scaler()
            self.X = self.normalize_x(self.X)
        self.cv_folds = cv_folds
        self.bins = fd_optimal_bins(self.y)
        print(f'Number of histogram bins automatically determined as: {self.bins}')

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
            return x

    def optimize_gpr_model(self, X=None, y=None):
        gpr = GaussianProcessRegressor()
        grid_search = GridSearchCV(gpr, self.hyperparam_grid, cv=self.cv_folds)
        grid_search.fit(X, y)
        best_estimator = grid_search.best_estimator_
        return best_estimator
    
    """
    Calculate utility values and corresponding GPR model.
    """
    def estimate_utility_function(self, X, y):
        utility_values = self.evaluate_single_point_impact(x_data=X, y_data=y, nfolds=self.cv_folds)
        utility_model = self.optimize_gpr_model(X=X, y=utility_values)
        return X, utility_values, utility_model

    """
    Evaluate sample point utilities using cross-validation.
    """
    def evaluate_single_point_impact(self, x_data=None, y_data=None, nfolds=5):
        x_data = self.check_2d_format(x_data)
        y_data = self.check_2d_format(y_data)
        loo = LeaveOneOut()
        point_utility_list = np.zeros(shape=(len(y_data), 1))
        for loo_ind, (inner_data_index, single_point_index) in enumerate(loo.split(x_data)):
            start_time = time.time()
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
            print(f'Data point {loo_ind+1}/{len(y_data)} utility evaluation took: {time.time()-start_time} seconds')
        return self.check_2d_format(point_utility_list) # Return the utility values as 2D-matrix.
                
    """
    The predictive uncertainty acquisition function.
    """
    def predictive_uncertainty(self, X=None, response_model=None):
        _, y_std = response_model.predict(X=X, return_std=True)
        return self.check_2d_format(y_std)

    """
    The inverse of LCB acquisition function.
    """
    def inverted_lower_confidence_bound(self, y_mean=0, y_std=0, l=0.2):
        return l*y_std - y_mean
    
    """
    The expected improvement (EI) acquisition function
    """
    def expected_improvement(self, x_data, surrogate_model, f_min, var_epsilon):
        mu, sigma = surrogate_model.predict(X=x_data, return_std=True)  # Get mean and standard deviation from the surrogate model.
        mu, sigma = self.check_2d_format(mu), self.check_2d_format(sigma)
        # Calculate the standard score Z with a small epsilon to avoid division by zero.
        sigma[sigma == 0] = var_epsilon # In places where sigma is 0, set sigma to small value to prevent division by zero.
        Z = (f_min - mu) / sigma
        # Calculate the Expected Improvement.
        phi = norm.pdf(Z)
        Phi = norm.cdf(Z)
        ei = (f_min - mu) * Phi + sigma * phi
        return ei

    """
    The scaled expected improvement (SEI) acquisition function
    """
    def scaled_expected_improvement(self, x_data, surrogate_model, f_min, var_epsilon):
        mu, sigma = surrogate_model.predict(X=x_data, return_std=True)  # Get mean and standard deviation from the surrogate model.
        mu, sigma = self.check_2d_format(mu), self.check_2d_format(sigma)
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
        # Make sure there are no 0 and 1 prob. normalized acquisition values.
        assert len(np.where(normalized_acquisition_values == 0)[0]) == 0
        assert len(np.where(normalized_acquisition_values == 1)[0]) == 0
        return normalized_acquisition_values
    
    
    def get_inclusion_probabilities(self, 
                                    X=None, # X-values for which acquisition values are requested
                                    method='pu', # Which acquisition type to use
                                    l=0.2, # Lambda parameter for ILCB
                                    response_model=None, # Response data model
                                    utility_model=None, # Utility model
                                    min_utility=None): # Minimum utility value
        X = self.check_2d_format(X)
        inclusion_probabilities = None
        # Next step, we solve the acquisition values.
        if method == 'pu': # Predictive uncertainty
            pu_acquisition_values = self.predictive_uncertainty(X=X, response_model=response_model)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=pu_acquisition_values)
        elif method == 'ilcb':
            y_mean, y_std = utility_model.predict(X=X, return_std=True)
            y_mean, y_std = self.check_2d_format(y_mean), self.check_2d_format(y_std)
            ilcb_acquisition_values = self.inverted_lower_confidence_bound(y_mean=y_mean, y_std=y_std, l=l)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=ilcb_acquisition_values)
        elif method == 'ei':
            #print(f'Minimum utility value is: {min_utility_y}')
            ei_acquisition_values = self.expected_improvement(X, utility_model, min_utility, 1e-10)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=ei_acquisition_values)
        elif method == 'sei':
            sei_acquisition_values = self.scaled_expected_improvement(X, utility_model, min_utility, 1e-10)
            inclusion_probabilities = self.acquisition_to_inclusion_probs(acquisition_values=sei_acquisition_values)
        # Make sure next that the probabilities sum to 1.
        inc_prob_sum = np.sum(inclusion_probabilities)
        norm_inc_probs = np.array([inc_prob/float(inc_prob_sum) for inc_prob in inclusion_probabilities])
        return self.check_2d_format(inclusion_probabilities), self.check_2d_format(norm_inc_probs)


    def sample_by_inclusion_probabilities(self, 
                                          population_X=None, 
                                          population_y=None, 
                                          inclusion_probabilities=None,
                                          normalized_inclusion_probabilities=None, 
                                          sample_count=1):
        population_X, population_y = self.check_2d_format(population_X), self.check_2d_format(population_y)
        # Create a list of indices representing the population
        index_list = list(range(population_y.size))
        # Take a sample without replacement
        if normalized_inclusion_probabilities is not None:
            normalized_inclusion_probabilities = np.reshape(normalized_inclusion_probabilities, newshape=(normalized_inclusion_probabilities.size,))
        sampled_indices = np.random.choice(a=index_list, size=sample_count, replace=False, p=normalized_inclusion_probabilities)
        # Remove sampled indices from the original set
        remaining_indices = [index for index in index_list if index not in sampled_indices]
        # Extract the sampled data based on the sampled indices
        sampled_X = population_X[sampled_indices, :]
        sampled_y = population_y[sampled_indices]
        sample_probabilities = None
        normalized_sample_probabilities = None
        if normalized_inclusion_probabilities is not None:
            normalized_sample_probabilities = normalized_inclusion_probabilities[sampled_indices]
            sample_probabilities = inclusion_probabilities[sampled_indices]
        # Update the original set by removing the sampled data
        remaining_population_X = population_X[remaining_indices, :]
        remaining_population_y = population_y[remaining_indices]
        return self.check_2d_format(sampled_X), self.check_2d_format(sampled_y), self.check_2d_format(remaining_population_X), self.check_2d_format(remaining_population_y), self.check_2d_format(sample_probabilities), self.check_2d_format(normalized_sample_probabilities)


    """
    Do data sampling using the BO-sampling and compare against SRS sampling.
    'sample_count' = How many samples to take per iteration
    'sampling_iterations' = How many times to perform the sampling
    'sampling_method_list' = What methods to use for sampling
    """
    def perform_sampling_comparison(self, 
                                    sample_count=1, 
                                    prior_sample_count=50,
                                    sampling_iterations=1, 
                                    sampling_method_list=['pu']):
        # Step 1: Get initial random sample from the whole population --> the prior data set.
        prior_X, prior_y, population_X, population_y, _, _ = self.sample_by_inclusion_probabilities(population_X=self.X, 
                                                                                              population_y=self.y, 
                                                                                              inclusion_probabilities=None, 
                                                                                              sample_count=prior_sample_count)
        sampling_data_container = {} # Dictionary to hold all sampling data.
        sampling_data_container['X'] = copy.deepcopy(self.X)
        sampling_data_container['y'] = copy.deepcopy(self.y)
        sampling_data_container['prior_X'] = copy.deepcopy(prior_X)
        sampling_data_container['prior_y'] = copy.deepcopy(prior_y)
        for sampling_method in sampling_method_list: # Initialize the container
            sample_data = {}
            sample_data['samples_X'] = []
            sample_data['samples_y'] = []
            sample_data['sample_inclusion_probabilities'] = []
            sample_data['normalized_sample_inclusion_probabilities'] = []
            sample_data['difference_estimator'] = []
            sample_data['KLD'] = []
            sample_data['population_X'] = copy.deepcopy(population_X)
            sample_data['population_y'] = copy.deepcopy(population_y)
            sample_data['mean_true_y'] = []
            sample_data['mean_estimated_y'] = []
            sample_data['MSE'] = []
            sampling_data_container[sampling_method] = sample_data
        fig, axs = None, None
        if self.X.shape[1] <= 1:
            fig, axs = plt.subplots(4, len(sampling_method_list), figsize=(25,18))
            plt.ion()
            plt.show()
        KL_list = []
        for sampling_iteration_idx in range(sampling_iterations): # How many times to perform the sampling
            for sampling_method_idx, sampling_method in enumerate(sampling_method_list): # Initialize the container
                start_time = time.time()
                print(f'Performing sampling {sampling_iteration_idx+1}/{sampling_iterations} for: {sampling_method}')
                # Build the current prior data. Add already sampled data to prior. 
                data_X, data_y = copy.deepcopy(sampling_data_container['prior_X']), copy.deepcopy(sampling_data_container['prior_y'])
                for subsample_X, subsample_y in zip(sampling_data_container[sampling_method]['samples_X'], sampling_data_container[sampling_method]['samples_y']):
                    data_X, data_y = np.vstack((data_X, subsample_X)), np.vstack((data_y, subsample_y))
                print(f'DEBUG: Model training data sizes are X: {data_X.shape}, y: {data_y.shape}')
                response_gpr_model = self.optimize_gpr_model(X=data_X, y=data_y)
                # Next, depending on the sampling method, we solve the inclusion probabilities and take the next sample.
                inclusion_probabilities, normalized_inclusion_probabilities = None, None
                if sampling_method == 'srs': # Same inclusion prob. to all. 
                    inclusion_probabilities = np.full(sampling_data_container[sampling_method]['population_y'].shape, 1.0/float(sampling_data_container[sampling_method]['population_y'].size))
                    normalized_inclusion_probabilities = inclusion_probabilities
                elif sampling_method == 'pu': # Predictive uncertainty
                    inclusion_probabilities, normalized_inclusion_probabilities = self.get_inclusion_probabilities(X=sampling_data_container[sampling_method]['population_X'], 
                                                                                method='pu',
                                                                                response_model=response_gpr_model)
                else: # This assumes ilcb, ei or sei
                    utility_X, utility_values, utility_model = self.estimate_utility_function(X=data_X, y=data_y)
                    inclusion_probabilities, normalized_inclusion_probabilities = self.get_inclusion_probabilities(X=sampling_data_container[sampling_method]['population_X'], 
                                                                               method=sampling_method,
                                                                               utility_model=utility_model,
                                                                               min_utility=np.min(utility_values))
                # Next, we take the sample and update the data containers accordingly
                assert sampling_data_container[sampling_method]['population_y'].size == inclusion_probabilities.size == normalized_inclusion_probabilities.size
                sample_X, sample_y, remaining_population_X, remaining_population_y, sample_probabilities, normalized_sample_probabilities = self.sample_by_inclusion_probabilities(
                                                                                                                                            population_X=sampling_data_container[sampling_method]['population_X'], 
                                                                                                                                            population_y=sampling_data_container[sampling_method]['population_y'], 
                                                                                                                                            inclusion_probabilities=inclusion_probabilities, 
                                                                                                                                            normalized_inclusion_probabilities=normalized_inclusion_probabilities,
                                                                                                                                            sample_count=sample_count)
                to_be_sampled_x = copy.deepcopy(sampling_data_container[sampling_method]['population_X']) # Save old X for plotting purpose
                # Update the sample data and remaining population
                sampling_data_container[sampling_method]['samples_X'].append(sample_X)
                sampling_data_container[sampling_method]['samples_y'].append(sample_y)
                sampling_data_container[sampling_method]['sample_inclusion_probabilities'].append(sample_probabilities)
                sampling_data_container[sampling_method]['normalized_sample_inclusion_probabilities'].append(normalized_sample_probabilities)
                sampling_data_container[sampling_method]['population_X'] = remaining_population_X
                sampling_data_container[sampling_method]['population_y'] = remaining_population_y
                # Now that the sample has been taken, we combine with current data and predict the rest of the population.
                train_X, train_y = np.vstack((data_X, sample_X)), np.vstack((data_y, sample_y))
                response_gpr_model_after_sample = self.optimize_gpr_model(X=train_X, y=train_y)
                estimated_remaining_y = self.check_2d_format(response_gpr_model_after_sample.predict(X=remaining_population_X))
                estimated_population_y = np.vstack((train_y, estimated_remaining_y))
                bins, densities, KL_divergence = calculate_histogram_distances(data_sources=[self.y, estimated_population_y], num_of_bins=self.bins)
                sampling_data_container[sampling_method]['KLD'].append(KL_divergence[1])
                sampling_data_container[sampling_method]['mean_true_y'].append(np.mean(self.y))
                sampling_data_container[sampling_method]['mean_estimated_y'].append(np.mean(estimated_population_y))
                sampling_data_container[sampling_method]['MSE'].append(np.mean((self.y - estimated_population_y) ** 2))
                assert data_y.size + sample_y.size + estimated_remaining_y.size == self.y.size # prior + sample + rest == all
                estimated_sample_y = self.check_2d_format(response_gpr_model_after_sample.predict(X=sample_X))
                difference_estimator = np.sum(data_y) + np.sum(estimated_sample_y) + np.sum(estimated_remaining_y) + np.sum((sample_y - estimated_sample_y) / sample_probabilities)
                sampling_data_container[sampling_method]['difference_estimator'].append(difference_estimator)
                print(f'Sampling iteration took: {time.time()-start_time} seconds.')
                
                """
                Dynamic visualization of the sampling
                """
                if self.X.shape[1] <= 1:
                    sorted_data_X, sorted_data_y = sort_by_x_values(data_X, data_y)
                    sorted_sample_data_X, sorted_sample_data_y = sort_by_x_values(sample_X, sample_y)
                    axs[0, sampling_method_idx].cla()
                    axs[0, sampling_method_idx].plot(self.X, self.y, color='blue', linestyle='--') # Plot of the original target function
                    
                    estimated_all_y, estimated_all_std = response_gpr_model.predict(X=self.X, return_std=True) # Get the function we estimated with current data
                    axs[0, sampling_method_idx].plot(self.X, estimated_all_y, label='GPR fit', color='black')
                    axs[0, sampling_method_idx].fill_between(self.X.ravel(), estimated_all_y.ravel() - 1.96*estimated_all_std.ravel(), estimated_all_y.ravel() + 1.96*estimated_all_std.ravel(), color='orange', alpha=0.2, label='95% confidence interval')
                    axs[0, sampling_method_idx].plot(sorted_data_X, sorted_data_y, marker='o', linestyle='', color='green', markerfacecolor='green', label='Prior data')
                    axs[0, sampling_method_idx].plot(sorted_sample_data_X, sorted_sample_data_y, marker='o', linestyle='', color='red', markerfacecolor='red', label='Sample points')
                    axs[0, sampling_method_idx].set_title(f'Method: {sampling_method}, iter.: {sampling_iteration_idx+1}/{sampling_iterations}') 
                    #axs[0, sampling_method_idx].legend()
                    axs[0, sampling_method_idx].grid(True)
                    if sampling_method_idx == 0:
                        axs[0, sampling_method_idx].set_ylabel(f'Function value')
                    # Next the inclusion probability graphs
                    axs[1, sampling_method_idx].cla()
                    to_be_sampled_x, inclusion_probabilities = sort_by_x_values(X=to_be_sampled_x, y=inclusion_probabilities)
                    axs[1, sampling_method_idx].plot(to_be_sampled_x, inclusion_probabilities, marker='', linestyle='-', color='violet', markerfacecolor='yellow', label='Inclusion probility')
                    sample_incprob_idxs = np.where(np.isin(to_be_sampled_x, sorted_sample_data_X))[0]
                    #sample_ind
                    axs[1, sampling_method_idx].plot(sorted_sample_data_X, inclusion_probabilities[sample_incprob_idxs], marker='o', linestyle='', color='red', markerfacecolor='red', label='Sample points')
                    axs[1, sampling_method_idx].grid(True)
                    #axs[1, sampling_method_idx].legend()
                    if sampling_method_idx == 0:
                        axs[1, sampling_method_idx].set_ylabel(f'Inclusion probability')
                    axs[2, sampling_method_idx].cla()
                    axs[2, sampling_method_idx].plot(sampling_data_container[sampling_method]['KLD'], marker='o', linestyle='-', markerfacecolor='orange', color='blue', label='KL-distance')
                    axs[2, sampling_method_idx].grid(True)
                    #axs[2, sampling_method_idx].legend()
                    if sampling_method_idx == 0:
                        axs[2, sampling_method_idx].set_ylabel(f'KL-distance')
                    KL_list += list(sampling_data_container[sampling_method]['KLD'])
                    axs[3, sampling_method_idx].cla()
                    axs[3, sampling_method_idx].bar(bins[:-1], densities[0], width=bins[1] - bins[0], alpha=0.5, label='True distribution', color='blue')
                    axs[3, sampling_method_idx].bar(bins[:-1], densities[1], width=bins[1] - bins[0], alpha=0.5, label='Estimated distribution', color='green')
                    axs[3, sampling_method_idx].grid(True)
                    #axs[3, sampling_method_idx].legend()
                    if sampling_method_idx == 0:
                        axs[3, sampling_method_idx].set_ylabel(f'Reponse density')
                    plt.pause(0.1)
            if self.X.shape[1] <= 1:
                for sampling_method_idx, sampling_method in enumerate(sampling_method_list):
                    axs[2, sampling_method_idx].set_ylim(np.min(KL_list)-1, np.max(KL_list)+1)
                plt.savefig(f'fig{sampling_iteration_idx+1}.png')
        return sampling_data_container

    
    def plot_target_function(self):
        fig = plt.figure(figsize=(6, 6))
        plt.ion()
        plt.show()
        plt.plot(self.X.ravel(), self.y, label='target function', color='blue', linestyle='--')
        plt.xlabel('Explanatory variable value')
        plt.ylabel('Function value')
        #plt.gca().set_aspect('equal')
        plt.ylim(-22, 20)  # Replace these values with your desired y-range
        plt.grid(True)
        plt.title('Target function plot')
        plt.legend()
        plt.tight_layout()  # Automatically adjusts spacing
        plt.pause(1)
        
            




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
# Disable ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BOsampler:
    # Constructor method (initialization)
    def __init__(self, 
                 hyperparam_grid=None,
                 X=None,
                 y=None):
        self.hyperparam_grid = hyperparam_grid # dictionary of model hyperparameters, including kernels.
        self.X = X # The input features of the data set provided for the sampler.
        self.y = y # The target variable value of the data set.
        self.scaler = None # The input feature scaler constructed with the X data.

    def initialize_scaler(self):
        self.scaler = StandardScaler().fit(self.X)

    def normalize_x(self):
        self.X = self.scaler.transform(self.X)


    def inverted_lower_confidence_bound(y_mean=0, y_std=0, l=0.2):
        return l*y_std - y_mean
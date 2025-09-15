# Hyperparameter tuning using Bayesian Optimization
# educative.io miniproject from 
# https://www.educative.io/courses/bayesian-machine-learning-for-optimization-in-python/hyperparameter-tuning-using-bayesian-optimization/project

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

import numpy as np
import dragonfly.exd as exd
import dragonfly.opt as opt
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.opt.gp_bandit import CPGPBandit
from dragonfly import load_config
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# random seed
RS = 42



# what proportion of data to use for testing
p_test = 0.2

y_col = 'target'
iris = load_iris(as_frame=True).frame
X = iris.drop(columns=y_col)
y = iris[y_col]

# def load_data():
#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=p_test, random_state=RS)
#     return X_train, X_test, y_train, y_test

# Support Vector Classifer (SVC) parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# params ={
#     'C':1
#     , 'kernel':'rbf'
#     , 'gamma': 'scale' # or auto, or float > 0
#     , 'shrinking': True
#     , 'random_state': RS
# }

# # Define objective function: maximize accuracy on validation data
# def objective_fn(params):
#     classifier = make_pipeline(
#         StandardScaler(),
#         SVC(**params)
#     )
#     classifier.fit(X, y)
#     y_pred = classifer.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy


# FROM THE SOLUTION
def load_and_split_data():
    data = load_iris()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function: maximize accuracy on validation data

def objective_function(x):
    C, gamma = x
    C_value = 10 ** C
    gamma_value = 10 ** gamma

    model = SVC(C=C_value, gamma=gamma_value, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return -accuracy_score(y_val, y_pred)



if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_and_split_data()

    domain_1 = exd.domains.EuclideanDomain([(np.log(0.001), np.log(100))])
    domain_3 = exd.domains.EuclideanDomain([(np.log(0.1), np.log(10))])

    domain = exd.domains.CartesianProductDomain([domain_1, domain_3])
    domain = [{'name': 'x1', 'type': 'float', 'min': np.log(0.001), 'max': np.log(100.0)},
    {'name': 'x2', 'type': 'float', 'min': np.log(0.1), 'max': np.log(10.0)}]
    config_params = {'domain': domain}
    ld_cfg = load_config(config_params)
    func_caller = CPFunctionCaller(None, ld_cfg.domain, domain_orderings=ld_cfg.domain_orderings)
    opt_algorithm = CPGPBandit(func_caller, ask_tell_mode=True)
    opt_algorithm.initialise()

    best_objective_value = float('inf')  # Initialize with a high value since we're minimizing
    best_params = None 
    for i in range(20):
        x = opt_algorithm.ask()
        y = objective_function(x)
        opt_algorithm.tell([(x, y)])
        if y < best_objective_value:
            best_objective_value = y
            best_params = {
                'C': 10 ** x[0],
                'gamma': 10 ** x[1]
            }
        
        print(f"Iter={i}. hyperparameters: {x} and accuracy={y} vs best acc={best_objective_value}")

    best_point=best_params
    best_value = -best_objective_value  # We were minimizing the negative accuracy

    print("Best hyperparameters:", best_point)
    print("Best objective value:", best_value)



# # define hyperparameter search space
# def define_search_space(lower, upper):
#     """Wrapper around dragonfly.exd.domain.EuclideanDomain 
#     to define search space. Takes the log of the lower and upper bounds."""
#     return exd.domains.EuclideanDomain(bounds=[(np.log(lower), np.log(upper))])
    
# if __name__ == "__main__":
#     # load iris data
#     X_train, X_test, y_train, y_test = load_data()

#     domain_1 = define_search_space(lower=0.001, upper=100)
#     domain_3 = define_search_space(lower=0.1, upper=10)

#     domain = exd.domains.CartesianProductDomain(([domain_1, domain_3]))
#     domain = [
#         {'name': 'x1', 'type': 'float', 'min': np.log(0.001), 'max':np.log(100.0)},
#         {'name': 'x2', 'type': 'float', 'min': np.log(0.1), 'max':np.log(10)},
#     ]
#     config_params = {'domain': domain}
#     ld_cfg = load_config(config_params)

#     func_caller = CPFunctionCaller(
#         func=None, 
#         domain=ld_cfg.domain, 
#         domain_orderings=ld_cfg.domain_orderings
#     )

#     opt_algorithm = CPGPBandit(func_caller=func_caller, ask_tell_mode=True)
#     opt_algorithm.initialise()


#     # initial number of random evaluations to bootstrap the optimization process
#     # and explore the hyperparameter
#     n_init = 30
#     best_list = []
#     accuracy_list = []
#     best_params, best_acc = None, float('inf') # accuracy is multiplied by -1 in objective func, so we want lower number

#     for i in range(n_init):
#         new_params = opt_algorithm.ask()
#         new_accuracy = objective_function(new_params)
#         opt_algorithm.tell([new_params, new_accuracy])
#         print(f"iteration {i}: new_params={new_params}, new_accuracy={new_accuracy}")
#         best_list.append(new_params)
#         accuracy_list.append(new_accuracy)

#         # Update new best if this iteration is better than previous ones
#         if new_accuracy < best_acc:
#             best_acc = new_accuracy
#             best_params = {
#                 'C': 10 ** x[0],
#                 'gamma': 10 ** x[1]
#             }
    
#     print(f"Final best: hyperparams: {best_params} and accuracy: {-1*accuracy}")


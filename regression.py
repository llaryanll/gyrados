from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import PassiveAggressiveRegressor, HuberRegressor, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score



def evaluateBestRegressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a dictionary to store the models and their scores
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet Regression": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Support Vector Machine": SVR(),
        "Passive Aggressive": PassiveAggressiveRegressor(),
        "Huber Regression": HuberRegressor(),
        "SGD Regression": SGDRegressor(),
        "Bayesian Ridge": BayesianRidge(),
        "Kernel Ridge": KernelRidge(),
        "Gaussian Process": GaussianProcessRegressor(),
        "Multi-layer Perceptron": MLPRegressor()
    }

    best_model_name_mse = None
    best_model_name_acc = None
    best_model_score = float('inf')
    best_model_accuracy = 0

    # Train and test each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        acc = r2_score(y_test,y_pred)
        print(f"{name} MSE: {mse}")
        print(f"{name} ACCURACY SCORE: {acc}")
        
        # Check if current model has better score
        if mse < best_model_score:
            best_model_score = mse
            best_model_name_mse = name
        
        if acc > best_model_accuracy:
            best_model_accuracy = acc
            best_model_name_acc = name

    print(f"\nThe best model is {best_model_name_mse} with MSE: {best_model_score}")
    print(f"\nThe best model is {best_model_name_acc} with R^2: {best_model_accuracy}")
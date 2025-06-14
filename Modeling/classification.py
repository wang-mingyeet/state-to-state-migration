import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# processing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# analysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean(df):
    """
    Prepare migration and feature data for classification modelling.

    Parameters:
        df: Dataframe of all features and migration rates.

    Returns:
        X: dataframe with all considered features
        y: 1D dataframe of all labels - net inflow/outflow, class 0 = outflow, class 1 = inflow
    """

    # drop unused columns 
    df = df.drop(columns=["net in", "net out", "state", "year"])
    
    # convert to 0 for negative (net outflow) and 1 for positive (net inflow)
    df["net_class"] = (df["net total"] > 0).astype(int)
    
    # make X and y by isolating net class and dropping unnecessary columns
    X = df.drop(columns=["net total", "net_class"])
    y = df["net_class"]

    return X, y

def select_features(X, y):
    """
    Select top five features using Lasso Regression.

    Parameters:
        X: All feature and migration data
        y: Output labels 

    Returns:
        Dataframe X with only the selected features. 
    """
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)
    return X[:, lasso.coef_ != 0]

def data_prep(X, y):
    """
    Prepare the data to train the model by standardizing, selecting features, and doing train-test split.

    Parameters:
        X: feature data
        y: corresponding migration rate

    Returns:
        X_train, X_test, y_train, y_test: data for training and testing the model.
    """
    # standardize the data
    # necessary because notice how columns like home value index have extremely large numbers - higher variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # select top 5 features using lasso regression
    X_selected = select_features(X_scaled, y)
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)

    return X_train, X_test, y_train, y_test    

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix using a more visual representation.

    Parameters:
        y_true : dataframe of true class labels 
        y_pred : dataframe of predicted class labels

    Returns:
        Nothing. Directly plots.

    """
    # set labels and title
    labels=["Net Outflow", "Net Inflow"]
    title="Confusion Matrix"
    
    # get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # plot confusion matrix visually 
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def rf_param_select(X_train, y_train):
    """
    Use GridSearchCV to find the best Random Forest parameters.

    Parameters:
        X_train: training feature data
        y_train: training migration rate data
    
    Returns:
        Best hyperparameters for the Random Forest Classifier.
    """
    # possible values for all hyperparameters
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    # fit to model and find best values 
    clf = RandomForestClassifier(class_weight={0: 1.116, 1: 0.906})
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    # print best parameters 
    print("Best Random Forest Params:", grid.best_params_)
    return grid.best_estimator_

def xgb_param_select(X_train, y_train):
    """
    Use GridSearchCV to find the best XGBoost parameters.

    Parameters:
        X_train: training feature data
        y_train: training migration rate data
    
    Returns:
        Best hyperparameters for the XGBoost model.
    """
    # possible values for all hyperparameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    # fit to model and find best values
    xgb = XGBClassifier(eval_metric='logloss')
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    # print best parameters 
    print("Best XGBoost Params:", grid.best_params_)
    return grid.best_estimator_


def lr_param_select(X_train, y_train):
    """
    Use GridSearchCV to find the best Logistic Regression parameters.
    
    Parameters:
        X_train: training feature data
        y_train: training migration rate data
    
    Returns:
        Best hyperparameters for the Logistic Regression model.
    """
    # possible values for all hyperparameters
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10]
    }

    # fit to model and find best values
    lr = LogisticRegression(solver='saga', class_weight={0: 1.116, 1: 0.906}, max_iter=5000)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    # print best parameters 
    print("Best Logistic Regression Params:", grid.best_params_)
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test, threshold=0.5, name="Model", print_results=True):
    """
    Predicts and prints evaluation metrics for a given model using a custom threshold.

    Parameters:
        model: machine learning model used
        X_test: test features
        y_test: true labels
        threshold: float, probability threshold for classifying as class 1.
        name: str, name to display in output header

    Returns:
        y_pred: predict
    """
    # show probabilities for class 1
    y_proba = model.predict_proba(X_test)[:, 1]
    # adjust for custom threshold
    y_pred = (y_proba > threshold).astype(int) 

    # skip this if False (we don't want to print everything when finding average)
    if print_results:
        # display results
        print(f"Evaluation for {name} at threshold = {threshold}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        # plot confusion matrix more visually 
        plot_confusion_matrix(y_test, y_pred)
    
        # overall summary using sklearn classification report
        print(classification_report(y_test, y_pred, target_names=["Net Outflow", "Net Inflow"]))

    return y_pred

def repeat_eval(X, y, model_fn, threshold=0.5, name="Model", repeats=5):
    """
    Repeat model training and evaluation to compute average accuracy.

    Parameters:
        X: feature data
        y: labels
        model_fn: function that returns a trained model (e.g. cl.rf_param_select)
        threshold: probability threshold for classification
        name: model name to show in output
        repeats: times to repeat the train-test split

    Returns:
        avg_acc: average accuracy over all runs
    """
    # list to store accuracies
    accuracies = []

    # repeat for desired number of times
    for i in range(repeats):
        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # train model
        model = model_fn(X_train, y_train)
        # evaluate model
        y_pred = evaluate_model(model, X_test, y_test, threshold=threshold, name=f"{name} (Run {i+1})", print_results=False)
        # get accuracy
        acc = (y_pred == y_test).mean()
        accuracies.append(acc)
    
    # get and print average accuracy
    avg_acc = np.mean(accuracies)
    print(f"\nAverage Accuracy for {name} over {repeats} runs: {avg_acc:.4f}")
    return avg_acc

def make_ensemble_model(X_train, y_train, use_rf=True, use_xgb=True, use_lr=True, weights=None, voting='soft'):
    """
    Create a VotingClassifier ensemble from selected base models.

    Parameters:
        X_train: feature training data
        y_train: training data with labels (migration classes)
        use_rf: whether to include Random Forest
        use_xgb: whether to include XGBoost
        use_lr: whether to include Logistic Regression
        weights: list of weights for each model in order (rf, xgb, lr)
        voting: 'soft' (default) or 'hard'

    Returns:
        ensemble: VotingClassifier model
    """
    # store models and weights
    estimators = []
    default_weights = []

    # if including Random Forest
    if use_rf:
        rf = rf_param_select(X_train, y_train)
        estimators.append(('rf', rf))
        default_weights.append(1)

    # if including XGBoost
    if use_xgb:
        xgb = xgb_param_select(X_train, y_train)
        estimators.append(('xgb', xgb))
        default_weights.append(1)
    
    # if including Logistic Regression
    if use_lr:
        lr = lr_param_select(X_train, y_train)
        estimators.append(('lr', lr))
        default_weights.append(1)

    # use custom weights if provided
    weights = weights if weights is not None else default_weights

    # create ensemble model
    ensemble = VotingClassifier(estimators=estimators, voting=voting, weights=weights)
    return ensemble

def get_ensemble_model_fn(use_rf=True, use_xgb=True, use_lr=True, weights=None, voting='soft'):
    """
    Returns a function that builds and fits a VotingClassifier ensemble model.
    Can be passed directly into repeat_eval as model_fn.

    Parameters:
        use_rf: whether to include Random Forest
        use_xgb: whether to include XGBoost
        use_lr: whether to include Logistic Regression
        weights: list of weights (ordered by included models)
        voting: 'soft' (default) or 'hard'

    Returns:
        model_fn: a function that takes (X_train, y_train) and returns a fitted VotingClassifier
    """
    
    def model_fn(X_train, y_train):
        # build the ensemble model with specified components and weights
        model = make_ensemble_model(
            X_train=X_train,
            y_train=y_train,
            use_rf=use_rf,
            use_xgb=use_xgb,
            use_lr=use_lr,
            weights=weights,
            voting=voting
        )
        # fit the ensemble model on the provided training data
        model.fit(X_train, y_train)
        # return the fitted ensemble model
        return model

    # return the function so it can be used with repeat_eval
    return model_fn

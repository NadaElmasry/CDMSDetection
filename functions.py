
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
# Models
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_curve
from constants import k_folds, scoring_metric, SEED, cmap


def count_plot(data, feature, title, hue=True):
    plt.figure(figsize=(5, 5), facecolor='#F6F5F4')
    total = float(len(data))
    if hue:
        ax = sns.countplot(
            x=data[feature], hue=data['group'], palette='coolwarm')

    else:
        ax = sns.countplot(x=data[feature], hue=None, palette='coolwarm')

    ax.set_facecolor('#F6F5F4')

    for p in ax.patches:

        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2., height + 3, '{:1.1f} %'.format((height/total)*100), ha="center",
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

    ax.set_title(title, fontsize=16, y=1.05)
    sns.despine(right=True)
    sns.despine(offset=5, trim=True)


# Creating a function for metrics
def model_performance_classification_sklearn(model, predictors, target):
    # Predicting using independent variables
    pred = model.predict(predictors)

    # Metrics used
    acc = accuracy_score(target, pred)  # To compute the accuracy score
    recall = recall_score(target, pred)  # To compute the recall score
    precision = precision_score(target, pred)  # To compute the precision score
    f1 = f1_score(target, pred)  # To predict the f-1 score

    # Creating the dataframe for the metrics
    df_perf = pd.DataFrame(
        {'Accuracy': acc, 'Recall': recall, 'Precision': precision, 'F1': f1, },
        index=[0],)

    return df_perf


def cross_val_train_eval(model, X_train, y_train, X_test, y_test, scoring_metric=scoring_metric, k_folds=k_folds, SEED=SEED):
    # Perform k-fold cross-validation
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X_train,
                             y_train, cv=cv, scoring=scoring_metric)

    # Get the cross-validated predictions
    preds = cross_val_predict(model, X_train, y_train, cv=cv)

    # Print the accuracy for each fold
    print("Accuracy for each fold:")
    for fold, score in enumerate(scores):
        print(f"Fold {fold+1}: {score}")

    # Calculate and print the average accuracy across all folds
    average_accuracy_training = np.mean(scores)
    print("Average Training Accuracy:", average_accuracy_training)

    # Evaluate the ensemble model on the test data
    model.fit(X_train, y_train)
    accuracy_testing = model.score(X_test, y_test)
    print("Accuracy:", accuracy_testing)


# Creating a function for the confusion matrix
def confusion_matrix_sklearn(model, model_name, predictors, target, cmap=cmap):
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f'Confusion Matrix {model_name} Model')
    plt.show()


def plot_feature_importance(model, model_name, features_names, cmap=cmap):
    if model_name.lower() == 'svc' or model_name.lower() == 'svm':
        # Retrieve the learned coefficients
        coefficients = model.coef_
        # Determine feature importance
        # Assuming binary classification, hence accessing the first row of normalized coefficients
        feature_importance = abs(coefficients[0])

    elif model_name.lower() == 'rf':
        feature_importance = model.feature_importances_

    else:
        raise ValueError(
            "Please enter a valid model name. The supported model names are svc and rf.")

    fi_df = pd.DataFrame({'Feature': features_names,
                          'Importance': feature_importance}).sort_values(by='Importance',
                                                                         ascending=False)

    fig, ax = plt.subplots(figsize=(15, 5))

    sns.barplot(data=fi_df,
                x='Importance',
                y='Feature',
                palette=cmap,
                ax=ax)
    ax.set_title(f'Feature Importance in {model_name}')


def plot_roc_auc(model, model_name, X_test, y_test):
    y_prob = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, cmap=cmap):
    # calc test pred
    y_pred = model.predict(X_test)
    # calc comfusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Calculate sensitivity
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    # Calculate specificity
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    # Calculate misclassification rate
    misclassification_rate = (cm[0, 1] + cm[1, 0]) / np.sum(cm)

    print(f"{model_name.upper()} Sensitivity:", sensitivity)
    print(f"{model_name.upper()} Specificity:", specificity)
    print(f"{model_name.upper()} Misclassification Rate:",
          misclassification_rate)
    # classification report
    classification_report(y_test, y_pred,
                          target_names=['CDMS', 'non-CDMS'], digits=3)
    # sensitivity, specificity, misclassification
    # Confusion Matrix
    confusion_matrix_sklearn(model, model_name, X_test, y_test, cmap=cmap)
    # ROC AUC Curve
    plot_roc_auc(model, model_name, X_test, y_test)

    # Feature Importance
    features_names = X_train.columns
    plot_feature_importance(model, model_name, features_names, cmap='coolwarm')

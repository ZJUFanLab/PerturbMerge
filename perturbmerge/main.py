import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import csv
from tqdm import tqdm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def train_models(x_data, y_data, model_type ='gbm', save_model=False, model_path=None,auc_threshold=0.8):
    """
    Train models on provided feature and target datasets and optionally save the trained models if performance exceeds the AUC threshold.
    
    Parameters:
    - x_data (DataFrame): Input features dataset.
    - y_data (DataFrame): Corresponding target dataset for each feature.
    - model_type (str): Type of model to train. Options are 'gbm' for Gradient Boosting Machine, 'randomforest' for Random Forest, or 'xgboost'.
    - save_model (bool): Flag to save the trained models to the specified path if their performance is above the AUC threshold.
    - model_path (str): Path where the models should be saved.
    - auc_threshold (float): AUC threshold above which the model will be saved.
    """
    feature_importance_dict = {}
    performance_metrics = []

    for i in tqdm(range(y_data.shape[1]), desc="Processing Columns"):
        y_column_name = y_data.columns[i]
        model_filename = f'{model_path}{y_column_name}_model.pkl' if model_path else f'{y_column_name}_model.pkl'

        # Prepare the data
        svm_data = pd.concat([x_data, y_data.iloc[:, i]], axis=1)
        svm_data = svm_data.dropna(axis=0, how='any') 
        X_data = svm_data.iloc[:, :-1]
        Y_data = svm_data.iloc[:, -1]

        scaler = StandardScaler()
        X_data_scaled = scaler.fit_transform(X_data)

        X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, Y_data, test_size=0.3, random_state=42)

        if model_type=='gbm':
            model = GradientBoostingClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        elif model_type=='randomforest':
            model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5, random_state=42)
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            threshold = (sum(y_train == 0) / len(y_train)) * 0.5 + (sum(y_train == 1) / len(y_train)) * 0.5
            y_pred = (y_pred_prob >= threshold).astype(int)
        
        elif model_type=='xgboost':
            model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        
        metrics = calculate_metrics(y_test, y_pred, y_train, model, X_train, X_test)
        
        if metrics[2]>auc_threshold and save_model:
            joblib.dump(model, model_filename)
            
        performance_metrics.append((y_column_name,) + metrics)

        feature_importance = update_feature_importances(model, X_data)
        feature_importance_dict[y_column_name] = feature_importance

    save_results(feature_importance_dict, performance_metrics, model_path)
    

def predict_models(new_data, model_path, model_prefix):
    """
    Loads models corresponding to the specified genes from the given path and makes predictions on new data provided.
    
    Parameters:
    - new_data (DataFrame): New data to predict.
    - model_path (str): Path where the models are stored.
    - model_prefix (str): Prefix used in the model file names before the gene name.
    
    Returns:
    - DataFrame: A DataFrame containing predictions for each gene model.
    """
    predictions = {}
    available_models = []
    model_files = os.listdir(model_path)

    # Find and prepare models for the genes in the list
    for filename in os.listdir(model_path):
        if filename.startswith(model_prefix) and filename.endswith("_model.pkl"):
            gene = filename[len(model_prefix):-len("_model.pkl")]
            available_models.append(gene)

    print(f"Loading models for {len(available_models)} genes.")

    # Load and predict using each available model
    for gene in tqdm(available_models, desc="Predicting models"):
        model_filename = f"{model_prefix}{gene}_model.pkl"
        full_model_path = os.path.join(model_path, model_filename)
        
        model = joblib.load(full_model_path)
        scaler = StandardScaler()
        predict_data_scaled = scaler.fit_transform(new_data)
        prediction = model.predict(predict_data_scaled)

        predictions[gene] = prediction

    prediction_df = pd.DataFrame(predictions, index=new_data.index)
    return prediction_df


def calculate_metrics(y_test, y_pred, y_train, model, X_train, X_test):
    accuracy = accuracy_score(y_test, y_pred)
    auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, auc_train, auc_test, recall, precision, f1

def update_feature_importances(model, X_data):
    feature_importance_indices = model.feature_importances_.argsort()[::-1][:20]
    feature_names = X_data.columns[feature_importance_indices]
    return list(feature_names)

def save_results(feature_importance_dict, performance_metrics, model_path):
    with open(f'{model_path}feature_importance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Column Name', 'Top 20 Features'])
        for column_name, top_features in feature_importance_dict.items():
            writer.writerow([column_name] + top_features)

    metrics_df = pd.DataFrame(performance_metrics, columns=['Column Name', 'Accuracy', 'Train AUC', 'Test AUC', 'Recall', 'Precision', 'F1 Score'])
    metrics_df.to_csv(f'{model_path}performance_metrics.csv', index=False)


def combine_predictions(prediction1, prediction2, prediction3):
    """
    Combines predictions from three different models based on a voting system.

    Parameters:
    - prediction1 (DataFrame): Prediction results from the first model.
    - prediction2 (DataFrame): Prediction results from the second model.
    - prediction3 (DataFrame): Prediction results from the third model.

    Returns:
    - DataFrame: A DataFrame containing combined predictions for each gene.
                  Each entry is 1 if at least two models agree on influence (value=1),
                  otherwise 0.
    """
    combined_predictions = pd.DataFrame(index=prediction1.index, columns=prediction1.columns)

    # Iterate over each column (perturbed gene)
    for gene in prediction1.columns:
        # Sum up the influence counts from each prediction DataFrame
        combined = prediction1[gene] + prediction2[gene] + prediction3[gene]
        # Determine influence based on majority vote (at least 2 out of 3)
        combined_predictions[gene] = (combined >= 2).astype(int)

    return combined_predictions
#!/usr/bin/env python
# coding: utf-8

import platform
import psutil
import GPUtil

system_info = platform.uname()

print("\n******System Information:")
print(f"System: {system_info.system}")
print(f"Node Name: {system_info.node}")
print(f"Release: {system_info.release}")
print(f"Version: {system_info.version}")
print(f"Machine: {system_info.machine}")
print(f"Processor: {system_info.processor}")

cpu_info = platform.processor()
cpu_count = psutil.cpu_count(logical=False)
logical_cpu_count = psutil.cpu_count(logical=True)

print("\n******CPU Information:")
print(f"Processor: {cpu_info}")
print(f"Physical Cores: {cpu_count}")
print(f"Logical Cores: {logical_cpu_count}")

memory_info = psutil.virtual_memory()

print("\n******Memory Information:")
print(f"Total Memory: {memory_info.total} bytes")
print(f"Available Memory: {memory_info.available} bytes")
print(f"Used Memory: {memory_info.used} bytes")
print(f"Memory Utilization: {memory_info.percent}%")

disk_info = psutil.disk_usage('/')

print("\n******Disk Information:")
print(f"Total Disk Space: {disk_info.total} bytes")
print(f"Used Disk Space: {disk_info.used} bytes")
print(f"Free Disk Space: {disk_info.free} bytes")
print(f"Disk Space Utilization: {disk_info.percent}%")

gpus = GPUtil.getGPUs()

if not gpus:
    print("\nNo GPU detected.")
else:
    for i, gpu in enumerate(gpus):
        print(f"\n******GPU {i + 1} Information:")
        print(f"ID: {gpu.id}")
        print(f"Name: {gpu.name}")
        print(f"Driver: {gpu.driver}")
        print(f"GPU Memory Total: {gpu.memoryTotal} MB")
        print(f"GPU Memory Free: {gpu.memoryFree} MB")
        print(f"GPU Memory Used: {gpu.memoryUsed} MB")
        print(f"GPU Load: {gpu.load * 100}%")
        print(f"GPU Temperature: {gpu.temperature}°C")


from eBoruta import eBoruta
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import time
from glob import glob
import tqdm as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import plotly.io as pio
import shap
import datetime

from plotly.io import show
from optuna.importance import get_param_importances, MeanDecreaseImpurityImportanceEvaluator
from datetime import datetime
# display(HTML("<style>.container { width:80% !important; }</style>"))
# pd.set_option("display.max_colwidth", 100)

print("\n******Current working dir", os.getcwd())

import optuna
import optunahub

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix
from xgboost.dask import DaskQuantileDMatrix
import xgboost as xgb
print('xgb=:',xgb.__version__)
print('optuna=:',optuna.__version__)
print('optunahub=:',optunahub.__version__)




# load and preprocess data
def prepare_DMatrix_array(csv_path, col, save_dir):
    """
    Load data from a CSV file, preprocess it, and convert it into a DMatrix format for XGBoost.

    Parameters:
    - csv_path: Path to the CSV file.
    - col: List of column names for features.
    - save_dir: Directory path to save preprocessed data.

    Returns:
    - X_train: DataFrame of features.
    - y_train: Series of target variable.
    - dtrain: DMatrix for XGBoost.
    """
    
    # Load the data
    train_df = pd.read_csv(csv_path)


    # Separate features and target variable for training and testing
    X_train = train_df[cols]
    y_train = train_df["AGBD"]

    X_train.to_csv(f"{save_dir}X_train.csv", index=False)
    y_train.to_csv(f"{save_dir}y_train.csv", index=False)
    
    print("\nTotal training samples=:", len(X_train))
    
    # Converting the datasets into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    print("\nxgb.DMatrix transition successfully!")
    with open(f"{save_dir}_used_features.txt", "w") as file:
         file.write(f"used features: \n{col}")
    return X_train, y_train, dtrain




# plot hist of target via in training dataset
def plot_target_histograms(y_train, save_dir):
    """
    Plot a histogram of the target variable in the training dataset and save it as an image file.

    Parameters:
    - y_train: Series of target variable.
    - save_dir: Directory path to save the plot.

    Returns:
    - None
    """
    
    # Calculate the sample counts
    train_count = len(y_train)
    min_value = y_train.min()
    max_value = y_train.max()

    plt.figure(figsize=(4, 3))

    # Plot histogram for training data
    plt.hist(y_train, bins=2000, alpha=0.7, 
             label=f'Total samples (n={train_count}) \nMin value={min_value:.3f} Mg/ha \nMax value={max_value:.3f} Mg/ha', 
             color='blue')

    plt.xlabel('AGBD (Mg/ha)')
    plt.ylabel('Frequency')
    plt.title('Histogram of samples')
    plt.legend(loc='upper right', frameon=False)

    plt.savefig(f"{save_dir}all-hist.png", dpi=600, pad_inches=0.02, bbox_inches='tight')
    # plt.show(block=False)
    plt.close()

    print("\nHistogram of number for train-test split saved successfully!")
    return




# define objective func
def objective(trial, save_dir):

    """
    Perform hyperparameter tuning for an XGBoost model using Optuna.

    Parameters:
    - trial: Optuna trial object.
    - save_dir: Directory path to save results.

    Returns:
    - The mean RMSE of the best model.
    """
    
    param = {# paras
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method' : "hist", 
        'device' : "cuda",
        'booster': "gbtree",
        # 'booster': trial.suggest_categorical("booster", ["dart"]),
        
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=False, step=0.001), #range: [0,1]
        'gamma': trial.suggest_float('gamma',0, 10, step=0.01, log=False), #range:  [0, ]
        'max_depth': trial.suggest_int('max_depth', 3, 25), #range:  [0, ]
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15), #range:  [0, ]
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=False, step=0.001,), #range:  (0,1]
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, log=False, step=0.001,),#range:   (0, 1]
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20, log=False, step=0.001,),  # L2 regularization range: [0, ]
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 20, log=False, step=0.001,),     # L1 regularization range: [0, ]
    }

    if param["booster"] == "dart":
        # param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        # param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = 0.5#trial.suggest_float("rate_drop", 1e-8, 1.0, log=False)
        param["skip_drop"] = 0.5#trial.suggest_float("skip_drop", 1e-8, 1.0, log=False)

        
    # Perform cross-validation
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=2000,
        nfold=10,
        metrics="rmse",
        early_stopping_rounds=100,
        seed=42,
        as_pandas=True,
    )
    
    # Record the optimal number of boosting rounds
    best_iteration = cv_results['test-rmse-mean'].idxmin() + 1 #  best_boost_round
    trial.set_user_attr('best_iteration', best_iteration)
    trial.set_user_attr('full_param', param)

    
    # Manually report intermediate results and check for pruning
    for step in range(len(cv_results)):
        mean_rmse = cv_results.loc[step, 'test-rmse-mean']
        trial.report(mean_rmse, step=step)

        if trial.should_prune():
            print(f"\nTrial {trial.number} pruned at step {step} with mean RMSE {mean_rmse}")
            raise optuna.exceptions.TrialPruned()

    print(f"\nTrial {trial.number} completed {len(cv_results)} boosting rounds.")


    return cv_results['test-rmse-mean'].iloc[-1] 



def get_best_trial_metrics(study, dtrain, save_dir):
    """
    Evaluate the best trial from an Optuna study and calculate its metrics using XGBoost's cross-validation functionality.

    Parameters:
    - study: Optuna study object.
    - dtrain: XGBoost DMatrix for training.
    - save_dir: Directory path to save results.

    Returns:
    - A dictionary containing the best RMSE and R2 values.
    """

    param = study.best_trial.user_attrs['full_param'].copy()
    best_iteration = study.best_trial.user_attrs['best_iteration']

    # Make sure *rmse* is the official eval metric – do NOT add `r2`
    eval_metric = param.get("eval_metric", [])
    if isinstance(eval_metric, str):
        eval_metric = [eval_metric]
    if "rmse" not in eval_metric:
        eval_metric.append("rmse")
    # IMPORTANT: do **not** append "r2" – it is unknown in this build
    param["eval_metric"] = eval_metric


    def r2_eval(preds, dmat):
        y_true = dmat.get_label()
        ss_res = np.sum((y_true - preds) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        return "r2", r2     # (name, value)


    cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=best_iteration+1,
        nfold=10,
        seed=42,
        as_pandas=True,
        custom_metric=r2_eval           # <- custom metric
        # metrics=["rmse"]      # <- optional; forces printing rmse only
    )


    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "best_trial_train_val_cv_metrics.csv")
    cv_results.to_csv(csv_path, index=False)
    print(f"Exported CV results to {csv_path}")


    best_rmse = cv_results["test-rmse-mean"].iloc[-1]
    best_r2   = cv_results["test-r2-mean"].iloc[-1]

    print("\nCalculate average R2 and rmse in CV successfully!")
    
    return {"rmse": best_rmse, "r2": best_r2}

    

# export trained model and the best paras
def save_best_results_and_train_model(study, X_train, y_train, save_dir):
    """
    Print the best hyperparameters and score, save results to a text file, and train the final XGBoost model.

    Parameters:
    study: Optuna study object containing the results of hyperparameter optimization
    X_train (array-like): Training features
    y_train (array-like): Training labels
    save_dir (str): Directory to save output files

    Returns:
    The trained XGBoost model
    """
    
    # Print the best hyperparameters and the best score
    print("\nBest trial number:", study.best_trial.number)
    print("Best hyperparameters:", study.best_params)
    print("Best num_boost_round:", study.best_trial.user_attrs['best_iteration']+1)
    print("Best RMSE:", study.best_value)
    print("XGB training time:", used_time)

    # Save the best results to a txt file
    with open(f"{save_dir}_best_para_result.txt", "w") as file:
        file.write(f"Best trial number: {study.best_trial.number}\n")
        file.write(f"Best hyperparameters: {study.best_params}\n")
        file.write(f"Best num_boost_round: {study.best_trial.user_attrs['best_iteration']+1}\n")
        file.write(f"Best RMSE: {study.best_value}\n")
        file.write(f"XGB training time: {used_time}")

    # Converting the datasets into DMatrix format for XGBoost prediction
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Get best parameters from the study
    best_params = study.best_trial.user_attrs['full_param'].copy()

    # Train the final model
    final_model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=study.best_trial.user_attrs['best_iteration']+1,
        verbose_eval=True
    )

    # Save the final model (optional)
    final_model.save_model(f"{save_dir}final_model_.json")

    print("\nFinal model trained, best parameters saved successfully!")
    return final_model



# define rRMSE func
def relative_rmse(y_true, y_pred):
    """
    Calculate the Relative Root Mean Squared Error (RRMSE).

    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values

    Returns:
    float: RRMSE value as a percentage
    """
    
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calculate the mean of the actual values
    mean_y_true = np.mean(y_true)

    # Calculate RRMSE
    rrmse = rmse / mean_y_true if mean_y_true != 0 else np.inf  # Avoid division by zero

    return rrmse * 100




# define func of scatter plot for y and y_prediction    
def evaluate_and_plot(final_model, y, x, tipe, save_dir):
    """
    Evaluate the model performance and create a scatter plot with regression line.
    Parameters:
    final_model: Trained model for predictions
    x (array-like): Test dataset for predictions
    y (array-like): Actual values for testing
    tipe: str, 'train' or 'test'
    save_dir (str): Directory to save the output files
    """
    
    # Calculate predictions
    y_pred = final_model.predict(x)

    # Calculate R2 and RMSE
    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    rrmse = relative_rmse(y, y_pred)


    # Create the scatter plot with regression line
    figsize = (2 ,2)  # Adjust the figure size as needed
    scale_factor = figsize[0]
    
    plt.figure(figsize=figsize)
    sns.regplot(x=y, 
                y=y_pred, 
                marker='o',
                scatter_kws={'color': '#005AB5', 's': 2.0, 'alpha': 0.6}, 
                line_kws={'color': '#0C7BDC'})

    # Add labels and title
    plt.xlabel("Measured AGBD (Mg/ha)", fontsize=4* scale_factor)
    plt.ylabel("Estimated AGBD (Mg/ha)", fontsize=4* scale_factor)


    # Find the maximum value
    max_value = max(max(y), max(y_pred))
    
    print("\nmax of y：",max(y))
    print("max of y_pred：",max(y_pred))
    print(f"max vlaue in {tipe}: {max_value}")

    # Set x-y axis limit from 0 to 900
    top = max_value+100
    plt.xlim(0, top)
    plt.ylim(0, top)

    # # Adjust the size of x and y axis tick labels
    plt.xticks(ticks=np.arange(0, top + 1, 100),fontsize=4 * scale_factor, rotation=90)
    plt.yticks(ticks=np.arange(0, top + 1, 100),fontsize=4 * scale_factor)
    
    # Plot the diagonal line representing perfect predictions
    plt.plot([0, top], [0, top], 
             color='black', 
             linestyle='--', 
             linewidth=0.7,
             alpha=0.7)

    # Equation of the regression line (slope and intercept)
    slope, intercept = np.polyfit(y, y_pred, 1)

    # Display the regression equation, R^2, and RMSE
    equation_text = f"y = {slope:.2f}x + {intercept:.2f}\n" \
                    f"$R^2$ = {r2:.2f}\n" \
                    f"RMSE = {rmse:.2f}\n" \
                    f"rRMSE = {rrmse:.2f}%"

    plt.text(0.05, 0.95, 
             equation_text, 
             transform=plt.gca().transAxes,
             fontsize=3.5* scale_factor, 
             verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.0))

    # Create a DataFrame with y and y_pred
    results_df = pd.DataFrame({'Measured_Biomass': y, 'Predicted_Biomass': y_pred})

    # Export the DataFrame to a CSV file
    results_df.to_csv(f"{save_dir}plot_scatter_{tipe}.csv", index=False)

    # Save the plot to a file
    plt.savefig(f"{save_dir}plot_scatter_{tipe}.png", dpi=600, pad_inches=0.02, bbox_inches='tight')
    # plt.gca().set_aspect('equal', adjustable='box') 
    
    # Show plot
    # plt.show(block=False)
    plt.close()
    print(f"\nScatter plot for {tipe}, prediction datafram exported successfully!")
    
    return




# SHAP based feature importance 
def compute_and_plot_shap(final_model, X_train, save_dir):
    """
    Compute and plot SHAP values with automatic GPU/CPU selection.
    
    Parameters:
    final_model: Trained model for SHAP explanation
    X_train: Training data
    save_dir (str): Directory to save the output files
    """
    
    # Try to use GPUTree explainer, fall back to TreeExplainer if not available
    try:
        explainer = shap.explainers.GPUTree(final_model, X_train)
        print("\n--- Using GPUTree explainer ---")
    except Exception as e:
        explainer = shap.TreeExplainer(final_model)
        print(f"\n--- GPUTree explainer not available. Using TreeExplainer instead. Error: {e} ---\n")


    # Compute SHAP values with additivity check disabled
    shap_values_array = explainer.shap_values(X_train, check_additivity=False)

    # Wrap shap_values_array in a shap.Explanation object
    shap_values = shap.Explanation(values=shap_values_array, 
                                   base_values=explainer.expected_value,
                                   data=X_train,
                                   feature_names=X_train.columns)

    # Convert SHAP values to a DataFrame for saving
    shap_df = pd.DataFrame(shap_values_array, columns=X_train.columns)
    shap_df.to_csv(f"{save_dir}shap_values.csv", index=False)
    print("\nSHAP values saved to CSV successfully!")

    # Set the max_display based on feature count
    max_display = min(100, X_train.shape[1])

    # Create and save the SHAP bar plot
    fig_bar = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.xlabel("Mean |SHAP value|", fontsize=14)  # Adjust x-axis label size
    fig_bar.savefig(f"{save_dir}shap_bar.png", dpi=600, bbox_inches='tight')
    plt.close(fig_bar)

    # Create and save the SHAP summary plot
    fig_summary_plot = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.summary_plot(shap_values, max_display=max_display, show=False)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=14)  # Adjust x-axis label size
    fig_summary_plot.savefig(f"{save_dir}shap_summary_plot.png", dpi=600, bbox_inches='tight')
    plt.close(fig_summary_plot)

    # Create and save the SHAP violin plot
    fig_violin = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.violin(shap_values, max_display=max_display, show=False)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=14)
    fig_violin.savefig(f"{save_dir}shap_violin.png", dpi=600, bbox_inches='tight')
    plt.close(fig_violin)

    # Create and save the SHAP beeswarm plot
    fig_beeswarm = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=14)
    fig_beeswarm.savefig(f"{save_dir}shap_beeswarm.png", dpi=600, bbox_inches='tight')
    plt.close(fig_beeswarm)

    print("\nSHAP plots saved successfully!")
    
    return




# history of optimization
def plot_optuna_results(study, save_dir, width=500, height=500):
    """
    Visualize the optimization history of an Optuna study.

    Parameters:
    study: Optuna study object containing the results of hyperparameter optimization
    save_dir (str): Directory to save output files
    width (int): Width of the plots (default: 500)
    height (int): Height of the plots (default: 500)

    Returns:
    None
    """
    
    # Plot optimization history
    fig_optimization_history = optuna.visualization.plot_optimization_history(study)
    fig_optimization_history.update_layout(
        title="Optimization history plot",
        xaxis_title="Step (Num of trial)",
        yaxis_title="Objective value (RMSE)",
        legend_title="Legend",
        # width=width,
        # height=height,
    )
    fig_optimization_history.write_html(f"{save_dir}plot_optimization_history.html")
    #fig_optimization_history.show(close=True)

    # Plot intermediate values
    fig_intermediate_values = optuna.visualization.plot_intermediate_values(study)
    fig_intermediate_values.update_layout(
        title="Intermediate Objective Values (RMSE) per Trial",
        xaxis_title="Step (number of rounds or iterations within a trial)",
        yaxis_title="Objective value (RMSE)",
        legend_title="Trial Number",
        # width=width,
        # height=height,        
    )
    fig_intermediate_values.write_html(f"{save_dir}plot_intermediate_values.html")
    #fig_intermediate_values.show(close=True)

    # Plot hyperparameter importance
    fig_param_importances = optuna.visualization.plot_param_importances(
        study,
        evaluator=MeanDecreaseImpurityImportanceEvaluator(),
    )
    fig_param_importances.update_layout(
        title="Hyperparameter importance",
        xaxis_title="SHAP Importance",
        yaxis_title="Hyperparameter",
        # width=width,
        # height=height,        
    )
    fig_param_importances.write_html(f"{save_dir}plot_param_importances.html")
    #fig_param_importances.show(close=True)

    # Compute hyperparameter importances and extract names ordered by importance
    importances = get_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    param_names_by_importance = list(importances.keys())

    # Plot slice of hyperparameters
    fig_slice = optuna.visualization.plot_slice(study)
    fig_slice.update_layout(
        title="slice_plot",
        # xaxis_title="Objective value",
        # yaxis_title="Hyperparameter",
        # width=width,
        # height=height, 
    )
    fig_slice.write_html(f"{save_dir}plot_slice_plot.html")
    #fig_slice.show(close=True)

    print("\nTraining history saved successfully!")
    
    return


# execution 
if __name__ == "__main__":
    
    # create export folder to store outputs
    # e.g. create folder "../run/exp"
    def create_directory(base_path = "../run/cv"):
    
        counter = 0
    
        while os.path.exists(base_path + (str(counter) if counter > 0 else "") + "/"):
            counter += 1

        new_directory = base_path + (str(counter) if counter > 0 else "") + "/"
        os.makedirs(new_directory)
        print(f"\n******Created directory: {new_directory}")
    
        return new_directory

    directory = create_directory()

    # define columns in model training
    cols = ["zmax", "zmeam", "zsd", "zskew", "zkurt", "zentropy", "pzabovezmean", "Pzabovex", 
            "zq10", "zq20" ,"zq30", "zq40", "zq50", "zq60", "zq70", "zq80", "zq90", "zq95",
            "Zpcum10", "Zpcum20", "Zpcum30", "Zpcum40", "Zpcum50", 
            "Zpcum60", "Zpcum70", "Zpcum80", "Zpcum90", "Zpcum95", 
            "idot", "imax", "imean", "istd', "iskew", "ikurt", "ipground",
            "Ipcumzq10", "Ipcumzq30","Ipcumzq50","Ipcumzq70","Ipcumzq90",
            "Ipsth10", "Ipsth50", "Ipsth70", "Ipsth90","Ipsth95",
            "pxth30", "pxth60", "pxth90",
            "pground"
        
    ]
    # load data
    csv_path = "../data/train_df.csv"  # "../data/INFyS_ts_name.csv"
    X_train, y_train, dtrain = prepare_DMatrix_array(csv_path, cols, directory)

    # plot hist
    plot_target_histograms(y_train, directory)

    # training starts
    starttime = datetime.now()

    # # Set up the Optuna study using the TPESampler and MedianPruner
    # study = optuna.create_study(
    #     # sampler=optuna.samplers.RandomSampler(seed=42),
    #     sampler=optuna.samplers.TPESampler(n_startup_trials=100),
    #     # sampler=optunahub.load_module( "samplers/auto_sampler").AutoSampler(n_startup_trials=100), 
    #     # pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=100),
    #     direction='minimize'
    # )
    
    # Set up the Optuna study using the TPESampler and MedianPruner
    study = optuna.create_study(
        # sampler=optuna.samplers.RandomSampler(seed=42),
        sampler=optuna.samplers.TPESampler(n_startup_trials=100),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=100),
        direction='minimize'
    )
    # optimization
    from functools import partial
    study.optimize(partial(objective, save_dir=directory), n_trials=2000)
    
    #end_time
    endtime = datetime.now()
    used_time = endtime - starttime
    
    print(f"\n--xgb training time--: {used_time }")

    # train and test rmse in best trial
    get_best_trial_metrics(study, dtrain, directory)
    
    # save scatterplot of test, prediection in dataframe using trained model
    final_model= save_best_results_and_train_model(study, X_train, y_train, directory)     
    
    # load test datasets from database
    test = pd.read_csv("../data/test_df.csv")
    X_test = test[cols] 
    y_test = test["AGBD"]
    dtest= xgb.DMatrix(X_test,  label=y_test)
    
    # save train and test scatter plot
    for tipe in['train', 'test']:
        if tipe == 'train':
            y = y_train
            x = dtrain
        else:  ## type=='test'
            y = y_test
            x = dtest
        evaluate_and_plot(final_model, y, x, tipe, directory)
    
    # save SHAP 
    compute_and_plot_shap(final_model, X_train, directory)
    
    # save htlm plots of traing history 
    plot_optuna_results(study, directory)
    print("\nExecution done!")




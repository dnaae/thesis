# %%
import os
import sys
import logging
from pathlib import Path

# Numeric/Stats imports
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Modelling imports
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

# Neuroimaging imports
from nipype.algorithms.confounds import FramewiseDisplacement
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

###############################################################################
# SET PATHS AND GLOBALS
###############################################################################
infolder= Path('...')
working_dir = infolder / 'code'
os.chdir(working_dir)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

n_repetitions = 500

###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def standardise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises numerical columns in the given DataFrame, leaving categorical
    columns unchanged. Returns a new DataFrame with scaled columns.
    """
    df = df.copy()
    numerical_columns = df.select_dtypes(include=['float', 'int']).columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df


def fit_nuisance_regression(
    train_data: pd.DataFrame,
    conn_vars: pd.Index
) -> dict:
    """
    Fits a nuisance regression model (OLS) for each connectivity feature
    using training data. Returns a dictionary of fitted models keyed by the
    connectivity variable.

    Parameters
    ----------
    train_data : pd.DataFrame
        DataFrame containing connectivity features and confound variables.
    conn_vars : pd.Index
        Columns corresponding to connectivity features to be modelled.

    Returns
    -------
    model_dict : dict
        Dictionary of statsmodels OLS results, keyed by conn_var name.
    """
    formula = ' + '.join([
        'Age',
        'C(Gender)',
        'Age*C(Gender)',
        'FS_BrainSeg_Vol',
        'average_FD'
    ])
    # For each conn_var, fit an OLS model: conn_var ~ confounds
    model_dict = {
        conn_var: smf.ols(
            f"{conn_var} ~ {formula}", data=train_data
        ).fit()
        for conn_var in conn_vars
    }
    return model_dict


def apply_nuisance_regression(
    test_data: pd.DataFrame,
    model_dict: dict,
    conn_vars: pd.Index
) -> pd.DataFrame:
    """
    Applies previously fitted nuisance regression models to 'test_data' to
    produce residuals for each connectivity variable.

    Parameters
    ----------
    test_data : pd.DataFrame
        DataFrame containing connectivity features (and confounds, though
        only confounds are used for prediction) for the test set.
    model_dict : dict
        Dictionary of statsmodels OLS results from training.
    conn_vars : pd.Index
        Columns corresponding to connectivity features to be corrected.

    Returns
    -------
    residuals_df : pd.DataFrame
        DataFrame of residuals, one column per connectivity variable.
    """
    test_data = test_data.copy()
    residuals = {}
    for conn_var, model in model_dict.items():
        # Subtract the model's predicted values from the observed values
        residuals[conn_var] = test_data[conn_var] - model.predict(test_data)
    residuals_df = pd.DataFrame(residuals, index=test_data.index)
    return residuals_df


def mk_kfold_indices(
    subj_list: pd.Index,
    k: int = 10
) -> np.ndarray:
    """
    Splits list of subjects into k folds for cross-validation. Returns an
    array of fold indices of length len(subj_list).
    """
    subj_list = list(subj_list)
    n_subs = len(subj_list)
    n_subs_per_fold = n_subs // k  # floor integer division
    remainder = n_subs % k

    # Create a repeated pattern of fold assignments
    indices_list = [[fold_no] * n_subs_per_fold for fold_no in range(k)]
    indices = [elem for sublist in indices_list for elem in sublist]

    # Distribute the remainder
    remainder_indices = list(range(remainder))
    indices.extend(remainder_indices)

    # Check we have one index per subject
    assert len(indices) == n_subs, (
        "Length of indices list does not match number of subjects."
    )
    np.random.shuffle(indices)
    return np.array(indices)


def split_train_test(
    subj_list: pd.Index,
    indices: np.ndarray,
    test_fold: int
) -> tuple[list, list]:
    """
    Given a subj_list, an array of fold indices, and the integer 'test_fold',
    return two lists: train_subs and test_subs.
    """
    train_inds = np.where(indices != test_fold)[0]
    test_inds = np.where(indices == test_fold)[0]
    train_subs = subj_list[train_inds].to_list()
    test_subs = subj_list[test_inds].to_list()
    return train_subs, test_subs


def get_train_test_data(
    all_fc_data: pd.DataFrame,
    train_subs: list,
    test_subs: list,
    behav_data: pd.DataFrame,
    behav: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extracts the required FC and behavioural data for the given train/test subject lists.
    Returns: (training FC, training behaviour, test FC).
    """
    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]
    train_behav = behav_data.loc[train_subs, behav]
    return train_vcts, train_behav, test_vcts


def select_features(
    train_vcts: pd.DataFrame,
    train_behav: pd.Series,
    r_thresh: float = 0.2,
    corr_type: str = 'pearson',
    verbose: bool = False
) -> dict[str, np.ndarray]:
    """
    Runs the CPM feature selection step by correlating each connectivity
    edge with behaviour. Returns masks for positively and negatively
    correlated edges above the threshold.

    Parameters
    ----------
    train_vcts : pd.DataFrame
        Connectivity data for training.
    train_behav : pd.Series
        Behavioural data for training.
    r_thresh : float
        Threshold for |correlation| for selecting features.
    corr_type : str
        'pearson' or 'spearman'.
    verbose : bool
        Whether to print summary info.

    Returns
    -------
    mask_dict : dict
        Dictionary with Boolean arrays under keys 'pos' and 'neg'.
    """
    assert train_vcts.index.equals(train_behav.index), (
        "Indices of FC vectors and behaviour do not match!"
    )

    # Mean-centre to get faster covariance-based correlation
    y = train_behav.values - train_behav.mean()
    X = train_vcts.values - train_vcts.values.mean(axis=0)

    if corr_type == 'pearson':
        # Equivalent to corr(x,y) for each edge
        cov = np.dot(y.T, X) / (train_behav.shape[0] - 1)
        corr = cov / np.sqrt(
            np.var(train_behav, ddof=1) * np.var(train_vcts, axis=0, ddof=1)
        )
    elif corr_type == 'spearman':
        corr_list = []
        for edge in train_vcts.columns:
            r_val = spearmanr(train_vcts[edge], train_behav)[0]
            corr_list.append(r_val)
        corr = np.array(corr_list)
    else:
        raise ValueError("corr_type must be 'pearson' or 'spearman'.")

    mask_dict = {}
    mask_dict["pos"] = corr > r_thresh
    mask_dict["neg"] = corr < -r_thresh

    if verbose:
        logging.info(
            "Found (%d/%d) edges pos/neg correlated with behaviour > r_thresh=%g",
            mask_dict["pos"].sum(),
            mask_dict["neg"].sum(),
            r_thresh
        )

    return mask_dict


def build_model(
    train_vcts: pd.DataFrame,
    mask_dict: dict[str, np.ndarray],
    train_behav: pd.Series
) -> dict:
    """
    Builds the linear CPM model by summing connectivity edges in each mask
    (pos/neg) and regressing on behaviour. Also fits a 'glm' combining both.

    Returns a dict of slope/intercept for each tail and the combined 'glm'.
    """
    assert train_vcts.index.equals(train_behav.index), (
        "Indices of FC and behaviour do not match!"
    )

    model_dict = {}

    # Prepare for the GLM combining pos and neg features
    X_glm = np.zeros((train_vcts.shape[0], len(mask_dict.keys())))

    # Each tail: sum edges + linear regression on behaviour
    for t, (tail, mask) in enumerate(mask_dict.items()):
        # Sum of edges in the chosen mask
        X = train_vcts.iloc[:, mask.values].sum(axis=1).values
        y = train_behav.values

        slope, intercept = np.polyfit(X, y, 1)
        model_dict[tail] = (slope, intercept)

        X_glm[:, t] = X

    # Add a constant column
    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    # Fit a 3-parameter model [pos_slope, neg_slope, intercept]
    glm_coef = np.linalg.lstsq(X_glm, y, rcond=None)[0]
    # Return as a tuple
    model_dict["glm"] = tuple(glm_coef)

    return model_dict


def apply_model(
    test_vcts: pd.DataFrame,
    mask_dict: dict[str, np.ndarray],
    model_dict: dict
) -> dict[str, np.ndarray]:
    """
    Applies the fitted linear CPM model to test data, returning predictions
    for each tail plus the combined 'glm'.

    Returns a dictionary with keys 'pos', 'neg', and 'glm'.
    """
    behav_pred = {}
    n_tails = len(mask_dict.keys())

    X_glm = np.zeros((test_vcts.shape[0], n_tails))

    for t, (tail, mask) in enumerate(mask_dict.items()):
        X = test_vcts.iloc[:, mask.values].sum(axis=1).values
        slope, intercept = model_dict[tail]
        behav_pred[tail] = slope * X + intercept
        X_glm[:, t] = X

    # Combine for the GLM
    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    glm_coefs = model_dict["glm"]
    behav_pred["glm"] = np.dot(X_glm, glm_coefs)

    return behav_pred


def cpm_wrapper(
    all_fc_data: pd.DataFrame,
    all_behav_data: pd.DataFrame,
    behav: str,
    k: int = 10,
    **cpm_kwargs
):
    """
    Coordinates the Cross-Validated Predictive Modelling (CPM) steps across k folds.

    1. Splits subjects into k folds.
    2. For each fold:
       - Fit nuisance regression on training data.
       - Regress out nuisance from train/test FC data.
       - Select features (edges) that pass the threshold.
       - Fit model (pos tail, neg tail, combined 'glm').
       - Predict on the test fold.
    3. Collect predictions and masks across folds.

    Returns
    -------
    behav_obs_pred, all_masks, model_dicts
    """
    # Basic index checks
    assert all_fc_data.index.equals(all_behav_data.index), (
        "Row indices of FC vectors and behaviour do not match!"
    )

    subj_list = all_fc_data.index
    indices = mk_kfold_indices(subj_list, k=k)

    # Optional confounds (extend here if desired)
    confound_vars = [
        'Age',
        'Gender',
        'FS_IntraCranial_Vol',
        'FS_BrainSeg_Vol',
        'average_FD'
    ]
    conn_vars = all_fc_data.columns

    # DataFrame for storing predictions across all folds
    col_list = [
        f"{behav} predicted ({tail})"
        for tail in ["pos", "neg", "glm"]
    ] + [f"{behav} observed"]

    behav_obs_pred = pd.DataFrame(index=subj_list, columns=col_list)

    # Arrays for storing feature masks across folds
    n_edges = all_fc_data.shape[1]
    all_masks = {
        "pos": np.zeros((k, n_edges)),
        "neg": np.zeros((k, n_edges))
    }
    model_dicts = []

    # k-fold cross-validation
    for fold in range(k):
        train_subs, test_subs = split_train_test(subj_list, indices, test_fold=fold)
        train_vcts, train_behav, test_vcts = get_train_test_data(
            all_fc_data, train_subs, test_subs, all_behav_data, behav=behav
        )

        # Combine train FC + confounds, standardise
        train_data = pd.concat([train_vcts, all_behav_data.loc[train_subs, confound_vars]], axis=1)
        train_data = standardise_data(train_data)

        # Combine test FC + confounds, standardise
        test_data = pd.concat([test_vcts, all_behav_data.loc[test_subs, confound_vars]], axis=1)
        test_data = standardise_data(test_data)

        # Fit nuisance models to training data, then apply to train/test
        nuisance_models = fit_nuisance_regression(train_data, conn_vars)
        train_vcts_resid = apply_nuisance_regression(train_data, nuisance_models, conn_vars)
        test_vcts_resid = apply_nuisance_regression(test_data, nuisance_models, conn_vars)

        # Feature selection and model building
        mask_dict = select_features(train_vcts_resid, train_behav, **cpm_kwargs)
        model_dict = build_model(train_vcts_resid, mask_dict, train_behav)

        # Predict behaviour in the test fold
        behav_pred = apply_model(test_vcts_resid, mask_dict, model_dict)

        # Store predictions
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, f"{behav} predicted ({tail})"] = predictions

        # Save masks and models for later analysis
        all_masks["pos"][fold, :] = mask_dict["pos"]
        all_masks["neg"][fold, :] = mask_dict["neg"]
        model_dicts.append(model_dict)

    # Store the observed behaviour
    behav_obs_pred.loc[subj_list, f"{behav} observed"] = all_behav_data[behav]

    return behav_obs_pred, all_masks, model_dicts


def plot_predictions(behav_obs_pred: pd.DataFrame, tail: str = "glm") -> sns.axisgrid.FacetGrid:
    """
    Convenience function to generate a regression plot of observed vs predicted
    behaviour for the specified tail (pos, neg, or glm).
    """
    x = behav_obs_pred.filter(regex=("obs")).astype(float).values.squeeze()
    y = behav_obs_pred.filter(regex=(tail)).astype(float).values.squeeze()

    g = sns.regplot(x=x, y=y, color='gray')
    ax_min = min(g.get_xlim()[0], g.get_ylim()[0])
    ax_max = max(g.get_xlim()[1], g.get_ylim()[1])
    g.set_xlim(ax_min, ax_max)
    g.set_ylim(ax_min, ax_max)
    g.set_aspect('equal', adjustable='box')

    r_val = pearsonr(x, y)[0]
    g.annotate(f'r = {r_val:.2f}', xy=(0.7, 0.1), xycoords='axes fraction')
    return g

def run_cpm_repetition(rep, X_train_df, y_train_df, behaviour, cpm_kwargs):
    """
    Runs one repetition of CPM, returning summary metrics like:
    - Pearson correlation (x vs y)
    - R^2, RMSD, MAE, MSE on validation set
    - Averaged positive/negative masks
    - Model parameter means
    """
    try:
        behav_obs_pred, all_masks, model_dicts = cpm_wrapper(
            X_train_df, y_train_df, behaviour, k=5, **cpm_kwargs
        )

        # Correlation in the cross-validated predictions
        x = behav_obs_pred.filter(regex=("obs")).astype(float).values.squeeze()
        y = behav_obs_pred.filter(regex=("pos")).astype(float).values.squeeze()

        r_val, p_val = pearsonr(x, y)

        # Calculate metrics for the validation predictions (cross-validation set)
        r2_val = r2_score(x, y)
        rmsd_val = np.sqrt(np.mean((x - y) ** 2))
        mae_val = mean_absolute_error(x, y)
        mse_val = mean_squared_error(x, y)

        # Collect model parameters
        new_model = pd.DataFrame(
            columns=['pos_param1', 'pos_param2',
                    'neg_param1', 'neg_param2',
                    'glm_param1', 'glm_param2', 'glm_param3']
        )
        for idx, model in enumerate(model_dicts):
            new_model.loc[idx, 'pos_param1'] = model['pos'][0]
            new_model.loc[idx, 'pos_param2'] = model['pos'][1]
            new_model.loc[idx, 'neg_param1'] = model['neg'][0]
            new_model.loc[idx, 'neg_param2'] = model['neg'][1]
            # glm has 3 parameters: pos_slope, neg_slope, intercept
            new_model.loc[idx, 'glm_param1'] = model['glm'][0]
            new_model.loc[idx, 'glm_param2'] = model['glm'][1]
            new_model.loc[idx, 'glm_param3'] = model['glm'][2]

        return {
            'rep': rep,
            'r': r_val,
            'p':p_val,
            'r2_val': r2_val,
            'rmsd_val': rmsd_val,
            'mae_val': mae_val,
            'mse_val': mse_val,
            'posmask': all_masks['pos'].mean(axis=0),
            'negmask': all_masks['neg'].mean(axis=0),
            'model_params': new_model.mean(axis=0).to_dict(),
        }
    except Exception as e:
        logging.error("Error in rep %d: %s", rep, e)
        return None
###############################################################################
# MAIN SCRIPT LOGIC
###############################################################################

# Load merged data
data = pd.read_csv(infolder / 'johnny_data.csv')
data.rename(columns={'Unnamed: 0': 'subject'}, inplace=True)
data.set_index('subject', inplace=True)

# Split into connectivity (X) vs behaviour/demographics (y)
X_data = data[[c for c in data.columns if c.startswith('conn_')]]
y_data = data[[c for c in data.columns if not c.startswith('conn_')]]

# Train/test split
train_ind, test_ind = train_test_split(
    np.arange(X_data.shape[0]),
    test_size=0.2, random_state=42
)
X_train_df, y_train_df = X_data.iloc[train_ind], y_data.iloc[train_ind]

cpm_kwargs = {'r_thresh': 0.05, 'corr_type': 'pearson'}

# %%
results = Parallel(n_jobs=25)(
    delayed(run_cpm_repetition)(
        rep, X_train_df, y_train_df, 'PMAT24_A_CR', cpm_kwargs
    )
    for rep in tqdm(range(n_repetitions), desc="CPM Repetitions")
)

# Drop failed reps
results = [res for res in results if res is not None]
results_df = pd.DataFrame(results)
#%%
print(results_df)
# %%
# Quick diagnostic: histogram of cross-val correlations
sns.histplot(results_df['r'], kde=True)
plt.axvline(x=results_df['r'].mean(), color='red')
plt.title('Distribution of cross-validated correlations over repetitions')
plt.xlabel('Pearson r')
plt.ylabel('Frequency')
plt.show()

# Extract average masks and model params
posmasks = np.array([entry['posmask'] for entry in results])
negmasks = np.array([entry['negmask'] for entry in results])
model_params_list = [entry['model_params'] for entry in results]

# Mean across repetitions
posmasks_mean = posmasks.mean(axis=0)
negmasks_mean = negmasks.mean(axis=0)

# You used > 0.9 as a threshold for "consistently selected"
new_posmasks = (posmasks_mean > 0.9)
new_negmasks = (negmasks_mean > 0.9)
new_mask_dict = {
    'pos': pd.Series(new_posmasks, index=X_train_df.columns),
    'neg': pd.Series(new_negmasks, index=X_train_df.columns)
}
print(new_mask_dict)
# Average model parameters
model_params_df = pd.DataFrame(model_params_list)
model_params_avg = model_params_df.mean(axis=0)
new_model_dict = {
    'pos': (model_params_avg['pos_param1'], model_params_avg['pos_param2']),
    'neg': (model_params_avg['neg_param1'], model_params_avg['neg_param2']),
    'glm': (
        model_params_avg['glm_param1'],
        model_params_avg['glm_param2'],
        model_params_avg['glm_param3']
    )
}

# Evaluate in the held-out test set
X_test_df = X_data.iloc[test_ind, :]
y_test_df = y_data.iloc[test_ind]

# Nuisance regression on entire training set
train_data = pd.concat([X_train_df, y_train_df], axis=1)
train_data = standardise_data(train_data)
nuisance_model = fit_nuisance_regression(train_data, X_train_df.columns)
train_vcts_resid = apply_nuisance_regression(train_data, nuisance_model, X_train_df.columns)

# Prepare the test set
test_data = pd.concat([X_test_df, y_test_df], axis=1)
test_data = standardise_data(test_data)
test_vcts_resid = apply_nuisance_regression(test_data, nuisance_model, X_test_df.columns)

# Predictions
for prediction_set, y_df in zip([train_vcts_resid, test_vcts_resid], [y_train_df, y_test_df]):

    # Final predictions
    behav_pred = apply_model(prediction_set, new_mask_dict, new_model_dict)
    test_behav = y_df['PMAT24_A_CR']

    # Merge predictions with observed
    prediction_df = pd.DataFrame({
        'PMAT24_A_CR': test_behav
    })
    for tail, values in behav_pred.items():
        prediction_df[tail] = values

    # Plot
    plt.figure(figsize=(10, 3))
    for i, tail in enumerate(['neg', 'pos', 'glm'], start=1):
        plt.subplot(1, 3, i)
        sns.regplot(
            x=tail, y='PMAT24_A_CR', data=prediction_df, scatter_kws={'alpha': 0.6}
        )
        plt.title(f"{tail.upper()} predictions")
    plt.tight_layout()
    plt.show()

    # Calculate R²
    r2_scores = {}
    for tail in ['neg', 'pos', 'glm']:
        r2_scores[tail] = sp.stats.pearsonr(prediction_df[tail], prediction_df['PMAT24_A_CR'])[0] ** 2
        if sp.stats.pearsonr(prediction_df[tail], prediction_df['PMAT24_A_CR'])[0] < 0:
            r2_scores[tail] = -r2_scores[tail]

    for tail, r2 in r2_scores.items():
        print(f"R² for {tail.upper()}: {r2:.3f}")



# %%
r_test = pearsonr(test_behav, behav_pred['neg'])[0]
print(r_test)

# %%
results_df
#%%
# Histogram of p-values
sns.histplot(results_df['p'], kde=True)
plt.axvline(x=0.05, color='red', linestyle='--', label='Significance threshold (p=0.05)')
plt.title('Distribution of p-values for validation Pearson r over repetitions')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#%%# Save the final masks
# Save the mask dict as CSV files
# new_mask_dict['pos'].to_csv('final_posmask.csv', header=True)
# new_mask_dict['neg'].to_csv('final_negmask.csv', header=True)

# %%
from sklearn.metrics import mean_squared_error
import numpy as np

#Extra plot: test predictions with significance 
# Extract predictions for the 'glm' model (or others like 'pos' or 'neg')
predicted_values = behav_pred['neg']  # Or 'neg', 'pos' as needed

# Calculate Pearson correlation and p-value for the test set
r_test, p_test = pearsonr(test_behav, predicted_values)

# Calculate R² for the test set
r2 = r2_score(test_behav, predicted_values)

# Calculate RMSE for the test set
rmse = np.sqrt(mean_squared_error(test_behav, predicted_values))

# Output the results
print(f"Test Set Results:")
print(f"  Pearson r: {r_test:.3f}")
print(f"  p-value: {p_test:.3e}")
print(f"  R²: {r2:.3f}")
# print(f"  RMSE: {rmse:.3f}")

# Plot with R², p-value, and RMSE included
plt.figure(figsize=(12, 4))
for i, tail in enumerate(['neg', 'pos', 'glm'], start=1):
    plt.subplot(1, 3, i)
    sns.regplot(
        x=prediction_df[tail], y=prediction_df['PMAT24_A_CR'], scatter_kws={'alpha': 0.6}
    )
    
    # Calculate metrics for each model
    r, p = pearsonr(prediction_df[tail], prediction_df['PMAT24_A_CR'])
    rmse_tail = np.sqrt(mean_squared_error(prediction_df['PMAT24_A_CR'], prediction_df[tail]))
    
    # Title with metrics
    plt.title(f"{tail.upper()} Predictions\n"
              f"r = {r:.2f}, p = {p:.3e}")
    plt.xlabel(f"{tail.upper()} Predicted")
    plt.ylabel("Observed PMAT24_A_CR")

plt.tight_layout()
plt.show()


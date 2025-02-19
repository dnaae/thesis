# %%
#Imports
import os
import sys
import logging
from pathlib import Path

# Numeric/Stats imports
import numpy as np
import pandas as pd

# Modelling imports
import statsmodels.formula.api as smf

# Neuroimaging imports
from nipype.algorithms.confounds import FramewiseDisplacement
from nilearn.connectome import sym_matrix_to_vec

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

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def calculate_avgFD(movement_file: Path) -> float:
    """
    Calculates the average framewise displacement (FD) for a movement parameter file.
    The 'movement_file' is expected to be in FSFAST format and contain 6 columns.
    """
    fdinterface = FramewiseDisplacement()
    fdinterface.inputs.in_file = str(movement_file)
    fdinterface.inputs.parameter_source = 'FSFAST'
    outfile = movement_file.parent / 'FD_plot.png'
    fdinterface.inputs.out_file = str(outfile)
    fdinterface.inputs.save_plot = False
    fdinterface_outputs = fdinterface.run()
    avgFD = fdinterface_outputs.outputs.fd_average
    return avgFD

#%%
###############################################################################
# MAIN SCRIPT LOGIC
###############################################################################
if __name__ == "__main__":

    ############################################################################
    # 1. Calculate the average framewise displacement for each subject
    ############################################################################
    # movement_folder = infolder / 'data/movement_anonymised'
    movement_folder = infolder / 'movement_anonymised'
    #movement_outfolder = infolder / 'data/movement_cleaned'
    movement_outfolder = infolder / 'movement_cleaned'
    os.makedirs(movement_outfolder, exist_ok=True)

    movement_files = [
        f for f in os.listdir(movement_folder) if f.endswith('.txt')
    ]
    FDs = []

    for i, movement_file in enumerate(movement_files, start=1):
        logging.info("Processing file %d of %d: %s", i, len(movement_files), movement_file)
        try:
            movement_params = pd.read_csv(
                movement_folder / movement_file,
                sep=r'\s+',
                header=None
            )
            # Only keep the first 6 columns (the second 6 are the derivatives)
            movement_params = movement_params.iloc[:, 0:6]
            movement_params.columns = [
                'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'
            ]
            movement_outfile = movement_outfolder / movement_file.replace('.txt', '_cleaned.csv')
            movement_params.to_csv(
                movement_outfile,
                header=None,
                sep='\t',
                index=False
            )
            avgFD = calculate_avgFD(movement_outfile)
            # Example naming convention: session_encoding_subject.txt
            filename_parts = movement_file.split('_')

            FDs.append({
                'session': filename_parts[0],
                'encoding': filename_parts[1] if len(filename_parts) > 2 else np.nan,
                'subject': filename_parts[2] if len(filename_parts) > 2 else np.nan,
                'FD': avgFD
            })
        except Exception as e:
            logging.warning("Error with file %s: %s", movement_file, e)

    fd_df = pd.DataFrame(FDs)
    # Average over encoding directions and runs
    fd_df = fd_df.groupby('subject')['FD'].mean().reset_index(name='average_FD')
    fd_df.to_csv(infolder / 'FDs.csv', index=False)

    ############################################################################
    # 2. Combining netmats and FDs
    ############################################################################
    ids = pd.read_csv(infolder / 'MRI_IDs_anonymised.csv')['MRI_ID'].values
    mats = pd.read_csv(infolder / 'netmats15.txt', sep=r'\s+', header=None).set_index(ids)
    fd_df = pd.read_csv(infolder / 'FDs.csv').set_index('subject')
    prs_df = pd.read_csv("/prssocial.csv").rename(columns={"SocialScore": "social_cognition_scores", "Subject": "subject"})
    prs_df.set_index("subject", inplace=True)

    # Extract upper triangle from each matrix
    triu_mats = []
    n_rows = mats.shape[0]
    # Each row is a flattened connectivity matrix
    for i, row in mats.iterrows():
        mat_2d = row.values.reshape(
            int(np.sqrt(row.values.size)),
            int(np.sqrt(row.values.size))
        )
        mat_triu = sym_matrix_to_vec(mat_2d, discard_diagonal=True)
        triu_mats.append(mat_triu)
    # Create DataFrame with flattened matrices
    mats_df = pd.DataFrame(triu_mats, index=mats.index)
    mats_df.rename(columns={i: f'conn_{i+1}' for i in range(mats_df.shape[1])}, inplace=True)

    # Filter FD and PRS data to match with mats_df
    common_subjects = mats_df.index.intersection(fd_df.index).intersection(prs_df.index)
    mats_df = mats_df.loc[common_subjects]
    filtered_fd_df = fd_df.loc[common_subjects]
    filtered_prs_df = prs_df.loc[common_subjects]

    #Multiply connectivity columns by PRS scores
    prs_scores = filtered_prs_df['Pt_0.0287001']
    conn_columns = [col for col in mats_df.columns if col.startswith('conn_')]
    for col in conn_columns:
        mats_df[col] += prs_scores

    # Merge FD data with modified connectivity data
    merged_df = pd.merge(filtered_fd_df, mats_df, left_index=True, right_index=True, how='right')

    # Log number of subjects
    logging.info("Number of subjects with imaging IDs, PRS, and FD data: %d", merged_df.shape[0])

    # Merge merged_df with social cognition scores
    merged_df = pd.merge(merged_df, prs_df[['social_cognition_scores']], left_index=True, right_index=True, how='left')

    print(merged_df)

# Add behavioural data
    behaviour_df = pd.read_csv(infolder / 'behavioural_data_anonymised.csv')
    behaviour_df.rename(columns={'Subject': 'subject'}, inplace=True)
    behaviour_df.set_index('subject', inplace=True)
    merged_df = pd.merge(
        merged_df, behaviour_df,
        left_index=True, right_index=True,
        sort=True, how='left'
    ).dropna()

    print(merged_df.columns.tolist)

    # Add phenotypic data (ensure no duplicates)
    phenotypic_df = pd.read_csv(infolder / 'phenotypic_data_anonymised.csv')
    phenotypic_df = phenotypic_df[['Subject', 'Age_in_Yrs']]  # Only extract relevant columns
    phenotypic_df.rename(columns={'Subject': 'subject'}, inplace=True)
    phenotypic_df.set_index('subject', inplace=True)

# Merge with merged_df based on 'subject' index
    merged_df = pd.merge(
        merged_df, phenotypic_df[['Age_in_Yrs']],  # Only merge Age_in_Yrs column
        left_index=True, right_index=True, sort=True, how='left'
    )

# Ensure Age column is replaced with Age_in_Yrs
    if 'Age' in merged_df.columns:
        merged_df.drop(columns=['Age'], inplace=True, errors='ignore')  # Drop existing 'Age' column
        merged_df.rename(columns={'Age_in_Yrs': 'Age'}, inplace=True)  # Rename 'Age_in_Yrs' to 'Age'

# Log information about the merge
    logging.info("Number of subjects with phenotypic data: %d", phenotypic_df.shape[0])
    logging.info("Final number of subjects with all data: %d", merged_df.shape[0])

    print(merged_df.columns.tolist())

merged_df.to_csv(infolder / 'merged_data.csv')


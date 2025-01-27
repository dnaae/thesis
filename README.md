Welcome to the repository of the project; "Beating Genetic Odds: Enhancing Social Cognition with Brain Compensation" 👋

Hey there! Welcome aboard! 🎉 You’re about to dive into an exciting pipeline that combines behavioral, genetic, and fMRI data—developed as part of my Bioinformatics MSc graduation project. 🚀

In this project, we used data from the Human Connectome Project, to investigate if genetic-moderated functional connectivity matrices are predictive of social cognition scores. This project was an attempt to reframe autism, in the context of brain compensation, based on the observation of neurotypical adults with a high genetic predisposition for autism who do not exhibit symptoms. The workflow entailed model selection for the extraction of social cognition scores using factor analysis, calculation of polygenic risk scores using PRSice and cross-validation for threshold selection, and behavioral predictions using connectome-based predictive modeling.

All the relevant scripts used for the analysis can be found in the code folder. 

First, the "social cognition scores" R script 1) filters participants based on exclusion criteria, 2) applies data preprocessing, 3) performs exploratory and confirmatory analysis for different combinations of behavioral metrics, and 4) calculates social cognition scores based on the best model.

Second, after running PRSice on genetic data (follow the command line and R tutorial by Choi, S.W., Mak, T.S. & O’Reilly, P.F. Tutorial: a guide to performing polygenic risk score analyses. Nat Protoc (2020). https://doi.org/10.1038/s41596-020-0353-1), the "prs best selection" R script finds the most predictive PRS threshold by 5-fold cv and linear regression on social cognition scores.

Third, the "CPM preprocessing" Python script performs the necessary data preprocessing steps for the preparation of functional connectivity matrices for CPM (frame-wise displacement correction, multiplication with PRS, and merging behavioral and phenotypic data into a dataframe). 

Fourth, the "CPM" Python script applies a CPM approach (based on Shen X, Finn ES, Scheinost D, Rosenberg MD, Chun MM, Papademetris X, Constable RT. Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nat Protoc. 2017 Mar;12(3):506-518. doi: 10.1038/nprot.2016.178. Epub 2017 Feb 9. PMID: 28182017; PMCID: PMC5526681) with the introduction of an inner-outer loop split, to ensure predictions on held-out data as well. 

Lastly, the "simple machine learning" Python script applies a baseline model for comparison with CPM.

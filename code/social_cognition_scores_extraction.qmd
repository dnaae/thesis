---
title: "Github"
format: html
editor: visual
---

```{r}
#install.packages("reticulate")
library(reticulate)
```

```{r}
#Function to select HCP subjects after applying exclusion criteria (rsfMRI+genetic data+unrelated)

py_run_string("
import pandas as pd
import numpy as np

def process_hcp_data(behavioural_file, phenotypic_file, measures, output_file):
    # Load datasets
    behavioural_data = pd.read_csv(behavioural_file, delimiter=',')
    phenotypic_data = pd.read_csv(phenotypic_file,delimiter=',')
    #print(behavioural_data)
    #print(phenotypic_data)
    # Replace empty strings with NA if they exist
    phenotypic_data[phenotypic_data == ''] = np.nan
    
    # Ensure 'Subject' is consistent in type
    behavioural_data['Subject'] = behavioural_data['Subject'].astype(str)
    phenotypic_data['Subject'] = phenotypic_data['Subject'].astype(str)
    
    # Merge datasets on 'Subject'
    merged_df = pd.merge(behavioural_data, phenotypic_data, on='Subject')
    
    # Filter participants with resting-state fMRI data
    columns_of_interest = ['3T_RS-fMRI_Count', '3T_RS-fMRI_PctCompl']
    merged_df = merged_df[(merged_df[columns_of_interest] != 0.0).all(axis=1)]
    
    # Filter participants with genetic data
    merged_df = merged_df[merged_df['HasGT'] == True]
    
    # Handle related family members by random sampling
    family_counts = merged_df['Family_ID'].value_counts()
    related_families = family_counts[family_counts > 1].index
    
    sampled_subjects = pd.DataFrame()
    for family_id in related_families:
        family_df = merged_df[merged_df['Family_ID'] == family_id]
        sampled_df = family_df.sample(n=1, random_state=1)
        sampled_subjects = pd.concat([sampled_subjects, sampled_df], ignore_index=True)
    
    unrelated_subjects = merged_df[~merged_df['Family_ID'].isin(related_families)]
    final_df = pd.concat([unrelated_subjects, sampled_subjects], ignore_index=True)
    
    # Select only the required measures
    final_df = final_df[measures]
    
    # Replace NaN values with np.nan
    final_df = final_df.replace({pd.NA: np.nan})
    
    # Save the processed DataFrame to a CSV file
    final_df.to_csv(output_file, index=False)
    
    # Return the processed DataFrame
    return final_df
")

```

```{r}
#Function to define different models based on combinations of behavioral metrics 

models <- list(
  no0bk = c("Subject", "Emotion_Task_Face_Median_RT", "Language_Task_Story_Median_RT", 
            "Social_Task_TOM_Median_RT_TOM", "ER40_CRT"),
  w0bk = c("Subject", "Emotion_Task_Face_Median_RT", "Language_Task_Story_Median_RT", 
           "Social_Task_TOM_Median_RT_TOM", "ER40_CRT", "WM_Task_0bk_Median_RT"),
  accuracies = c("Subject", "Language_Task_Story_Acc", "ER40_CR", "Emotion_Task_Face_Acc", 
                 "Social_Task_TOM_Perc_TOM"),
  questionnaires = c("Subject", "Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", 
                     "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj"),
  rtquestionnaires = c("Subject", "Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", 
                       "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj", 
                       "Emotion_Task_Face_Median_RT", "Language_Task_Story_Median_RT", 
                       "Social_Task_TOM_Median_RT_TOM", "ER40_CRT"),
  cumulrtquestionnaire= c("Subject", "Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj", 
                       "Emotion_Task_Face_Median_RT", "Language_Task_Story_Median_RT", 
                       "Social_Task_TOM_Median_RT_TOM", "ER40_CRT"),
  all= c("Subject", "Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", 
                       "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj", 
                       "Emotion_Task_Face_Median_RT", "Language_Task_Story_Median_RT", 
                       "Social_Task_TOM_Median_RT_TOM", "ER40_CRT","Language_Task_Story_Acc", "ER40_CR", "Emotion_Task_Face_Acc","Social_Task_TOM_Perc_TOM")
)

```

```{r}
#Function to create processed csv per model

process_model <- function(model_name, measures, behavioural_file, phenotypic_file, output_dir) {
  output_file <- file.path(output_dir, paste0(model_name, "_processed.csv"))
  processed_data <- py$process_hcp_data(behavioural_file, phenotypic_file, measures, output_file)
  
  # Convert NaN to NA in R
  processed_data[] <- lapply(processed_data, function(x) ifelse(is.nan(x), NA, x))
  
  # Return the processed data
  return(processed_data)
}
```

```{r}
#Implementation
# File paths
behavioural_file <- "/behavioural_data_anonymised.csv"
phenotypic_file <-"/phenotypic_data_anonymised.csv"
output_dir <- "..."

# Process all models
results <- lapply(names(models), function(model_name) {
  message("Processing model: ", model_name)
  process_model(model_name, models[[model_name]], behavioural_file, phenotypic_file, output_dir)
})
```

```{r}
#Main pipeline for social cognition score extraction

#Data preprocessing functions
# Load necessary libraries
library("caret")
library("corrplot")
library("dplyr")
library("GGally")
library("ggplot2")
library("lavaan")
library("mice")
library("psych")

# Set random seed
set.seed(123)

# Read and process data
load_and_process_data <- function(file_path) {
  df <- read.csv(file_path, stringsAsFactors = FALSE)
  df$Subject <- factor(df$Subject)
  print(df)
  return(df)
}

handle_extreme_missingness <- function(df) {
  # Calculate the total number of columns
  total_cols <- ncol(df)
  
  # Calculate the threshold for "extreme missingness" (50% of total columns missing)
  missing_threshold <- floor(total_cols * 0.5)
  
  # Identify rows where more than 50% of the features are missing
  extreme_missingness_rows <- rowSums(is.na(df)) > missing_threshold
  
  # Filter out rows with extreme missingness
  df_filtered <- df[!extreme_missingness_rows, ]
  
  # Print a summary of the operation
  cat("Removed", sum(extreme_missingness_rows), "rows with extreme missingness.\n")
  #print(df_filtered)
  return(df_filtered)
}

# Z-score the dataset
scale_data <- function(df_reduced) {
  df_scaled <- df_reduced %>%
    mutate(across(where(is.numeric), ~ ( . - mean(., na.rm = TRUE)) / sd(., na.rm = TRUE)))
  return(df_scaled)
}


# Detect and mark outliers as NA using IQR (without removing rows with NAs)
mark_outliers <- function(df_scaled) {
  df_na_outliers <- df_scaled %>%
    mutate(across(
      where(is.numeric),
      ~ ifelse(
        . >= quantile(., 0.25, na.rm = TRUE) - 1.5 * IQR(., na.rm = TRUE) &
          . <= quantile(., 0.75, na.rm = TRUE) + 1.5 * IQR(., na.rm = TRUE),
        ., NA
      )
    ))
  #print(df_na_outliers)
  return(df_na_outliers)
}



# Impute missing values using MICE
impute_missing_values <- function(df_na_outliers) {
  imputed <- mice(df_na_outliers %>% select(where(is.numeric)), m = 10, method = "pmm")
  df_imputed <- df_na_outliers
  df_imputed[, names(imputed$imp)] <- complete(imputed)
  #summary(imputed)
  # View all imputed datasets (m = 10)
  imputed_data <- imputed$imp
  # Print imputed datasets for debugging
  #print(imputed_data[[1]])  # First imputed dataset
  #print(imputed_data[[2]])  # Second imputed dataset

  print(df_imputed)
  return(df_imputed)
}

# Visualize the correlation matrix
visualize_correlation <- function(df) {
  M <- cor(df[ ,!colnames(df) %in% c('Subject')])
  corrplot(M, method = "number", type = "upper")
}

# Split the dataset into training and testing sets
split_data <- function(df, response_column) {
  trainIndex <- createDataPartition(df[[response_column]], p = .8, list = FALSE)
  df_train <- df[trainIndex, ]
  df_test <- df[-trainIndex, ]
  return(list(train = df_train, test = df_test))
}

#Factor analysis functions

# Run EFA on the training set
run_efa <- function(df_train, nfactors = 1) {
  efa <- fa(df_train[, !colnames(df_train) %in% c('Subject')], nfactors = nfactors, rotate = "none")
  print(efa)
  fa.diagram(efa)
}


# Run CFA on the test set
perform_cfa <- function(df_test, model) {
  cfa_model <- cfa(model, data = df_test)
  cfa_summary <- summary(cfa_model, fit.measures = TRUE)
  return(list(model = cfa_model, summary = cfa_summary))
}
```

```{r}
#Execution with model specific functions
execute_analysis <- function(file_path, model_type = NULL) {
  # Read and process data
  print('Reading file')
  df <- read.csv(file_path, stringsAsFactors = FALSE)
  df$Subject <- factor(df$Subject)
  # Adjust ER40_CR if model type is "accuracies"
  if (model_type == "accuracies") {
    df$ER40_CR <- (df$ER40_CR / 40) * 100
    print("Adjusted ER40_CR for accuracies model.")
  }
  # Adjust ER40_CR if model type is "accuracies"
  if (model_type == "all") {
    df$ER40_CR <- (df$ER40_CR / 40) * 100
    print("Adjusted ER40_CR for accuracies model.")
  }
  # Conditional processing based on model type
  if (model_type == "questionnaire") {
    # Invert variables for questionnaire model
    df$Loneliness_Unadj <- 100 - df$Loneliness_Unadj
    df$PercHostil_Unadj <- 100 - df$PercHostil_Unadj
    df$PercReject_Unadj <- 100 - df$PercReject_Unadj
  }
  if (model_type == "all") {
    # Invert variables for all model
    df$Loneliness_Unadj <- 100 - df$Loneliness_Unadj
    df$PercHostil_Unadj <- 100 - df$PercHostil_Unadj
    df$PercReject_Unadj <- 100 - df$PercReject_Unadj
  }
  if (model_type == "rtquestionnaire"){
  # Invert questionnaire variables
  df$Loneliness_Unadj <- 100 - df$Loneliness_Unadj
  df$PercHostil_Unadj <- 100 - df$PercHostil_Unadj
  df$PercReject_Unadj <- 100 - df$PercReject_Unadj
  }
  if (model_type == "cumulrtquestionnaire"){
  # Invert questionnaire variables
  df$Loneliness_Unadj <- 100 - df$Loneliness_Unadj
  df$PercHostil_Unadj <- 100 - df$PercHostil_Unadj
  df$PercReject_Unadj <- 100 - df$PercReject_Unadj
  # Create cumulative questionnaire score
  df$cumulative_questionnaire <- rowSums(df[, c("Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj")], na.rm = TRUE)
  
  # Remove individual questionnaire variables
  df <- df[, !(names(df) %in% c("Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj"))]
  
  # Divide by 6 to get average cumulative score
  df[, "cumulative_questionnaire"] <- df[, "cumulative_questionnaire"] / 6
  }
  
  # Remove rows with excessive missing values
  df_reduced <- handle_extreme_missingness(df)
  
  # Scale the data
  df_scaled <- scale_data(df_reduced)
  
  #Detect outliers
  df_na_outliers<-mark_outliers(df_scaled)
  
  # Impute missing values using MICE
  df_imputed <- impute_missing_values(df_na_outliers)
  print(df_imputed)
  
  # Conditional processing based on model type
  if (model_type == "w0bk") {
    df_imputed <- df_imputed %>%
      mutate(
        Emotion_Task_Face_Median_RT_Adj = Emotion_Task_Face_Median_RT - WM_Task_0bk_Median_RT,
        Language_Task_Story_Median_RT_Adj = Language_Task_Story_Median_RT - WM_Task_0bk_Median_RT,
        Social_Task_Median_RT_TOM_Adj = Social_Task_TOM_Median_RT_TOM - WM_Task_0bk_Median_RT,
        ER40_CRT_Adj = ER40_CRT - WM_Task_0bk_Median_RT
      ) %>%
      select(-Emotion_Task_Face_Median_RT, 
             -Language_Task_Story_Median_RT, 
             -Social_Task_TOM_Median_RT_TOM, 
             -ER40_CRT, 
             -WM_Task_0bk_Median_RT)
    response_column <- "ER40_CRT_Adj"
    
  } else if (model_type == "accuracies") {
    # Check if accuracy variables have variance
    accuracy_vars <- c("Language_Task_Story_Acc", "ER40_CR","Social_Task_TOM_Perc_TOM","Emotion_Task_Face_Acc")
    accuracy_variance <- apply(df_imputed[accuracy_vars], 2, var, na.rm = TRUE)
    
    # If any of the accuracy variables have zero variance, reject the model and stop
    if (any(accuracy_variance == 0)) {
      print("Warning: Accuracy variables have zero variance. Cannot proceed with 'accuracies' model.")
      return(NULL)  # Stop further execution
    }
    
    response_column <- "ER40_CR"
  } else if (model_type == "all") {
    # Check if accuracy variables have variance
    all_vars <- c("Language_Task_Story_Acc", "ER40_CR","Social_Task_TOM_Perc_TOM","Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", 
                       "PercReject_Unadj", "EmotSupp_Unadj", "InstruSupp_Unadj", 
                       "Emotion_Task_Face_Median_RT", "Language_Task_Story_Median_RT", 
                       "Social_Task_TOM_Median_RT_TOM", "ER40_CRT","Emotion_Task_Face_Acc")
    all_variance <- apply(df_imputed[all_vars], 2, var, na.rm = TRUE)
    
    # If any of the accuracy variables have zero variance, reject the model and stop
    if (any(all_variance == 0)) {
      print("Warning: Accuracy variables have zero variance. Cannot proceed with 'accuracies' model.")
      return(NULL)  # Stop further execution
    }
    
    response_column <- "ER40_CR"
  } else if (model_type == "no0bk") {
    response_column <- "ER40_CRT"
  } else if (model_type == "all") {
    response_column <- "ER40_CRT"
  } else if (model_type == "questionnaire") {
    # For the questionnaire-only model
    df_imputed <- df_imputed %>%
      select(
        Subject, # Retain Subject column for clarity
        Friendship_Unadj,
        Loneliness_Unadj,
        PercHostil_Unadj,
        PercReject_Unadj,
        EmotSupp_Unadj,
        InstruSupp_Unadj
      )
    response_column <- "Friendship_Unadj"  
    
  } else if (model_type == "rtquestionnaire") {
    response_column <- "ER40_CRT"
  } else if (model_type == "cumulrtquestionnaire") {
    response_column <- "ER40_CRT"
  } else {
    stop("Unknown model type. Please specify a valid model type.")
  }
  
  print(df_imputed)
  
  # Visualize correlation matrix
  visualize_correlation(df_imputed)
  
  # Split data into train and test sets based on response_column
  split_data_result <- split_data(df_imputed, response_column = response_column)
  df_train <- split_data_result$train
  df_test <- split_data_result$test
  
  print("===== Exploratory Factor Analysis (EFA) Results =====")
  # Run EFA on the training set
  run_efa(df_train, nfactors = 1)
  
  # Define CFA model based on model type
  if (model_type == "w0bk") {
    model <- 'SocialScore =~ Emotion_Task_Face_Median_RT_Adj + 
              Language_Task_Story_Median_RT_Adj + 
              Social_Task_Median_RT_TOM_Adj + 
              ER40_CRT_Adj'
  } else if (model_type == "accuracies") {
    model <- 'SocialScore =~ Language_Task_Story_Acc + 
              ER40_CR + 
              Social_Task_TOM_Perc_TOM'
  } else if (model_type == "no0bk") {
    model <- 'SocialScore =~ Emotion_Task_Face_Median_RT + 
              Language_Task_Story_Median_RT + 
              Social_Task_TOM_Median_RT_TOM + 
              ER40_CRT'
  } else if (model_type == "questionnaire"|model_type == "rtquestionnaire") {
    model <- 'SocialScore =~ Friendship_Unadj + 
              Loneliness_Unadj + 
              PercHostil_Unadj + 
              PercReject_Unadj + 
              EmotSupp_Unadj + 
              InstruSupp_Unadj'
  } else if (model_type == "cumulrtquestionnaire") {
    model <- 'SocialScore =~ Emotion_Task_Face_Median_RT + 
              Language_Task_Story_Median_RT + 
              Social_Task_TOM_Median_RT_TOM + 
              ER40_CRT'
  }
  print("===== Confirmatory Factor Analysis (CFA) Results =====")
  # Run CFA on the test set
  perform_cfa(df_test, model)
  
  # Fit the model to the whole data
  cfa_full <- cfa(model, data=df_imputed)
  print(summary(cfa_full, fit.measures = TRUE))  # Display summary of CFA
  
  # Calculate factor scores based on the CFA model
  fscores <- lavPredict(cfa_full, newdata=df_imputed)
  df_imputed[ , 'SocialScore'] <- fscores[, 'SocialScore']
  ggpairs(df_imputed[ ,!colnames(df_imputed) %in% c('Subject')])
  
 # Save the result
  write.csv(df_imputed[, c('Subject', 'SocialScore')], 
            file = sprintf("../%s_social_cognition_scores.csv", model_type))
}
```

```{r}
#Test all models 
#all metrics
execute_analysis("/all_processed.csv", model_type = "all")
#accuracies
execute_analysis("/accuracies_processed.csv", model_type = "accuracies")
#rt+questionnaires
execute_analysis("/rtquestionnaires_processed.csv", model_type = "rtquestionnaire")
#cumulative questionnaires + rts
execute_analysis("/cumulrtquestionnaire_processed.csv", model_type = "cumulrtquestionnaire")
#questionnaires
execute_analysis("/questionnaires_processed.csv", model_type = "questionnaire")
#w0bk
execute_analysis("/w0bk_processed.csv", model_type = "w0bk")
#no0bk
execute_analysis("/no0bk_processed.csv", model_type = "no0bk")
```

```{r}
#Example plot for the RT model without 0bk subtracted

#Load dataset
social_scores<-read.csv("/no0bk_social_cognition_scores.csv")

#Select column

# Create the plot
sc_density_histogram_plot <- ggplot(social_scores, aes(x = .data[[social]])) +
  # Add histogram with density scaling
  geom_histogram(aes(y = ..density..), binwidth = 0.1, fill = "blue", alpha = 0.4, color = "black") +
  # Add normal distribution curve using the data's mean and standard deviation
  stat_function(
    fun = dnorm, 
    args = list(mean = mean(social_scores[[social]], na.rm = TRUE), sd = sd(social_scores[[social]], na.rm = TRUE)),
    color = "darkgreen", size = 1.5
  ) +
  # Add vertical lines for reference
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_vline(xintercept = 0.5, color = "red", linetype = "dashed", size = 1) +
  geom_vline(xintercept = -0.5, color = "red", linetype = "dashed", size = 1) +
  
  labs(
    x = "Social Cognition Scores",
    y = "Density"
  ) +
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 10),  # Adjust interval breaks
    labels = scales::label_number(scale = 1)
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(size = 18, margin = margin(t = 10)),
    axis.title.y = element_text(size = 18, margin = margin(r = 10)),
    axis.line = element_line(color = "black", size = 1.2),
    plot.title = element_text(size = 20)
  )

# Print the plot
print(sc_density_histogram_plot)
```

---
title: "Github"
format: html
editor: visual
---

```{r}
# Load packages
library(dplyr)
library(caret)
library(data.table)
library(ggplot2)
```

```{r}
#Load PRS data (Neuro_Chip:data folder with PRSice output)
prs_data<- fread("/Neuro_Chip/Neuro_Chip.all_score")

#Load social cognition scores
social_cognition_data <-fread("/no0bk_social_cognition_scores.csv")

#Create merged dataframe
merged_data <- merge(social_cognition_data, prs_data, by.x = "Subject", by.y = "FID")
```

```{r}
#Cross-valiation and regression for best model selection

# Automatically extract p-value columns (all columns except FID and IID)
p_value_columns <- setdiff(names(prs_data), c("FID", "IID"))
# Initialize storage for results
model_comparison <- data.frame(Threshold = character(),
                               SSR = numeric(),
                               R2 = numeric())

# Number of folds for cross-validation
k_folds <- 5
folds <- createFolds(merged_data$SocialScore, k = k_folds, list = TRUE)

# Cross-validation loop
for (threshold in p_value_columns) {
  residuals_list <- numeric(length = nrow(merged_data))
  r2_list <- numeric(k_folds) # Store R2 values for each fold
  
  for (fold in 1:k_folds) {
    # Split data into training and test sets
    train_data <- merged_data[-folds[[fold]], ]
    test_data <- merged_data[folds[[fold]], ]
    
    # Scale the data based on training data
    train_mean <- mean(train_data[[threshold]], na.rm = TRUE)
    train_sd <- sd(train_data[[threshold]], na.rm = TRUE)
    
    # Apply scaling to training and test sets using training parameters
    train_data[[threshold]] <- (train_data[[threshold]] - train_mean) / train_sd
    test_data[[threshold]] <- (test_data[[threshold]] - train_mean) / train_sd
    
    # Dynamically construct the model formula 
    formula <- as.formula(substitute(SocialScore ~ threshold, 
                                     list(threshold = as.name(threshold))))
    # Fit the model
    model <- lm(formula, data = train_data)
    
    # Predict and calculate residuals for test set
    predictions <- predict(model, newdata = test_data)
    residuals_list[folds[[fold]]] <- test_data$SocialScore - predictions
    
    # Store R2 for this fold
    r2_list[fold] <- summary(model)$r.squared
  }
  
  # Sum of squared residuals (SSR)
  ssr <- sum(residuals_list^2)
  
  # Mean R2 for the cross-validation folds
  mean_r2 <- mean(r2_list)
  
  # Store results for this threshold
  model_comparison <- rbind(model_comparison, 
                            data.frame(Threshold = threshold, SSR = ssr, R2 = mean_r2))
}
```

```{r}
# Find the threshold with the highest R2
best_threshold <- model_comparison[which.max(model_comparison$SSR), "Threshold"]

#Or print more thresholds for comparison
# Sort results by R2 or SSR to get the best thresholds
sorted_results <- model_comparison[order(-model_comparison$R2), ]

# Print the top 2 thresholds
best_threshold <- sorted_results[1, "Threshold"]
second_best_threshold <- sorted_results[2, "Threshold"]

# Print results for debugging
print(paste("Best PRS threshold: ", best_threshold, " with SSR: ", sorted_results[1, "SSR"]))
print(paste("Second best PRS threshold: ", second_best_threshold, " with SSR: ", sorted_results[2, "SSR"]))

# Print the best threshold for debugging
print(paste("Best PRS threshold: ", best_threshold))
```

```{r}
#After selecting best threshold print regression statistics

best_threshold<-"Pt_0.0287001"
#Print regresssion metrics
best_formula <- as.formula(substitute(SocialScore ~ threshold, 
                                      list(threshold = as.name(best_threshold))))
best_model <- lm(best_formula, data = best_train_data)

# Print the summary of the best model to show coefficients, p-values, etc.
best_model_summary <- summary(best_model)

# Print out the coefficients (betas), p-values, and other regression statistics
cat("Regression Coefficients (Betas):\n")
print(best_model_summary$coefficients)

cat("\nR-squared: ", best_model_summary$r.squared, "\n")
cat("Adjusted R-squared: ", best_model_summary$adj.r.squared, "\n")
cat("F-statistic: ", best_model_summary$fstatistic[1], "\n")
cat("p-value for F-statistic: ", pf(best_model_summary$fstatistic[1], 
                                   best_model_summary$fstatistic[2], 
                                   best_model_summary$fstatistic[3], lower.tail = FALSE), "\n")
```

```{r}
# Compute scaling parameters (mean and standard deviation) for the best threshold on the entire dataset
best_threshold <- "Pt_0.0287001"  

# Calculate mean and standard deviation on the full dataset (merged_data)
full_data_mean <- mean(merged_data[[best_threshold]], na.rm = TRUE)
full_data_sd <- sd(merged_data[[best_threshold]], na.rm = TRUE)

# Scale the PRS values using the full dataset parameters
merged_data[[best_threshold]] <- (merged_data[[best_threshold]] - full_data_mean) / full_data_sd


```

```{r}
# Update the merged_data dataframe to only keep the best threshold p-value
updated_data <- merged_data %>%
  select(Subject, SocialScore, best_threshold)  # Select only Subject, SocialScore, and the best p-value column

# View the updated dataframe
head(updated_data)

```

```{r}
# Save the new dataframe to CSV
write.csv(updated_data, file = "scaled_prs_data.csv", row.names = FALSE)

cat("New dataframe with scaled PRS and SocialScore saved as 'scaled_prs_data.csv'.\n")
```

```{r}
# Plot PRS distribution for best threshold

prs_column <- "Pt_0.0287001"

# Calculate the mode (value where the density is highest)
density_data <- density(updated_data[[prs_column]], na.rm = TRUE)
mode_value <- density_data$x[which.max(density_data$y)]

# Create the plot
prs_density_histogram_plot <- ggplot(updated_data, aes(x = .data[[prs_column]])) +
  # Add histogram with adjusted bins and transparency
  geom_histogram(aes(y = ..density..), binwidth = 0.2, fill = "blue", alpha = 0.4, color = "black") +
  # Add the kernel density estimate (KDE) curve
  geom_density(color = "darkblue", size = 1.5) +
  # Add vertical lines for visual reference
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_vline(xintercept = 0.5, color = "red", linetype = "dashed", size = 1) +
  geom_vline(xintercept = -0.5, color = "red", linetype = "dashed", size = 1) +
  
  labs(
    x = paste("PRS Values for", prs_column),
    y = "Density",
  ) +
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 6),
    labels = scales::label_number(scale = 1)
  ) +
  theme(
    axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 16),
    axis.title.x = element_text(size = 18, margin = margin(t = 10)),
    axis.title.y = element_text(size = 18, margin = margin(r = 10)),
    axis.line = element_line(color = "black", size = 1.2),
    plot.title = element_text(size = 20),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16)
  )

# Print the plot
print(prs_density_histogram_plot)
```

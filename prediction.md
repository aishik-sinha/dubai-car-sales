# Dubai Used Cars Deal Prediction - Complete Machine Learning Guide

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [High-Level Strategy](#high-level-strategy)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Model Building](#model-building)
5. [Cross-Validation](#cross-validation)
6. [Final Predictions](#final-predictions)
7. [Key Learnings](#key-learnings)

---

## Understanding the Problem

### What Are We Trying to Do?

We need to predict whether a used car in Dubai is a **"Good" deal, "Average" deal, or "Bad" deal** based on its characteristics.

**Given Information:**
- **Features**: Make, Model, Price, Mileage, Location
- **Target Variable**: `Deal` (Good/Average/Bad)
- **Mystery Variable**: `ValueBenchmark` - a numeric score calculated using `log()` transformations
- **Goal**: Build a model that predicts the `Deal` category

**The Challenge:**
The `ValueBenchmark` is a "secret formula" that aggregates the other features using logarithms. While we don't know the exact formula, understanding it will help us create better features for our model.

---

## High-Level Strategy

### Our Machine Learning Pipeline

```
1. LOAD DATA
   ↓
2. EXPLORE DATA (Understand patterns)
   ↓
3. ENGINEER FEATURES (Create log transformations)
   ↓
4. SPLIT DATA (Train/validation split if needed)
   ↓
5. BUILD MODEL (Decision tree with rpart)
   ↓
6. CROSS-VALIDATE (Test model reliability)
   ↓
7. MAKE PREDICTIONS (Apply to test data)
   ↓
8. EVALUATE & ITERATE (Improve model)
```

### Why This Approach Works

1. **Feature Engineering First**: Since the hint mentions `log()`, we'll create logarithmic features
2. **Decision Trees**: Easy to understand and visualize - perfect for beginners
3. **Cross-Validation**: Ensures our model works on unseen data
4. **Iterative Improvement**: Start simple, then enhance

---

## Step-by-Step Implementation

### Step 1: Setup and Load Data

```r
# Load required libraries
library(rpart)        # Decision trees
library(rpart.plot)   # Visualizing trees
library(caret)        # Machine learning utilities
library(dplyr)        # Data manipulation
library(ggplot2)      # Visualization

# Set working directory (adjust to your path)
# setwd("path/to/your/data")

# Load the data
train_data <- read.csv("CarsTrainNew.csv", stringsAsFactors = FALSE)

# Quick look at the data
head(train_data)
cat("Dataset dimensions:", dim(train_data), "\n")
cat("Column names:", names(train_data), "\n")
```

**What This Does:**
- Loads necessary packages for analysis
- Reads the CSV file into R
- Shows first few rows to understand data structure

---

### Step 2: Exploratory Data Analysis (EDA)

```r
# ===== UNDERSTAND THE DATA =====

# 1. Check data structure
str(train_data)

# 2. Summary statistics
summary(train_data)

# 3. Check for missing values
cat("\nMissing values per column:\n")
colSums(is.na(train_data))

# 4. Check target variable distribution
cat("\n=== Deal Distribution ===\n")
table(train_data$Deal)
prop.table(table(train_data$Deal)) * 100  # Percentages

# 5. Visualize target distribution
ggplot(train_data, aes(x = Deal, fill = Deal)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Distribution of Deal Categories",
       x = "Deal Type", y = "Count") +
  theme_minimal()

# 6. Explore the mysterious ValueBenchmark
cat("\n=== ValueBenchmark Statistics ===\n")
summary(train_data$ValueBenchmark)

# Plot ValueBenchmark by Deal category
ggplot(train_data, aes(x = Deal, y = ValueBenchmark, fill = Deal)) +
  geom_boxplot() +
  labs(title = "ValueBenchmark by Deal Category",
       y = "ValueBenchmark Score") +
  theme_minimal()
```

**Key Insights to Look For:**
- Are the Deal categories balanced?
- What's the range of ValueBenchmark?
- How does ValueBenchmark relate to Deal categories?

---

### Step 3: Feature Engineering - The Critical Step!

Since the hint mentions `log()` and `ValueBenchmark` is an aggregate, let's create features that might help.

```r
# ===== CREATE LOGARITHMIC FEATURES =====

# Why logarithms?
# - Prices and mileage have wide ranges (e.g., $10,000 to $2,000,000)
# - Log transforms compress large values and expand small values
# - This makes patterns easier for models to learn

# Create log-transformed features
train_data$log_Price <- log(train_data$Price + 1)
train_data$log_Mileage <- log(train_data$Mileage + 1)

# Calculate car age (assuming current year is 2025)
# Extract year from model name or use a reasonable estimate
# For simplicity, we'll estimate age based on price/mileage patterns
# In real scenario, you'd have manufacture year

# Create price-to-mileage ratio (value retention indicator)
train_data$Price_per_Mile <- train_data$Price / (train_data$Mileage + 1)
train_data$log_Price_per_Mile <- log(train_data$Price_per_Mile + 1)

# Create interaction features
train_data$Price_Mileage_Interaction <- train_data$log_Price * train_data$log_Mileage

# Visualize the log transformations
par(mfrow=c(2,2))

# Original Price distribution
hist(train_data$Price, main="Original Price", 
     xlab="Price (AED)", col="skyblue", breaks=50)

# Log-transformed Price
hist(train_data$log_Price, main="Log(Price)", 
     xlab="Log Price", col="lightgreen", breaks=50)

# Original Mileage
hist(train_data$Mileage, main="Original Mileage", 
     xlab="Mileage (km)", col="coral", breaks=50)

# Log-transformed Mileage
hist(train_data$log_Mileage, main="Log(Mileage)", 
     xlab="Log Mileage", col="gold", breaks=50)

par(mfrow=c(1,1))
```

**Why These Features Matter:**
- `log_Price`: Normalizes price distribution
- `log_Mileage`: Normalizes mileage distribution  
- `Price_per_Mile`: Captures value retention
- Interactions: Captures combined effects

---

### Step 4: Prepare Data for Modeling

```r
# ===== DATA PREPARATION =====

# Convert categorical variables to factors
train_data$Deal <- as.factor(train_data$Deal)
train_data$Make <- as.factor(train_data$Make)
train_data$Model <- as.factor(train_data$Model)

# Note: We're excluding Location as per instructions

# Select features for modeling
# We'll use engineered features plus some originals
feature_columns <- c("log_Price", "log_Mileage", "Price_per_Mile", 
                     "Price_Mileage_Interaction", "ValueBenchmark")

# Create modeling dataset
model_data <- train_data[, c(feature_columns, "Deal")]

# Check for any remaining missing values
cat("Missing values after feature engineering:\n")
print(colSums(is.na(model_data)))

# Remove any rows with missing values if they exist
model_data <- na.omit(model_data)

cat("\nFinal modeling dataset dimensions:", dim(model_data), "\n")
```

**Important Note:**
We're focusing on numeric features and the mysterious `ValueBenchmark`. Make and Model have too many categories for a beginner model, but you could explore them later with one-hot encoding.

---

## Model Building

### Step 5: Build the Decision Tree Model

```r
# ===== BUILD THE MODEL =====

# Set seed for reproducibility
set.seed(42)

# Method 1: Simple model with default parameters
simple_model <- rpart(
  Deal ~ .,  # Predict Deal using all features
  data = model_data,
  method = "class"
)

# View the tree
print(simple_model)
rpart.plot(simple_model, 
           main = "Simple Decision Tree for Deal Prediction",
           extra = 104)  # Shows class probabilities

# Method 2: Tuned model with controlled parameters
tuned_model <- rpart(
  Deal ~ .,
  data = model_data,
  method = "class",
  control = rpart.control(
    cp = 0.001,          # Complexity parameter (smaller = more complex)
    minsplit = 30,       # Min observations needed to split
    minbucket = 10,      # Min observations in leaf node
    maxdepth = 15        # Maximum tree depth
  )
)

# Visualize the tuned tree
rpart.plot(tuned_model,
           main = "Tuned Decision Tree for Deal Prediction",
           type = 4,
           extra = 104,
           fallen.leaves = TRUE,
           cex = 0.8)

# Compare models on training data
train_pred_simple <- predict(simple_model, model_data, type = "class")
train_pred_tuned <- predict(tuned_model, model_data, type = "class")

cat("\n=== Training Accuracy ===\n")
cat("Simple Model:", 
    mean(train_pred_simple == model_data$Deal) * 100, "%\n")
cat("Tuned Model:", 
    mean(train_pred_tuned == model_data$Deal) * 100, "%\n")

# Confusion Matrix for tuned model
cat("\n=== Confusion Matrix (Tuned Model) ===\n")
confusionMatrix(train_pred_tuned, model_data$Deal)
```

**Understanding the Output:**

- **cp (complexity parameter)**: Controls tree growth. Smaller values allow more complex trees.
- **minsplit**: Minimum observations required to attempt a split
- **minbucket**: Minimum observations required in a leaf node
- **maxdepth**: Maximum levels in the tree

**What to Look For:**
- Training accuracy should be good (>80%) but not perfect (>99% suggests overfitting)
- Confusion matrix shows which categories are hardest to predict

---

### Step 6: Variable Importance

```r
# ===== UNDERSTAND FEATURE IMPORTANCE =====

# Get variable importance
importance <- tuned_model$variable.importance

# Sort by importance
importance_sorted <- sort(importance, decreasing = TRUE)

cat("\n=== Feature Importance ===\n")
print(importance_sorted)

# Visualize importance
importance_df <- data.frame(
  Feature = names(importance_sorted),
  Importance = importance_sorted
)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance in Predicting Deals",
       x = "Feature", y = "Importance Score") +
  theme_minimal()
```

**Interpretation:**
- Features at the top are most important for predictions
- `ValueBenchmark` likely has high importance (it's the "secret sauce")
- This helps us understand what makes a car a good/bad deal

---

## Cross-Validation

### Step 7: Implement K-Fold Cross-Validation

Cross-validation is **CRITICAL** - it tells us if our model will work on new data!

```r
# ===== MANUAL K-FOLD CROSS-VALIDATION =====

perform_cv <- function(data, k = 5, cp_value = 0.001) {
  
  cat("=== Starting", k, "-Fold Cross-Validation ===\n\n")
  
  # Shuffle data
  set.seed(42)
  data <- data[sample(nrow(data)), ]
  
  # Create fold assignments
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  
  # Store results
  fold_accuracies <- numeric(k)
  fold_confusion_matrices <- list()
  
  # Perform k-fold CV
  for(i in 1:k) {
    
    cat("--- Fold", i, "---\n")
    
    # Split data
    test_indices <- which(folds == i)
    cv_train <- data[-test_indices, ]
    cv_test <- data[test_indices, ]
    
    # Train model
    cv_model <- rpart(
      Deal ~ .,
      data = cv_train,
      method = "class",
      control = rpart.control(cp = cp_value, minsplit = 30, maxdepth = 15)
    )
    
    # Predict
    cv_predictions <- predict(cv_model, cv_test, type = "class")
    
    # Calculate accuracy
    fold_accuracies[i] <- mean(cv_predictions == cv_test$Deal)
    
    cat("  Accuracy:", round(fold_accuracies[i] * 100, 2), "%\n\n")
    
    # Store confusion matrix
    fold_confusion_matrices[[i]] <- confusionMatrix(cv_predictions, cv_test$Deal)
  }
  
  # Summary statistics
  mean_acc <- mean(fold_accuracies)
  sd_acc <- sd(fold_accuracies)
  
  cat("\n=== Cross-Validation Results ===\n")
  cat("Mean Accuracy:", round(mean_acc * 100, 2), "%\n")
  cat("Std Deviation:", round(sd_acc * 100, 2), "%\n")
  cat("Min Accuracy:", round(min(fold_accuracies) * 100, 2), "%\n")
  cat("Max Accuracy:", round(max(fold_accuracies) * 100, 2), "%\n")
  
  # Interpretation
  cat("\n=== Interpretation ===\n")
  if(sd_acc < 0.03) {
    cat("✓ LOW variance - Model is STABLE\n")
  } else if(sd_acc < 0.05) {
    cat("⚠ MODERATE variance - Model is reasonably stable\n")
  } else {
    cat("✗ HIGH variance - Model is UNSTABLE\n")
  }
  
  return(list(
    fold_accuracies = fold_accuracies,
    mean_accuracy = mean_acc,
    sd_accuracy = sd_acc,
    confusion_matrices = fold_confusion_matrices
  ))
}

# Run cross-validation
cv_results <- perform_cv(model_data, k = 5, cp_value = 0.001)

# Visualize CV results
cv_df <- data.frame(
  Fold = 1:length(cv_results$fold_accuracies),
  Accuracy = cv_results$fold_accuracies
)

ggplot(cv_df, aes(x = factor(Fold), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  geom_hline(yintercept = cv_results$mean_accuracy, 
             color = "red", linetype = "dashed", size = 1) +
  geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")),
            vjust = -0.5) +
  ylim(0, 1) +
  labs(title = "5-Fold Cross-Validation Results",
       subtitle = paste("Mean:", round(cv_results$mean_accuracy * 100, 1), 
                       "% ± ", round(cv_results$sd_accuracy * 100, 1), "%"),
       x = "Fold Number",
       y = "Accuracy") +
  theme_minimal()
```

**What Cross-Validation Tells Us:**

✓ **Good Result**: Fold accuracies are similar (e.g., 82%, 84%, 83%, 85%, 82%)
- Model is stable and reliable

⚠ **Warning Sign**: Fold accuracies vary widely (e.g., 75%, 92%, 68%, 88%, 71%)
- Model is unstable, might be overfitting

---

### Step 8: Alternative - Caret Cross-Validation

```r
# ===== USING CARET FOR CROSS-VALIDATION =====

# Configure cross-validation
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

# Train with automatic tuning
cv_model_caret <- train(
  Deal ~ .,
  data = model_data,
  method = "rpart",
  trControl = train_control,
  tuneGrid = expand.grid(cp = c(0.001, 0.005, 0.01, 0.02, 0.05)),
  metric = "Accuracy"
)

# View results
print(cv_model_caret)

# Plot accuracy vs complexity parameter
plot(cv_model_caret, 
     main = "Model Accuracy vs Complexity Parameter")

# Best model parameters
cat("\nBest cp value:", cv_model_caret$bestTune$cp, "\n")
```

**Benefits of Caret:**
- Automatically finds best parameters
- Built-in cross-validation
- Easy to use once you understand the basics

---

## Final Predictions

### Step 9: Train Final Model and Make Predictions

```r
# ===== TRAIN FINAL MODEL ON ALL DATA =====

# Use the best parameters from cross-validation
final_model <- rpart(
  Deal ~ .,
  data = model_data,
  method = "class",
  control = rpart.control(
    cp = cv_model_caret$bestTune$cp,  # Use best cp from caret
    minsplit = 30,
    maxdepth = 15
  )
)

# Visualize final model
rpart.plot(final_model,
           main = "Final Decision Tree Model",
           type = 4,
           extra = 104,
           fallen.leaves = TRUE,
           cex = 0.8)

# ===== PREPARE TEST DATA =====

# Load test data (when available)
# test_data <- read.csv("CarsTestNew.csv", stringsAsFactors = FALSE)

# For now, let's create a sample prediction workflow
# You'll replace this with actual test data

# Create same features for test data as we did for training
prepare_test_data <- function(test_df) {
  
  # Create log features
  test_df$log_Price <- log(test_df$Price + 1)
  test_df$log_Mileage <- log(test_df$Mileage + 1)
  test_df$Price_per_Mile <- test_df$Price / (test_df$Mileage + 1)
  test_df$log_Price_per_Mile <- log(test_df$Price_per_Mile + 1)
  test_df$Price_Mileage_Interaction <- test_df$log_Price * test_df$log_Mileage
  
  # Select same feature columns
  test_features <- test_df[, c("log_Price", "log_Mileage", "Price_per_Mile",
                                "Price_Mileage_Interaction", "ValueBenchmark")]
  
  return(test_features)
}

# Example: Make predictions on test data
# test_features <- prepare_test_data(test_data)
# final_predictions <- predict(final_model, test_features, type = "class")

# Get prediction probabilities
# prediction_probs <- predict(final_model, test_features, type = "prob")

# ===== CREATE SUBMISSION FILE =====

# create_submission <- function(test_data, predictions) {
#   submission <- data.frame(
#     ID = 1:nrow(test_data),
#     Make = test_data$Make,
#     Model = test_data$Model,
#     Predicted_Deal = predictions
#   )
#   
#   write.csv(submission, "dubai_cars_predictions.csv", row.names = FALSE)
#   cat("Submission file created: dubai_cars_predictions.csv\n")
# }

# create_submission(test_data, final_predictions)
```

---

## Advanced Tips for Better Performance

### Step 10: Model Improvement Strategies

```r
# ===== STRATEGY 1: Try Different Models =====

# Random Forest (more powerful than single tree)
library(randomForest)

rf_model <- randomForest(
  Deal ~ .,
  data = model_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

# Check performance
rf_pred <- predict(rf_model, model_data)
cat("Random Forest Training Accuracy:", 
    mean(rf_pred == model_data$Deal) * 100, "%\n")

# Variable importance from Random Forest
importance(rf_model)
varImpPlot(rf_model, main = "Random Forest Feature Importance")

# ===== STRATEGY 2: Ensemble Methods =====

# Combine predictions from multiple models
ensemble_predict <- function(data, models) {
  predictions <- sapply(models, function(m) {
    as.character(predict(m, data, type = "class"))
  })
  
  # Majority vote
  final_pred <- apply(predictions, 1, function(x) {
    names(which.max(table(x)))
  })
  
  return(factor(final_pred, levels = levels(data$Deal)))
}

# ===== STRATEGY 3: Feature Selection =====

# Recursive Feature Elimination
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
results <- rfe(
  model_data[, -which(names(model_data) == "Deal")],
  model_data$Deal,
  sizes = c(1:5),
  rfeControl = control
)

print(results)
plot(results, type = c("g", "o"))
```

---

## Key Learnings

### What Makes a Good Deal?

Based on our model, here are the patterns we discovered:

1. **ValueBenchmark is King**: This feature (likely a combination of log transformations) is most predictive

2. **Price-Mileage Relationship**: Cars with low price relative to mileage are better deals

3. **Log Transformations Help**: They normalize wide-ranging values and reveal patterns

### Important Concepts You Learned

✓ **Feature Engineering**: Creating new features from existing ones
✓ **Cross-Validation**: Testing model reliability
✓ **Decision Trees**: How they split data to make predictions
✓ **Model Evaluation**: Using confusion matrices and accuracy

### Common Pitfalls to Avoid

❌ **Overfitting**: Model works great on training data but fails on test data
- Solution: Use cross-validation and simpler models

❌ **Data Leakage**: Using test data information during training
- Solution: Keep test data completely separate

❌ **Ignoring Feature Engineering**: Using raw features only
- Solution: Create meaningful transformations (logs, ratios, interactions)

❌ **Not Understanding Your Data**: Blindly applying algorithms
- Solution: Always do EDA first!

---

## Complete Workflow Summary

```r
# ===== COMPLETE WORKFLOW =====

# 1. Load data
train_data <- read.csv("CarsTrainNew.csv", stringsAsFactors = FALSE)

# 2. Feature engineering
train_data$log_Price <- log(train_data$Price + 1)
train_data$log_Mileage <- log(train_data$Mileage + 1)
train_data$Price_per_Mile <- train_data$Price / (train_data$Mileage + 1)
train_data$Price_Mileage_Interaction <- train_data$log_Price * train_data$log_Mileage

# 3. Prepare modeling data
train_data$Deal <- as.factor(train_data$Deal)
model_data <- train_data[, c("log_Price", "log_Mileage", "Price_per_Mile",
                              "Price_Mileage_Interaction", "ValueBenchmark", "Deal")]
model_data <- na.omit(model_data)

# 4. Cross-validation
cv_results <- perform_cv(model_data, k = 5, cp_value = 0.001)

# 5. Train final model
final_model <- rpart(Deal ~ ., data = model_data, method = "class",
                     control = rpart.control(cp = 0.001, minsplit = 30, maxdepth = 15))

# 6. Make predictions (when test data available)
# test_features <- prepare_test_data(test_data)
# predictions <- predict(final_model, test_features, type = "class")

# 7. Create submission
# create_submission(test_data, predictions)
```

---

## Next Steps for Kaggle Competition

1. **Load the actual test data** when available
2. **Apply the same feature engineering** to test data
3. **Make predictions** using final model
4. **Submit to Kaggle** and check leaderboard score
5. **Iterate**: Try different features, models, parameters based on results

**Good luck with your Kaggle competition!** 🚀

Remember: Machine learning is iterative. Your first model won't be perfect, and that's okay. Each iteration teaches you something new about the data and the problem.

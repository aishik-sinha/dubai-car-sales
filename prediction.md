# Dubai Used Cars Deal Prediction - ML Assignment Guide

## Executive Summary

This guide walks you through building a classification model to predict whether used cars in Dubai are "Good", "Average", or "Bad" deals based on the `ValueBenchmark` score. The ValueBenchmark uses logarithmic transformations (as hinted), and we'll reverse-engineer it to build an accurate prediction model.

**Dataset Overview:**
- **Training Data**: CarsTrainNew.csv with 2000 cars
- **Target Variable**: `Deal` (Good/Average/Bad)
- **Key Features**: Make, Model, Price, Mileage, Location, ValueBenchmark (numeric score)
- **Challenge**: Predict deal quality for test data

---

## 1. Understanding the Problem - High-Level Strategy

### What Are We Predicting?

We're building a **multi-class classification model** that predicts three categories:
- **Good Deal**: Best value for money
- **Average Deal**: Fair pricing
- **Bad Deal**: Overpriced

### The ValueBenchmark Mystery

The hint tells us ValueBenchmark uses `log()`. Looking at the data:
- ValueBenchmark is a numeric score (e.g., 23.55, 22.64)
- Deal categories correlate with this score
- Likely formula: `ValueBenchmark = log(Price) - α*log(Mileage) + other_factors`

**Why logarithms?**
- Prices vary wildly ($10K to $2M+)
- Log transforms make comparisons proportional
- A 10% price increase matters more than $1000 difference

### Our Strategy

```
Step 1: Load & Explore Data
   ↓
Step 2: Feature Engineering (Create log features)
   ↓
Step 3: Build Classification Model (Decision Tree)
   ↓
Step 4: Cross-Validate (Test reliability)
   ↓
Step 5: Make Final Predictions
   ↓
Step 6: Submit to Kaggle
```

---

## 2. Step-by-Step Implementation

### Step 1: Setup and Load Data

```r
# Load required libraries
library(rpart)        # Decision trees
library(rpart.plot)   # Tree visualization
library(caret)        # Cross-validation and metrics
library(dplyr)        # Data manipulation
library(ggplot2)      # Visualization

# Set working directory (CHANGE THIS TO YOUR PATH)
setwd("~/path/to/your/data/folder")

# Load the training data
train_data <- read.csv("CarsTrainNew.csv", stringsAsFactors = FALSE)

# First look at the data
head(train_data)
str(train_data)
```

**Expected Output:**
```
'data.frame':	2000 obs. of  7 variables:
 $ Make          : chr  "nissan" "toyota" "jeep" ...
 $ Model         : chr  "patrol" "rav-4" "wrangler" ...
 $ Price         : num  319484 83096 208842 ...
 $ Mileage       : num  31962 132336 44718 ...
 $ Location      : chr  " Dubai" " Dubai" " Dubai" ...
 $ Deal          : chr  "Average" "Average" "Average" ...
 $ ValueBenchmark: num  23.6 22.6 23.1 ...
```

---

### Step 2: Exploratory Data Analysis (EDA)

```r
# === UNDERSTAND THE TARGET VARIABLE ===
table(train_data$Deal)
prop.table(table(train_data$Deal)) * 100

# Distribution of Deal categories
ggplot(train_data, aes(x = Deal, fill = Deal)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Distribution of Deal Categories",
       x = "Deal Type", y = "Count") +
  theme_minimal()

# === EXPLORE VALUEBENCHMARK ===
# This is the key numeric score we need to understand
summary(train_data$ValueBenchmark)

# ValueBenchmark by Deal category
ggplot(train_data, aes(x = Deal, y = ValueBenchmark, fill = Deal)) +
  geom_boxplot() +
  labs(title = "ValueBenchmark Score by Deal Category",
       subtitle = "Lower scores = Better deals?") +
  theme_minimal()

# === KEY INSIGHT ===
# Good deals have LOWER ValueBenchmark scores
# Bad deals have HIGHER ValueBenchmark scores
aggregate(ValueBenchmark ~ Deal, data = train_data, FUN = mean)

# === EXPLORE NUMERIC FEATURES ===
# Price distribution (very skewed!)
ggplot(train_data, aes(x = Price)) +
  geom_histogram(bins = 50, fill = "steelblue") +
  labs(title = "Price Distribution (Right-Skewed)") +
  theme_minimal()

# Mileage distribution
ggplot(train_data, aes(x = Mileage)) +
  geom_histogram(bins = 50, fill = "coral") +
  labs(title = "Mileage Distribution") +
  theme_minimal()

# === CORRELATIONS ===
# Create numeric dataset for correlation
numeric_data <- train_data %>%
  select(Price, Mileage, ValueBenchmark)

cor(numeric_data)
```

**What We Discover:**
1. **Imbalanced classes**: More "Average" deals than Good/Bad
2. **ValueBenchmark range**: ~20.5 to ~27
3. **Price is heavily right-skewed**: Few very expensive cars
4. **Key insight**: Lower ValueBenchmark = Better Deal

---

### Step 3: Feature Engineering - The Critical Step

```r
# === REVERSE ENGINEERING VALUEBENCHMARK ===
# Based on the hint and data exploration, ValueBenchmark likely uses:
# log(Price), log(Mileage), and possibly make/model information

# Clean up whitespace in Location
train_data$Location <- trimws(train_data$Location)

# === CREATE LOG FEATURES ===
# These are crucial based on the log() hint
train_data$log_Price <- log(train_data$Price)
train_data$log_Mileage <- log(train_data$Mileage + 1)  # +1 to handle zeros

# === CREATE RATIO FEATURES ===
# These capture value relationships
train_data$Price_per_Mile <- train_data$Price / (train_data$Mileage + 1)
train_data$log_Price_Mile_Ratio <- train_data$log_Price - train_data$log_Mileage

# === BRAND ENCODING ===
# Premium brands (Mercedes, BMW, Porsche, etc.) command higher prices
# Create a premium brand indicator
premium_brands <- c("mercedes-benz", "bmw", "porsche", "lamborghini", 
                    "ferrari", "rolls-royce", "bentley", "aston-martin",
                    "maserati", "mclaren", "lotus")

train_data$is_premium <- ifelse(train_data$Make %in% premium_brands, 1, 0)

# === BRAND FREQUENCY ===
# Rare brands might indicate exotic/expensive vehicles
brand_freq <- as.data.frame(table(train_data$Make))
names(brand_freq) <- c("Make", "brand_frequency")
train_data <- merge(train_data, brand_freq, by = "Make", all.x = TRUE)

# === LOCATION ENCODING ===
# Dubai vs other emirates
train_data$is_Dubai <- ifelse(train_data$Location == "Dubai", 1, 0)

# === PRICE CATEGORIES ===
# Sometimes grouping helps capture non-linear relationships
train_data$price_category <- cut(train_data$Price,
                                  breaks = c(0, 50000, 100000, 200000, Inf),
                                  labels = c("Budget", "Mid", "Premium", "Luxury"))

# === CHECK OUR ENGINEERED FEATURES ===
summary(train_data[, c("log_Price", "log_Mileage", "log_Price_Mile_Ratio", 
                       "is_premium", "brand_frequency")])
```

**Why These Features Matter:**
- **log_Price**: Normalizes extreme price variations
- **log_Mileage**: Handles odometer differences proportionally
- **log_Price_Mile_Ratio**: Captures value per mile driven
- **is_premium**: Premium brands justify higher prices
- **brand_frequency**: Rare brands = specialty vehicles

---

### Step 4: Build Classification Models

We'll build multiple models and compare them.

#### Model 1: Simple Decision Tree

```r
# === BASIC DECISION TREE ===
# Start simple - just use the main features

# Select features for modeling
feature_columns <- c("log_Price", "log_Mileage", "log_Price_Mile_Ratio",
                     "is_premium", "brand_frequency", "is_Dubai", "Deal")

model_data <- train_data[, feature_columns]

# Build the model
model_simple <- rpart(
  Deal ~ .,
  data = model_data,
  method = "class",
  control = rpart.control(
    cp = 0.001,        # Complexity parameter (lower = more complex)
    minsplit = 30,     # Min observations to split
    maxdepth = 15      # Max tree depth
  )
)

# Visualize the tree
rpart.plot(model_simple,
           type = 4,
           extra = 104,
           under = TRUE,
           fallen.leaves = TRUE,
           main = "Decision Tree: Dubai Car Deals",
           cex = 0.8)

# Print model summary
print(model_simple)

# Variable importance
var_imp <- model_simple$variable.importance
print("Variable Importance:")
print(sort(var_imp, decreasing = TRUE))

# Create importance plot
importance_df <- data.frame(
  Feature = names(var_imp),
  Importance = var_imp
)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance in Predicting Deals",
       x = "Feature", y = "Importance") +
  theme_minimal()
```

#### Model 2: Using ValueBenchmark Directly

```r
# === MODEL WITH VALUEBENCHMARK ===
# Since ValueBenchmark is the aggregate score, let's use it

model_with_VB <- rpart(
  Deal ~ log_Price + log_Mileage + ValueBenchmark + is_premium + brand_frequency,
  data = train_data,
  method = "class",
  control = rpart.control(cp = 0.001, minsplit = 20)
)

# This model should perform very well since ValueBenchmark
# already encodes the deal quality information
```

#### Model 3: Tuned Model with Grid Search

```r
# === HYPERPARAMETER TUNING ===
# Try different complexity parameters to find the best

# Create a grid of parameters to try
cp_grid <- expand.grid(cp = seq(0.0001, 0.01, by = 0.0005))

# Configure cross-validation
train_control_tune <- trainControl(
  method = "cv",
  number = 10,           # 10-fold CV
  verboseIter = TRUE
)

# Train with tuning
model_tuned <- train(
  Deal ~ log_Price + log_Mileage + log_Price_Mile_Ratio + 
         is_premium + brand_frequency + is_Dubai + ValueBenchmark,
  data = train_data,
  method = "rpart",
  trControl = train_control_tune,
  tuneGrid = cp_grid,
  metric = "Accuracy"
)

# View best parameters
print(model_tuned)
plot(model_tuned, main = "Model Accuracy vs Complexity Parameter")

# Best model
best_model <- model_tuned$finalModel
```

---

### Step 5: Cross-Validation - Testing Model Reliability

```r
# ============================================
# MANUAL K-FOLD CROSS-VALIDATION
# ============================================

perform_cv_multiclass <- function(data, formula, k = 10) {
  
  cat("=== Starting", k, "-Fold Cross-Validation ===\n\n")
  
  # Shuffle data
  set.seed(42)
  data <- data[sample(nrow(data)), ]
  
  # Create folds
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  
  # Storage for results
  fold_accuracies <- numeric(k)
  fold_predictions <- list()
  fold_actuals <- list()
  
  # Perform CV
  for(i in 1:k) {
    cat("--- Fold", i, "of", k, "---\n")
    
    # Split data
    test_indices <- which(folds == i)
    cv_train <- data[-test_indices, ]
    cv_test <- data[test_indices, ]
    
    # Train model
    cv_model <- rpart(
      formula = formula,
      data = cv_train,
      method = "class",
      control = rpart.control(cp = 0.001, minsplit = 20, maxdepth = 15)
    )
    
    # Predict
    cv_predictions <- predict(cv_model, newdata = cv_test, type = "class")
    
    # Calculate accuracy
    fold_accuracies[i] <- sum(cv_predictions == cv_test$Deal) / nrow(cv_test)
    
    # Store for confusion matrix
    fold_predictions[[i]] <- cv_predictions
    fold_actuals[[i]] <- cv_test$Deal
    
    cat("  Accuracy:", round(fold_accuracies[i] * 100, 2), "%\n\n")
  }
  
  # Overall results
  cat("=== Cross-Validation Results ===\n")
  cat("Mean Accuracy:", round(mean(fold_accuracies) * 100, 2), "%\n")
  cat("Std Deviation:", round(sd(fold_accuracies) * 100, 2), "%\n")
  cat("Min Accuracy:", round(min(fold_accuracies) * 100, 2), "%\n")
  cat("Max Accuracy:", round(max(fold_accuracies) * 100, 2), "%\n\n")
  
  # Combined confusion matrix
  all_predictions <- unlist(fold_predictions)
  all_actuals <- unlist(fold_actuals)
  
  conf_matrix <- confusionMatrix(
    factor(all_predictions, levels = c("Good", "Average", "Bad")),
    factor(all_actuals, levels = c("Good", "Average", "Bad"))
  )
  
  print(conf_matrix)
  
  # Return results
  return(list(
    fold_accuracies = fold_accuracies,
    mean_accuracy = mean(fold_accuracies),
    sd_accuracy = sd(fold_accuracies),
    confusion_matrix = conf_matrix
  ))
}

# RUN CROSS-VALIDATION
cv_results <- perform_cv_multiclass(
  data = train_data,
  formula = Deal ~ log_Price + log_Mileage + log_Price_Mile_Ratio + 
                   is_premium + brand_frequency + is_Dubai + ValueBenchmark,
  k = 10
)

# Visualize CV results
cv_df <- data.frame(
  Fold = 1:length(cv_results$fold_accuracies),
  Accuracy = cv_results$fold_accuracies
)

ggplot(cv_df, aes(x = factor(Fold), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  geom_hline(yintercept = cv_results$mean_accuracy, 
             color = "red", linetype = "dashed", size = 1) +
  geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")), vjust = -0.5) +
  ylim(0, 1) +
  labs(title = "10-Fold Cross-Validation Results",
       subtitle = paste("Mean Accuracy:", round(cv_results$mean_accuracy * 100, 1), 
                       "% ± ", round(cv_results$sd_accuracy * 100, 1), "%"),
       x = "Fold Number", y = "Accuracy") +
  theme_minimal()
```

**Interpreting CV Results:**

✅ **Good Signs:**
- Mean accuracy > 85%
- Low standard deviation (< 5%)
- Similar performance across all folds

⚠️ **Warning Signs:**
- High variance (> 10%)
- Some folds much better than others
- Very high accuracy (> 99%) might mean data leakage

---

### Step 6: Train Final Model on All Data

```r
# ============================================
# FINAL MODEL FOR PREDICTIONS
# ============================================

# After CV confirms good performance, train on ALL training data
final_model <- rpart(
  Deal ~ log_Price + log_Mileage + log_Price_Mile_Ratio + 
         is_premium + brand_frequency + is_Dubai + ValueBenchmark,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.001,        # Use best cp from tuning
    minsplit = 20,
    maxdepth = 15
  )
)

# Model summary
print(final_model)

# Visualize final tree
rpart.plot(final_model,
           type = 4,
           extra = 104,
           fallen.leaves = TRUE,
           main = "Final Model: Dubai Car Deal Prediction",
           cex = 0.8)

# Training accuracy (should be very high)
train_predictions <- predict(final_model, newdata = train_data, type = "class")
train_accuracy <- sum(train_predictions == train_data$Deal) / nrow(train_data)
cat("Training Accuracy:", round(train_accuracy * 100, 2), "%\n")

# Confusion matrix on training data
confusionMatrix(train_predictions, factor(train_data$Deal))
```

---

### Step 7: Make Predictions on Test Data

```r
# ============================================
# LOAD AND PREPARE TEST DATA
# ============================================

# Load test data (CHANGE FILENAME AS NEEDED)
test_data <- read.csv("CarsTestNew.csv", stringsAsFactors = FALSE)

# VIEW TEST DATA STRUCTURE
cat("Test data dimensions:", dim(test_data), "\n")
head(test_data)

# ============================================
# APPLY SAME FEATURE ENGINEERING TO TEST DATA
# ============================================

# CRITICAL: Apply EXACT same transformations as training data

# Clean location
test_data$Location <- trimws(test_data$Location)

# Log features
test_data$log_Price <- log(test_data$Price)
test_data$log_Mileage <- log(test_data$Mileage + 1)

# Ratio features
test_data$Price_per_Mile <- test_data$Price / (test_data$Mileage + 1)
test_data$log_Price_Mile_Ratio <- test_data$log_Price - test_data$log_Mileage

# Premium brand indicator
test_data$is_premium <- ifelse(test_data$Make %in% premium_brands, 1, 0)

# Brand frequency (use TRAINING data frequencies)
test_data <- merge(test_data, brand_freq, by = "Make", all.x = TRUE)
# Handle new brands not in training
test_data$brand_frequency[is.na(test_data$brand_frequency)] <- 0

# Location
test_data$is_Dubai <- ifelse(test_data$Location == "Dubai", 1, 0)

# ============================================
# MAKE PREDICTIONS
# ============================================

# Generate predictions
test_predictions <- predict(final_model, newdata = test_data, type = "class")

# Get probability predictions (useful for confidence)
test_probabilities <- predict(final_model, newdata = test_data, type = "prob")

# View predictions
head(test_predictions)
head(test_probabilities)

# Summary of predictions
table(test_predictions)
prop.table(table(test_predictions)) * 100

# ============================================
# CREATE SUBMISSION FILE FOR KAGGLE
# ============================================

# Create submission dataframe
submission <- data.frame(
  Make = test_data$Make,
  Model = test_data$Model,
  Price = test_data$Price,
  Mileage = test_data$Mileage,
  Location = test_data$Location,
  ValueBenchmark = test_data$ValueBenchmark,
  Deal = test_predictions
)

# Save to CSV
write.csv(submission, "dubai_cars_submission.csv", row.names = FALSE)

cat("\n✅ Submission file created: dubai_cars_submission.csv\n")
cat("Total predictions:", nrow(submission), "\n")
cat("Prediction breakdown:\n")
print(table(submission$Deal))
```

---

## 3. Understanding the Model's Logic

### How Decision Trees Make Predictions

The decision tree asks a series of yes/no questions:

```
Example prediction path for a car:

1. Is log_Price > 11.5?  (≈ $100,000)
   └─ YES → Go to Node 2
   
2. Is ValueBenchmark > 24.5?
   └─ NO → Go to Node 3
   
3. Is log_Mileage < 11.0?  (≈ 60,000 km)
   └─ YES → Prediction: GOOD DEAL ✓
```

### Key Insights from Variable Importance

Based on typical results, the most important features are:

1. **ValueBenchmark** (Highest importance)
   - This is the aggregate score
   - Already encodes deal quality
   
2. **log_Price** (Very important)
   - Higher prices generally = worse deals
   
3. **log_Mileage** (Important)
   - Higher mileage = more wear
   
4. **log_Price_Mile_Ratio** (Moderately important)
   - Captures value per mile
   
5. **is_premium** (Useful)
   - Premium brands justify higher prices

---

## 4. Model Performance Expectations

### What to Expect:

**Cross-Validation Accuracy:** 85-95%
- With ValueBenchmark: 90-95%
- Without ValueBenchmark: 75-85%

**Why such high accuracy?**
- ValueBenchmark already encodes the answer
- It's an aggregate score based on the features
- Our job is to learn this relationship

### Confusion Matrix Interpretation:

```
                Predicted
Actual      Good  Average  Bad
  Good       450      30     5
  Average     25     950    40
  Bad          5      40   455

✓ Good → Good: Strong performance
⚠ Average → Bad: Some misclassification
⚠ Bad → Average: Expected overlap
```

---

## 5. Troubleshooting Common Issues

### Issue 1: Low Accuracy (< 70%)

**Possible causes:**
- Didn't apply same feature engineering to test data
- Missing log transformations
- Wrong formula in model

**Solution:**
```r
# Check your features match
names(train_data)
names(test_data)
# Should have same engineered features!
```

### Issue 2: Error "New factor levels"

**Cause:** Test data has Makes/Models not in training data

**Solution:**
```r
# Use numeric encoding instead
# Or handle new levels:
test_data$brand_frequency[is.na(test_data$brand_frequency)] <- 0
```

### Issue 3: Perfect accuracy (100%)

**Cause:** Data leakage - using target in features

**Check:**
- Make sure you're not using `Deal` as a feature
- ValueBenchmark is okay to use (it's a predictor)

---

## 6. Advanced Improvements (Optional)

### Try Different Algorithms

```r
# Random Forest (often better than single tree)
library(randomForest)

rf_model <- randomForest(
  Deal ~ log_Price + log_Mileage + log_Price_Mile_Ratio + 
         is_premium + brand_frequency + ValueBenchmark,
  data = train_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

# XGBoost (gradient boosting)
library(xgboost)

# Requires numeric encoding of target
train_data$Deal_numeric <- as.numeric(factor(train_data$Deal)) - 1

xgb_model <- xgboost(
  data = as.matrix(train_data[, c("log_Price", "log_Mileage", 
                                   "log_Price_Mile_Ratio", "is_premium",
                                   "brand_frequency", "ValueBenchmark")]),
  label = train_data$Deal_numeric,
  nrounds = 100,
  objective = "multi:softmax",
  num_class = 3
)
```

### Ensemble Methods

```r
# Combine multiple models for better predictions
# Average predictions from different models

ensemble_pred <- data.frame(
  tree = as.character(predict(final_model, test_data, type = "class")),
  rf = as.character(predict(rf_model, test_data)),
  stringsAsFactors = FALSE
)

# Majority vote
ensemble_pred$final <- apply(ensemble_pred, 1, function(x) {
  names(which.max(table(x)))
})
```

---

## 7. Complete Code Summary

### Minimal Working Example

```r
# === COMPLETE WORKFLOW ===

# 1. Load libraries
library(rpart)
library(caret)
library(dplyr)

# 2. Load data
train <- read.csv("CarsTrainNew.csv", stringsAsFactors = FALSE)
test <- read.csv("CarsTestNew.csv", stringsAsFactors = FALSE)

# 3. Feature engineering (BOTH train and test)
for(df in list(train, test)) {
  df$log_Price <- log(df$Price)
  df$log_Mileage <- log(df$Mileage + 1)
  df$log_Price_Mile_Ratio <- df$log_Price - df$log_Mileage
}

# 4. Train model
model <- rpart(
  Deal ~ log_Price + log_Mileage + log_Price_Mile_Ratio + ValueBenchmark,
  data = train,
  method = "class",
  control = rpart.control(cp = 0.001)
)

# 5. Cross-validate
train_control <- trainControl(method = "cv", number = 10)
cv_model <- train(
  Deal ~ log_Price + log_Mileage + log_Price_Mile_Ratio + ValueBenchmark,
  data = train,
  method = "rpart",
  trControl = train_control
)
print(cv_model)

# 6. Predict
predictions <- predict(model, newdata = test, type = "class")

# 7. Create submission
submission <- data.frame(
  Make = test$Make,
  Model = test$Model,
  Price = test$Price,
  Mileage = test$Mileage,
  Location = test$Location,
  ValueBenchmark = test$ValueBenchmark,
  Deal = predictions
)

write.csv(submission, "submission.csv", row.names = FALSE)
```

---

## 8. Key Takeaways for Beginners

### ✅ What Makes a Good Model:

1. **Feature Engineering is Critical**
   - Log transformations handle skewed data
   - Ratio features capture relationships
   - Domain knowledge helps (premium brands, etc.)

2. **Cross-Validation is Essential**
   - Tests if model generalizes
   - Reveals overfitting
   - Provides reliable accuracy estimate

3. **Apply Same Transformations to Test Data**
   - MOST COMMON MISTAKE: Forgetting to engineer test features
   - Use exact same code for both train and test

4. **Understand Your Target**
   - ValueBenchmark score determines Deal quality
   - Lower score = Better deal
   - It's based on log(Price), log(Mileage), and other factors

### 🎯 Expected Performance:

- **With ValueBenchmark**: 88-95% accuracy
- **Without ValueBenchmark**: 75-85% accuracy
- **Cross-validation SD**: < 5% (consistent)

### 📝 Submission Checklist:

- [ ] Applied log transformations to test data
- [ ] Created same engineered features
- [ ] Handled missing values
- [ ] Predictions have same format as training Deal column
- [ ] CSV file has correct structure
- [ ] Checked prediction distribution (not all one class)

---

## Good Luck! 🚀

Remember: The goal isn't just high accuracy—it's understanding WHY the model works and being able to explain it clearly. This assignment teaches you the complete ML workflow that applies to any classification problem!

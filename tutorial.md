# Dubai Used Cars Deal Prediction - Machine Learning Assignment

## 1. High-Level Approach and Strategy

### Understanding the Problem

We're building a **classification model** to predict whether a used car in Dubai is a "Deal" or "Not a Deal". The target variable `ValueBenchmark` is our label that we need to predict, and it's calculated using a logarithmic transformation of other features in the dataset.

**What is Classification?**
Classification is a type of supervised machine learning where we predict categories or labels. Think of it like sorting emails into "Spam" or "Not Spam", or in our case, cars into "Deal" or "Not a Deal".

**Supervised Learning Basics:**
- We have **features** (also called predictors or independent variables): Price, Mileage, Age, etc.
- We have a **target** (also called label or dependent variable): ValueBenchmark (Deal/Not Deal)
- The model learns patterns from data where we already know the answer (training data)
- Then it applies these patterns to predict answers for new data (test data)

---

### Strategy Overview - The Machine Learning Pipeline

Think of machine learning as a cooking recipe with specific steps. Here's our recipe:

---

#### **Step 1: Data Preparation - "Know Your Ingredients"**

**What we do:**
- Load and explore the dataset
- Understand feature distributions
- Handle missing values if any
- Split data into training and testing sets

**Why it matters:**
Just like a chef examines ingredients before cooking, we need to understand our data. Bad data leads to bad predictions (garbage in, garbage out).

**Mini Tutorial: Understanding Your Data**

```r
# Basic data exploration commands you'll use:

# 1. View the data
head(data)        # See first 6 rows
tail(data)        # See last 6 rows
View(data)        # Open spreadsheet view

# 2. Understand structure
str(data)         # Data types of each column
dim(data)         # Dimensions: rows x columns
names(data)       # Column names

# 3. Get statistics
summary(data)     # Min, max, mean, median for each column
table(data$Brand) # Count frequency of categories

# 4. Check for issues
sum(is.na(data))           # Count missing values
colSums(is.na(data))       # Missing values per column
any(duplicated(data))      # Check for duplicate rows
```

**Common Data Issues:**
- **Missing values**: Some cells are empty (NA)
- **Outliers**: Extreme values that don't make sense (e.g., car price = $1)
- **Imbalanced classes**: 90% "Not Deal", 10% "Deal" - model might be biased
- **Wrong data types**: Numbers stored as text

---

#### **Step 2: Feature Engineering - "Prepare Your Ingredients"**

**What we do:**
- Since the hint mentions `ValueBenchmark` uses `log()`, we'll reverse-engineer this
- Create or transform features that might be aggregated with logarithms
- Examples: log(Price), log(Mileage), log(Age), etc.

**Why it matters:**
Raw ingredients often need preparation. Features in their original form might not reveal patterns clearly. Transformations help the model "see" patterns better.

> **Important Quote:** "Applied machine learning is basically feature engineering." - Andrew Ng
> 
> Feature engineering is often the difference between a mediocre model (70% accuracy) and a great model (90% accuracy). The same algorithm with better features performs dramatically better!

---

### 🎯 DEEP DIVE: Feature Engineering Masterclass

Feature engineering is the art and science of creating new features or transforming existing ones to help machine learning models learn better. Think of it as translating raw data into a language your model can understand.

---

### **Part A: Why Feature Engineering Matters**

**Real-World Example:**

Imagine teaching a child to identify "expensive" vs "cheap" toys:

❌ **Bad Approach (Raw Features):**
- "This toy costs 2500 cents"
- "This toy weighs 450 grams"
- Too complex! The child struggles to find patterns.

✅ **Good Approach (Engineered Features):**
- "This toy costs $25" (transformed: cents → dollars)
- "This toy is heavy for its price" (created: weight/price ratio)
- "Toys over $20 are usually expensive" (binned: price ranges)
- Patterns are clearer!

**The same applies to machine learning models!**

---

### **Part B: Types of Feature Engineering**

#### **1. TRANSFORMATION - Changing the Scale or Distribution**

Transformations reshape your data to reveal hidden patterns.

##### **A. Logarithmic Transformation**

**When to use:**
- Data is right-skewed (long tail to the right)
- Dealing with prices, salaries, populations
- Want to capture proportional/percentage changes
- Need to reduce impact of outliers

**Why it works:**

```r
# Original data: Car prices
prices <- c(15000, 20000, 50000, 150000, 500000)

# Problem: The difference between $15k and $20k feels similar to 
# the difference between $150k and $500k, but numerically:
20000 - 15000    # = 5,000
500000 - 150000  # = 350,000  (70x larger!)

# This makes it hard for models to learn patterns

# Solution: Log transformation
log_prices <- log(prices)
# log(20000) - log(15000)   # = 0.29
# log(500000) - log(150000) # = 1.20

# Now differences reflect percentage changes, not absolute amounts
```

**Practical Example for Used Cars:**

```r
# Create example data
car_data <- data.frame(
  Brand = c("Toyota", "BMW", "Toyota", "BMW"),
  Price = c(25000, 80000, 30000, 90000),
  Mileage = c(50000, 30000, 80000, 40000)
)

# WITHOUT log transformation
car_data$PricePerMile <- car_data$Price / car_data$Mileage
print(car_data$PricePerMile)
# [1] 0.50 2.67 0.38 2.25
# BMW values dominate because of high prices

# WITH log transformation
car_data$log_Price <- log(car_data$Price)
car_data$log_Mileage <- log(car_data$Mileage)
car_data$log_ratio <- car_data$log_Price - car_data$log_Mileage

print(car_data$log_ratio)
# More balanced comparison of value relationships

# Key insight: log(A/B) = log(A) - log(B)
# This is why ValueBenchmark likely uses log subtraction!
```

**Other Log Properties Used in Feature Engineering:**

```r
# Property 1: log(A × B) = log(A) + log(B)
# Useful for combining multiplicative effects
total_value <- log(Price * Condition_Score)

# Property 2: log(A / B) = log(A) - log(B)
# Useful for ratios (price per mile, value per year)
value_ratio <- log(Price) - log(Age + 1)

# Property 3: log handles zero with a trick
safe_log <- log(Mileage + 1)  # Add 1 to avoid log(0) = -Infinity
```

##### **B. Standardization (Z-Score Normalization)**

**When to use:**
- Features have very different scales
- Using distance-based algorithms (KNN, SVM)
- Want to compare importance of features

```r
# Example: Comparing apples and oranges
Price_raw <- c(20000, 50000, 80000)      # Scale: thousands
Age_raw <- c(1, 3, 5)                     # Scale: single digits
Mileage_raw <- c(10000, 50000, 100000)   # Scale: ten thousands

# Problem: Price dominates because numbers are bigger
# Solution: Standardize (mean=0, std=1)

standardize <- function(x) {
  return((x - mean(x)) / sd(x))
}

Price_std <- standardize(Price_raw)
Age_std <- standardize(Age_raw)
Mileage_std <- standardize(Mileage_raw)

# Now all features are on the same scale!
# A value of 1.5 means "1.5 standard deviations above average"
# regardless of whether it's Price, Age, or Mileage
```

##### **C. Min-Max Normalization**

**When to use:**
- Want features in specific range (e.g., 0 to 1)
- Neural networks often prefer this
- Preserves exact relationships (unlike standardization)

```r
# Scale to 0-1 range
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

Mileage_normalized <- normalize(Mileage_raw)
# Now: 0 = lowest mileage, 1 = highest mileage
```

##### **D. Power Transformations**

```r
# Square root: For moderately skewed data
sqrt_mileage <- sqrt(Mileage)

# Square: To emphasize larger values
age_squared <- Age^2
# Useful for: "Older cars depreciate faster" (quadratic relationship)

# Cube root: For extremely skewed data
cube_root_price <- Price^(1/3)
```

---

#### **2. CREATION - Building New Features from Existing Ones**

This is where creativity meets domain knowledge!

##### **A. Ratio Features**

Ratios often capture relationships better than raw values.

```r
# Example: What makes a car a good deal?

# Raw features don't tell the full story:
# Car A: Price=$30k, Mileage=100k km
# Car B: Price=$30k, Mileage=50k km
# Same price, but Car B is clearly better!

# Solution: Create ratio features
car_data$PricePerMile <- car_data$Price / car_data$Mileage
car_data$PricePerYear <- car_data$Price / (car_data$Age + 1)
car_data$MileagePerYear <- car_data$Mileage / (car_data$Age + 1)

# Now patterns are clearer:
# - Low PricePerMile = Good deal
# - High MileagePerYear = Heavily used
```

**More Ratio Ideas for Car Data:**

```r
# Value retention
car_data$ValueRetention <- car_data$Price / car_data$OriginalPrice

# Efficiency metric
car_data$PricePerHP <- car_data$Price / car_data$Horsepower

# Usage intensity
car_data$MilesPerAge <- car_data$Mileage / car_data$Age

# Combined quality score
car_data$QualityScore <- (car_data$Condition * car_data$BrandRating) / 
                          (car_data$Age + 1)
```

##### **B. Interaction Features**

Some effects only appear when features combine.

```r
# Example: Luxury brand + low mileage = Premium price justified
car_data$LuxuryLowMileage <- (car_data$BrandRating > 7) * 
                               (car_data$Mileage < 50000)
# This creates a binary flag (TRUE/FALSE or 1/0)

# Example: Age × Mileage interaction
car_data$Age_Mileage_Interaction <- car_data$Age * car_data$Mileage
# Captures: "Old car with high mileage is worse than sum of parts"

# Example: Price sensitivity by brand
car_data$PriceBrandInteraction <- car_data$Price * car_data$BrandRating
# Luxury brands can command higher prices
```

##### **C. Aggregation Features**

Summarize related information.

```r
# Example: Overall vehicle score
car_data$OverallScore <- 
  0.3 * car_data$Condition +
  0.2 * car_data$BrandRating +
  0.2 * (10 - car_data$Age) +  # Inverse age (newer = higher score)
  0.3 * (200000 - car_data$Mileage) / 200000  # Inverse mileage

# Example: Depreciation rate
car_data$DepreciationRate <- 
  (car_data$OriginalPrice - car_data$Price) / car_data$Age
```

##### **D. Date/Time Features**

If you have dates, extract meaningful components.

```r
# Assuming you have a SaleDate column
car_data$SaleYear <- year(car_data$SaleDate)
car_data$SaleMonth <- month(car_data$SaleDate)
car_data$SaleQuarter <- quarter(car_data$SaleDate)
car_data$DayOfWeek <- weekdays(car_data$SaleDate)
car_data$IsWeekend <- car_data$DayOfWeek %in% c("Saturday", "Sunday")

# Time since manufacture
car_data$MonthsSinceManufacture <- 
  as.numeric(difftime(car_data$SaleDate, car_data$ManufactureDate, 
                      units = "days")) / 30
```

---

#### **3. BINNING - Converting Continuous to Categorical**

Sometimes grouping values reveals patterns better than exact numbers.

```r
# Example: Age categories
car_data$AgeCategory <- cut(car_data$Age,
                             breaks = c(-Inf, 2, 5, 10, Inf),
                             labels = c("New", "Recent", "Used", "Old"))

# Example: Price categories
car_data$PriceRange <- cut(car_data$Price,
                            breaks = c(0, 30000, 60000, 100000, Inf),
                            labels = c("Budget", "Mid", "Premium", "Luxury"))

# Example: Mileage categories
car_data$MileageLevel <- cut(car_data$Mileage,
                              breaks = c(0, 30000, 70000, 120000, Inf),
                              labels = c("Low", "Average", "High", "VeryHigh"))

# Why this helps:
# - Captures non-linear relationships
# - Reduces impact of outliers
# - Creates interpretable rules
```

---

#### **4. ENCODING - Converting Categories to Numbers**

Machine learning algorithms need numbers, not text!

##### **A. One-Hot Encoding (Dummy Variables)**

Best for nominal categories (no order).

```r
# Example: Brand encoding
brands <- c("Toyota", "BMW", "Honda", "Toyota", "BMW")

# One-hot encoding creates binary columns
library(caret)
dummy_brands <- dummyVars(~ Brand, data = car_data)
brand_encoded <- predict(dummy_brands, car_data)

# Result:
#   Brand_BMW  Brand_Honda  Brand_Toyota
#      0          0            1
#      1          0            0
#      0          1            0
#      0          0            1
#      1          0            0

# Why: No brand is "greater than" another numerically
```

##### **B. Ordinal Encoding**

Best for ordinal categories (has order).

```r
# Example: Condition rating
car_data$Condition_Text <- c("Poor", "Fair", "Good", "Excellent")

# Ordinal encoding maintains order
condition_mapping <- c("Poor" = 1, "Fair" = 2, "Good" = 3, "Excellent" = 4)
car_data$Condition_Numeric <- condition_mapping[car_data$Condition_Text]

# Why: "Excellent" > "Good" > "Fair" > "Poor" has meaning
```

##### **C. Frequency Encoding**

Encode by how common each category is.

```r
# Example: Rare brands might be exotic/expensive
brand_freq <- table(car_data$Brand) / nrow(car_data)
car_data$BrandFrequency <- brand_freq[car_data$Brand]

# Rare brands get low frequency, common brands get high frequency
```

---

### **Part C: Feature Engineering for This Assignment**

Given the hint about `log()` and `ValueBenchmark`, here's a strategic approach:

```r
# Step 1: Create basic log features
train_data$log_Price <- log(train_data$Price)
train_data$log_Mileage <- log(train_data$Mileage + 1)
train_data$log_Age <- log(train_data$Age + 1)

# Step 2: Create ratio features (likely in the formula)
train_data$Price_to_Mileage_Ratio <- log(train_data$Price) - 
                                      log(train_data$Mileage + 1)

train_data$Price_to_Age_Ratio <- log(train_data$Price) - 
                                  log(train_data$Age + 1)

# Step 3: Create quality-adjusted price
train_data$QualityAdjustedPrice <- log(train_data$Price) - 
                                    0.1 * train_data$Condition -
                                    0.1 * train_data$BrandRating

# Step 4: Create depreciation indicator
train_data$DepreciationIndex <- log(train_data$Age + 1) + 
                                 log(train_data$Mileage + 1)

# Step 5: Try to reconstruct ValueBenchmark
train_data$Reconstructed_VB <- 
  log(train_data$Price) -
  0.3 * log(train_data$Mileage + 1) -
  0.5 * log(train_data$Age + 1) +
  0.2 * train_data$BrandRating +
  0.3 * train_data$Condition
```

---

### **Part D: Feature Engineering Best Practices**

#### **✅ DO:**

1. **Start Simple**: Begin with basic transformations, add complexity gradually
2. **Use Domain Knowledge**: Think about what makes cars valuable
3. **Visualize**: Plot distributions before and after transformations
4. **Test Features**: Add one feature at a time and check if performance improves
5. **Document**: Keep notes on what each feature represents

```r
# Example: Test if a feature helps
baseline_model <- rpart(ValueBenchmark ~ Price + Mileage + Age, 
                        data = train_data)
baseline_accuracy <- mean(predict(baseline_model, type="class") == 
                          train_data$ValueBenchmark)

# Add new feature
train_data$new_feature <- log(train_data$Price / train_data$Mileage)

new_model <- rpart(ValueBenchmark ~ Price + Mileage + Age + new_feature, 
                   data = train_data)
new_accuracy <- mean(predict(new_model, type="class") == 
                     train_data$ValueBenchmark)

improvement <- new_accuracy - baseline_accuracy
cat("Improvement:", improvement * 100, "%\n")
```

#### **❌ DON'T:**

1. **Create Too Many Features**: Can lead to overfitting
2. **Use Test Data**: Only engineer features using training data
3. **Include Target Leakage**: Don't use future information
4. **Ignore Missing Values**: Handle them before engineering
5. **Forget to Apply to Test Data**: All transformations must apply to both

```r
# WRONG: Calculating stats from all data
all_data <- rbind(train_data, test_data)
mean_price <- mean(all_data$Price)  # ❌ Uses test data!

# RIGHT: Calculate from training only
mean_price <- mean(train_data$Price)  # ✅ Training only

# Then apply to both
train_data$Price_centered <- train_data$Price - mean_price
test_data$Price_centered <- test_data$Price - mean_price
```

---

### **Part E: Visual Guide to Feature Engineering**

```r
# Let's see the impact visually
library(ggplot2)

# Before transformation: Right-skewed price distribution
ggplot(car_data, aes(x = Price)) +
  geom_histogram(bins = 30, fill = "skyblue") +
  ggtitle("Original Price Distribution (Right-Skewed)")

# After log transformation: More normal distribution
ggplot(car_data, aes(x = log(Price))) +
  geom_histogram(bins = 30, fill = "lightgreen") +
  ggtitle("Log-Transformed Price (More Symmetric)")

# Feature importance visualization
# This shows which features matter most
library(randomForest)
rf_model <- randomForest(ValueBenchmark ~ ., data = train_data)
importance_df <- data.frame(
  Feature = names(importance(rf_model)),
  Importance = importance(rf_model)
)
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +
  ggtitle("Feature Importance")
```

---

### **Part F: Quick Reference - Feature Engineering Checklist**

```r
# Copy this checklist for your assignment!

# □ Log transformations for skewed features
train_data$log_Price <- log(train_data$Price)
train_data$log_Mileage <- log(train_data$Mileage + 1)
train_data$log_Age <- log(train_data$Age + 1)

# □ Ratio features for relationships
train_data$PricePerMile <- train_data$Price / train_data$Mileage
train_data$PricePerYear <- train_data$Price / (train_data$Age + 1)

# □ Interaction features for combined effects
train_data$Age_Mileage <- train_data$Age * train_data$Mileage

# □ Polynomial features for non-linear patterns
train_data$Age_Squared <- train_data$Age^2

# □ Binning for categorical groupings
train_data$AgeGroup <- cut(train_data$Age, breaks = c(0, 3, 7, Inf))

# □ Encoding for categorical variables
train_data$Brand_Encoded <- as.numeric(factor(train_data$Brand))

# □ Standardization if needed
train_data$Price_Scaled <- scale(train_data$Price)

# REMEMBER: Apply exact same transformations to test data!
```

---

**KEY TAKEAWAY:** Feature engineering is where YOU add human intelligence to machine learning. The algorithm can only work with what you give it. Better features = Better predictions!

---

#### **Step 3: Model Building - "Cook the Dish"**

**What we do:**
- Use `rpart` (Recursive Partitioning and Regression Trees) - a decision tree algorithm
- Train the model on training data
- Decision trees are interpretable and good for beginners

**Why Decision Trees?**
They work like a flowchart of yes/no questions:

```
Is Price > $100,000?
├─ NO → Is Mileage < 50,000?
│       ├─ YES → DEAL! (low price, low mileage)
│       └─ NO → Not a Deal
└─ YES → Is BrandRating > 8?
        ├─ YES → Not a Deal (luxury car, expected price)
        └─ NO → Is Age > 5?
                ├─ YES → DEAL! (high price but old, might be overpriced)
                └─ NO → Not a Deal
```

**Mini Tutorial: How Decision Trees Learn**

Decision trees split data to create the "purest" groups:

```r
# Example: Splitting on Price
# Original data: 50% Deal, 50% Not Deal (very mixed)

# After split at Price = $80,000:
# - Left branch (Price < $80,000): 80% Deal, 20% Not Deal (more pure!)
# - Right branch (Price > $80,000): 20% Deal, 80% Not Deal (more pure!)

# The tree keeps splitting until groups are "pure enough"
```

**Key Parameters:**
- `cp` (complexity parameter): How much improvement needed to keep splitting
  - Low cp → Complex tree (might overfit)
  - High cp → Simple tree (might underfit)
- `minsplit`: Minimum data points needed to attempt a split
- `maxdepth`: Maximum levels in the tree

---

#### **Step 4: Cross-Validation - "Taste Test Multiple Times"**

**What we do:**
- Implement k-fold cross-validation to assess model performance
- This helps us understand how well our model generalizes to unseen data

**Why it matters:**
A model might memorize training data but fail on new data (overfitting). Cross-validation tests if the model truly learned patterns or just memorized.

---

### 🎯 DEEP DIVE: Cross-Validation Masterclass

Cross-validation is one of the most important concepts in machine learning. It's your quality control system that tells you if your model is truly good or just got lucky.

---

### **Part A: The Fundamental Problem - Why We Need Cross-Validation**

#### **The Overfitting Trap**

Imagine a student preparing for an exam:

**Scenario 1: The Memorizer (Overfitting)**
```
Practice Problems: Q1, Q2, Q3, Q4, Q5
Student memorizes: "Q1=A, Q2=C, Q3=B, Q4=D, Q5=A"

Practice Exam Score: 100% ✓
Real Exam (new questions): 40% ✗

Problem: Memorized answers, didn't learn concepts
```

**Scenario 2: The Learner (Good Generalization)**
```
Practice Problems: Q1, Q2, Q3, Q4, Q5
Student learns: Underlying principles and methods

Practice Exam Score: 85% ✓
Real Exam (new questions): 82% ✓

Success: Learned patterns, can apply to new problems
```

**In Machine Learning Terms:**

```r
# Your model on training data
Training Accuracy: 99%  # Model knows training data perfectly

# Your model on test data (the real challenge)
Test Accuracy: 65%  # Model fails on new data

# The Gap = Overfitting!
# Model memorized training examples instead of learning patterns
```

#### **The Single Split Problem**

Most beginners do this:

```r
# Split data once
set.seed(123)
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Train model
model <- rpart(ValueBenchmark ~ ., data = train_data)

# Evaluate
predictions <- predict(model, test_data, type = "class")
accuracy <- mean(predictions == test_data$ValueBenchmark)
cat("Model Accuracy:", accuracy)
# Output: 85%
```

**What's the problem?** You got ONE number: 85%. But ask yourself:

- What if you got lucky with an easy test set? → 85% is **overestimate**
- What if you got unlucky with a hard test set? → 85% is **underestimate**
- Would you get 85% again with a different split? → **Unknown**
- Is your model consistently good? → **Can't tell**

This is like taking ONE practice exam and assuming that's your real skill level!

---

### **Part B: K-Fold Cross-Validation - The Solution**

Instead of one split, we test our model multiple times on different data combinations.

#### **How K-Fold Works: Step-by-Step**

Let's use **5-Fold Cross-Validation** as an example:

**Step 1: Divide Your Data**
```
Imagine you have 500 training examples.
Split into 5 equal parts (folds) of 100 examples each:

Fold 1: [100 examples]
Fold 2: [100 examples]
Fold 3: [100 examples]
Fold 4: [100 examples]
Fold 5: [100 examples]
```

**Step 2: Train and Test 5 Times**

```
Iteration 1:
  Test  = Fold 1 (100 examples)
  Train = Folds 2,3,4,5 (400 examples)
  → Build model on 400, test on 100
  → Accuracy: 84%

Iteration 2:
  Test  = Fold 2 (100 examples)
  Train = Folds 1,3,4,5 (400 examples)
  → Build model on 400, test on 100
  → Accuracy: 87%

Iteration 3:
  Test  = Fold 3 (100 examples)
  Train = Folds 1,2,4,5 (400 examples)
  → Build model on 400, test on 100
  → Accuracy: 85%

Iteration 4:
  Test  = Fold 4 (100 examples)
  Train = Folds 1,2,3,5 (400 examples)
  → Build model on 400, test on 100
  → Accuracy: 86%

Iteration 5:
  Test  = Fold 5 (100 examples)
  Train = Folds 1,2,3,4 (400 examples)
  → Build model on 400, test on 100
  → Accuracy: 83%
```

**Step 3: Calculate Average Performance**

```r
fold_accuracies <- c(84, 87, 85, 86, 83)

# Mean accuracy
mean_accuracy <- mean(fold_accuracies)
cat("Average Accuracy:", mean_accuracy, "%\n")
# Output: 85%

# Standard deviation (consistency measure)
sd_accuracy <- sd(fold_accuracies)
cat("Standard Deviation:", sd_accuracy, "%\n")
# Output: 1.58%

# Report final result
cat("Model Performance: 85% ± 1.58%\n")
```

**What We Learned:**
- Model performs around **85%** on average
- Performance is **consistent** (low standard deviation)
- We tested on ALL data (each example was in test set once)
- Much more reliable than a single 85% from one split!

---

### **Part C: Visual Understanding**

#### **ASCII Visualization of 5-Fold CV**

```
Data: [■■■■■■■■■■■■■■■■■■■■] (500 examples)
Split into 5 folds of 100 each:

Round 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN] → Accuracy: 84%
           ↑
         Fold 1

Round 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN] → Accuracy: 87%
                  ↑
                Fold 2

Round 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN] → Accuracy: 85%
                        ↑
                      Fold 3

Round 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN] → Accuracy: 86%
                                ↑
                              Fold 4

Round 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST] → Accuracy: 83%
                                        ↑
                                      Fold 5

Final Result: Average = 85% ± 1.58%
```

#### **Data Flow Diagram**

```
Original Training Data (500 examples)
         |
         ↓
    Shuffle randomly
         |
         ↓
    Split into K folds
         |
    ┌────┴────┬────┬────┬────┐
    ↓         ↓    ↓    ↓    ↓
  Fold1   Fold2 Fold3 Fold4 Fold5
    |
    ↓
FOR each fold:
  1. Hold out this fold as test
  2. Use other folds as train
  3. Build model
  4. Calculate accuracy
  5. Store result
    ↓
Average all accuracies
    ↓
Final CV Score ± Standard Deviation
```

---

### **Part D: Complete Code Implementation with Detailed Explanation**

#### **Method 1: Manual Implementation (Best for Learning)**

```r
# MANUAL K-FOLD CROSS-VALIDATION FUNCTION
# Understanding every step helps you learn!

perform_cv <- function(data, k = 5, model_formula = ValueBenchmark ~ .) {
  
  cat("=== Starting", k, "-Fold Cross-Validation ===\n\n")
  
  # STEP 1: Shuffle the data
  # Why? To ensure random distribution across folds
  set.seed(42)  # For reproducibility
  data <- data[sample(nrow(data)), ]
  cat("✓ Shuffled", nrow(data), "rows randomly\n")
  
  # STEP 2: Create fold assignments
  # cut() divides data into k equal-sized groups
  fold_size <- nrow(data) / k
  folds <- cut(seq(1, nrow(data)), 
               breaks = k, 
               labels = FALSE)
  
  cat("✓ Created", k, "folds of ~", round(fold_size), "examples each\n\n")
  
  # STEP 3: Initialize storage for results
  fold_accuracies <- numeric(k)
  fold_confusion_matrices <- list()
  
  # STEP 4: Perform k iterations
  for(i in 1:k) {
    
    cat("--- Fold", i, "of", k, "---\n")
    
    # Identify which rows belong to test fold
    test_indices <- which(folds == i)
    
    # Split data
    cv_train <- data[-test_indices, ]  # All rows EXCEPT test fold
    cv_test <- data[test_indices, ]     # Only test fold
    
    cat("  Training on:", nrow(cv_train), "examples\n")
    cat("  Testing on:", nrow(cv_test), "examples\n")
    
    # Train the model on training fold
    cv_model <- rpart(
      formula = model_formula,
      data = cv_train,
      method = "class",
      control = rpart.control(
        cp = 0.01,
        minsplit = 20,
        maxdepth = 10
      )
    )
    
    # Make predictions on test fold
    cv_predictions <- predict(cv_model, 
                              newdata = cv_test, 
                              type = "class")
    
    # Calculate accuracy for this fold
    correct_predictions <- sum(cv_predictions == cv_test$ValueBenchmark)
    total_predictions <- length(cv_predictions)
    fold_accuracies[i] <- correct_predictions / total_predictions
    
    cat("  Accuracy:", round(fold_accuracies[i] * 100, 2), "%\n")
    cat("  (", correct_predictions, "correct out of", total_predictions, ")\n\n")
    
    # Store confusion matrix for detailed analysis
    fold_confusion_matrices[[i]] <- table(
      Predicted = cv_predictions,
      Actual = cv_test$ValueBenchmark
    )
  }
  
  # STEP 5: Calculate summary statistics
  mean_accuracy <- mean(fold_accuracies)
  sd_accuracy <- sd(fold_accuracies)
  min_accuracy <- min(fold_accuracies)
  max_accuracy <- max(fold_accuracies)
  
  # STEP 6: Report results
  cat("=== Cross-Validation Results ===\n")
  cat("Mean Accuracy:", round(mean_accuracy * 100, 2), "%\n")
  cat("Std Deviation:", round(sd_accuracy * 100, 2), "%\n")
  cat("Min Accuracy:", round(min_accuracy * 100, 2), "%\n")
  cat("Max Accuracy:", round(max_accuracy * 100, 2), "%\n")
  cat("Range:", round((max_accuracy - min_accuracy) * 100, 2), "%\n\n")
  
  # STEP 7: Interpret results
  cat("=== Interpretation ===\n")
  if(sd_accuracy < 0.05) {
    cat("✓ LOW variance - Model is CONSISTENT and STABLE\n")
  } else if(sd_accuracy < 0.10) {
    cat("⚠ MODERATE variance - Model is reasonably stable\n")
  } else {
    cat("✗ HIGH variance - Model is UNSTABLE, may be overfitting\n")
  }
  
  if(mean_accuracy > 0.85) {
    cat("✓ GOOD performance - Model learns patterns well\n")
  } else if(mean_accuracy > 0.70) {
    cat("⚠ MODERATE performance - Room for improvement\n")
  } else {
    cat("✗ POOR performance - Model needs significant improvement\n")
  }
  
  # Return detailed results
  return(list(
    fold_accuracies = fold_accuracies,
    mean_accuracy = mean_accuracy,
    sd_accuracy = sd_accuracy,
    confusion_matrices = fold_confusion_matrices
  ))
}

# RUN CROSS-VALIDATION
cv_results <- perform_cv(train_data, k = 5)

# VISUALIZE RESULTS
library(ggplot2)

# Create a dataframe for plotting
cv_df <- data.frame(
  Fold = 1:length(cv_results$fold_accuracies),
  Accuracy = cv_results$fold_accuracies
)

# Plot fold accuracies
ggplot(cv_df, aes(x = factor(Fold), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  geom_hline(yintercept = cv_results$mean_accuracy, 
             color = "red", linetype = "dashed", size = 1) +
  geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")),
            vjust = -0.5) +
  ylim(0, 1) +
  labs(title = "Cross-Validation Accuracy by Fold",
       subtitle = paste("Mean:", round(cv_results$mean_accuracy * 100, 1), 
                       "% ± ", round(cv_results$sd_accuracy * 100, 1), "%"),
       x = "Fold Number",
       y = "Accuracy") +
  theme_minimal()
```

#### **Method 2: Using Caret (Industry Standard)**

```r
# USING CARET PACKAGE
# Simpler but less transparent (good for production)

library(caret)

# STEP 1: Configure cross-validation
train_control <- trainControl(
  method = "cv",              # Cross-validation
  number = 5,                 # 5 folds
  verboseIter = TRUE,         # Show progress
  savePredictions = "final",  # Save predictions
  classProbs = TRUE,          # Get probability predictions
  summaryFunction = multiClassSummary  # Get detailed metrics
)

# STEP 2: Train with cross-validation
# Caret automatically:
# - Splits data into folds
# - Trains on each fold combination
# - Tests on held-out fold
# - Averages results
cv_model_caret <- train(
  ValueBenchmark ~ .,
  data = train_data,
  method = "rpart",           # Decision tree
  trControl = train_control,
  tuneLength = 5,             # Try 5 different complexity parameters
  metric = "Accuracy"         # Optimize for accuracy
)

# STEP 3: View results
print(cv_model_caret)

# The output shows:
# - Best parameter (cp value)
# - Cross-validated accuracy
# - Other metrics (Kappa, etc.)

# STEP 4: Plot results
plot(cv_model_caret, 
     main = "Cross-Validation: Accuracy vs Complexity Parameter")

# STEP 5: Get detailed CV results
cv_model_caret$results
cv_model_caret$resample  # Individual fold results
```

---

### **Part E: Interpreting Cross-Validation Results**

#### **Scenario 1: Good Model (Consistent & Accurate)**

```r
Fold 1: 85%
Fold 2: 87%
Fold 3: 86%
Fold 4: 84%
Fold 5: 88%

Mean: 86% ± 1.5%
```

**Interpretation:**
- ✓ High accuracy (86%)
- ✓ Low variance (±1.5%)
- ✓ All folds similar
- **Conclusion**: Model learned genuine patterns, will likely perform well on new data

#### **Scenario 2: Overfitting Model (Inconsistent)**

```r
Fold 1: 95%
Fold 2: 70%
Fold 3: 88%
Fold 4: 65%
Fold 5: 92%

Mean: 82% ± 13.5%
```

**Interpretation:**
- ⚠ Moderate accuracy (82%)
- ✗ High variance (±13.5%)
- ✗ Folds very different
- **Conclusion**: Model is unstable, likely overfitting. Try:
  - Simpler model (increase cp)
  - More training data
  - Better features
  - Regularization

#### **Scenario 3: Underfitting Model (Consistent but Poor)**

```r
Fold 1: 63%
Fold 2: 61%
Fold 3: 64%
Fold 4: 62%
Fold 5: 60%

Mean: 62% ± 1.5%
```

**Interpretation:**
- ✗ Low accuracy (62%)
- ✓ Low variance (±1.5%)
- ✓ Consistent but consistently bad
- **Conclusion**: Model too simple, not learning patterns. Try:
  - More complex model (decrease cp)
  - Better features
  - Feature engineering
  - Different algorithm

---

### **Part F: Common Cross-Validation Mistakes**

#### **Mistake 1: Data Leakage**

```r
# WRONG: Feature engineering before splitting
data$log_price_centered <- log(data$Price) - mean(log(data$Price))
# ↑ Uses mean from ALL data including test folds!

cv_results <- perform_cv(data, k = 5)
# Results are INVALID - test data leaked into training

# RIGHT: Feature engineering inside CV loop
perform_cv_proper <- function(data, k = 5) {
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  
  for(i in 1:k) {
    cv_train <- data[-which(folds == i), ]
    cv_test <- data[which(folds == i), ]
    
    # Calculate mean from training data ONLY
    train_mean <- mean(log(cv_train$Price))
    
    # Apply to both train and test
    cv_train$log_price_centered <- log(cv_train$Price) - train_mean
    cv_test$log_price_centered <- log(cv_test$Price) - train_mean
    
    # Now train and test...
  }
}
```

#### **Mistake 2: Using CV for Final Predictions**

```r
# WRONG: Using CV model for Kaggle submission
cv_results <- perform_cv(train_data, k = 5)
# CV creates 5 different models - which one to use?

# RIGHT: Use CV for evaluation, then retrain on ALL data
cv_results <- perform_cv(train_data, k = 5)
cat("Expected performance:", cv_results$mean_accuracy)

# Now train final model on ALL training data
final_model <- rpart(ValueBenchmark ~ ., data = train_data)

# Use this for Kaggle predictions
kaggle_predictions <- predict(final_model, test_data, type = "class")
```

#### **Mistake 3: Too Few or Too Many Folds**

```r
# k = 2 (too few)
# - Only 2 tests, not reliable
# - Uses only 50% of data for training each time

# k = 100 (too many)
# - Very slow (100 iterations)
# - Each training set very similar to others
# - High computational cost for little benefit

# Sweet spot: k = 5 or k = 10
# - 5: Good balance, common choice
# - 10: More reliable but slower
```

---

### **Part G: Advanced: Leave-One-Out Cross-Validation (LOOCV)**

```r
# Extreme case: k = number of examples
# For 500 examples, k = 500

# Fold 1: Test on example 1, train on 499 others
# Fold 2: Test on example 2, train on 499 others
# ...
# Fold 500: Test on example 500, train on 499 others

# Pros:
# - Maximum use of data
# - Nearly unbiased estimate

# Cons:
# - VERY slow (500 iterations!)
# - High variance in estimate
# - Not recommended unless very small dataset

# In R:
train_control_loocv <- trainControl(method = "LOOCV")
```

---

### **Part H: Summary Checklist**

```r
# Cross-Validation Best Practices Checklist

# ✓ Use 5 or 10 folds for most problems
# ✓ Shuffle data before splitting
# ✓ Calculate fold assignments properly
# ✓ Do feature engineering INSIDE CV loop (avoid leakage)
# ✓ Check both mean accuracy AND variance
# ✓ Retrain on ALL data after CV for final model
# ✓ Use CV to compare different models
# ✓ Report results as: Mean ± SD
# ✓ Interpret consistency as important as accuracy
# ✓ Visualize fold accuracies to spot problems
```

**KEY TAKEAWAY:** Cross-validation is like taking multiple practice exams instead of just one. It gives you a reliable estimate of how well your model truly performs and whether it's ready for the real test (Kaggle competition)!

---

#### **Step 5: Prediction and Evaluation - "Serve and Get Feedback"**

**What we do:**
- Generate predictions on test set
- Evaluate model accuracy and other metrics

**Why it matters:**
The final test! This is like submitting to Kaggle - we see how our model performs on completely unseen data.

**Mini Tutorial: Understanding Evaluation Metrics**

**1. Confusion Matrix:**
```
                  Predicted
              Deal    Not Deal
Actual Deal    50        10       ← 10 Deals we missed!
    Not Deal   5         35       ← 5 False alarms
```

**2. Accuracy:**
```r
# Accuracy = Correct Predictions / Total Predictions
# Accuracy = (50 + 35) / (50 + 10 + 5 + 35) = 85%

# BUT BEWARE: Accuracy can be misleading!
# If 95% of cars are "Not Deal", predicting "Not Deal" for 
# everything gives 95% accuracy but is useless!
```

**3. Other Important Metrics:**

```r
# Precision: Of all cars we predicted as "Deal", how many actually were?
# Precision = True Deals / (True Deals + False Deal Predictions)
# Precision = 50 / (50 + 5) = 91%

# Recall: Of all actual deals, how many did we find?
# Recall = True Deals / (True Deals + Missed Deals)
# Recall = 50 / (50 + 10) = 83%

# F1-Score: Harmonic mean of Precision and Recall
# Balances both metrics
```

**Which Metric Matters?**
- **High Precision**: Use when false alarms are costly (don't want to advertise bad "deals")
- **High Recall**: Use when missing opportunities is costly (want to find all deals)
- **Balanced (F1)**: Use when both matter equally

---

### The Complete Pipeline Flow

```
Raw Data
   ↓
[1. Data Prep] → Clean, explore, understand
   ↓
[2. Feature Engineering] → Transform, create new features
   ↓
[3. Model Building] → Train decision tree
   ↓
[4. Cross-Validation] → Test robustness
   ↓
[5. Prediction] → Apply to new data
   ↓
[6. Evaluation] → Measure performance
   ↓
[7. Iterate] → Improve based on results (repeat steps 2-6)
```

---

### Common Beginner Mistakes to Avoid

1. **Data Leakage**: Using test data information in training
   - ❌ Calculate median from entire dataset, then split
   - ✅ Split first, then calculate median from training only

2. **Overfitting**: Model memorizes training data
   - Signs: 99% training accuracy, 60% test accuracy
   - Solution: Simpler model, more data, cross-validation

3. **Not Scaling Features**: Some algorithms need features on same scale
   - Not critical for decision trees, but important for many models

4. **Ignoring Imbalanced Classes**: 90% one class, 10% another
   - Model might just predict majority class always
   - Solutions: Resampling, class weights, different metrics

5. **Not Understanding Your Data**: 
   - ALWAYS look at your data before modeling
   - Plot distributions, check correlations, find outliers

---

### Tips for This Specific Assignment

**Reverse Engineering ValueBenchmark:**

The hint says it uses `log()` and is aggregate-based. Think about what makes a car a good deal:

1. **Lower Price** (relative to car quality) → Good deal
2. **Lower Mileage** → Better condition → Good deal
3. **Newer (Lower Age)** → More valuable → Good deal
4. **Better Brand** → Holds value → Affects deal status
5. **Better Condition** → More valuable → Good deal

Possible formula structure:
```r
ValueBenchmark_Score = 
  log(Price)              # Higher price = higher score (worse deal)
  - α * log(Mileage)      # Higher mileage = lower score (better deal formula)
  - β * log(Age)          # Older = lower score
  + γ * BrandRating       # Premium brand = higher score
  + δ * Condition         # Better condition = higher score
```

If score is **low** → Price is low relative to quality → **DEAL!**
If score is **high** → Price is high relative to quality → **NOT A DEAL**

**Your Task:** Create log-transformed features and let the model discover these relationships!

---

## 2. Step-by-Step Implementation with Explanations

### Step 1: Setup and Data Creation

Since the dataset is missing, we'll create synthetic data that mimics used car data from Dubai.

```r
# Load required libraries
# These libraries provide functions for machine learning and data manipulation
library(rpart)        # For building decision trees
library(rpart.plot)   # For visualizing decision trees
library(caret)        # For machine learning utilities and cross-validation
library(dplyr)        # For data manipulation

# Set seed for reproducibility
# This ensures we get the same "random" results each time we run the code
set.seed(123)

# Create synthetic training dataset (500 observations)
n_train <- 500

train_data <- data.frame(
  # Price in AED (20,000 to 200,000)
  Price = runif(n_train, 20000, 200000),
  
  # Mileage in kilometers (5,000 to 200,000)
  Mileage = runif(n_train, 5000, 200000),
  
  # Age of car in years (0 to 15)
  Age = runif(n_train, 0, 15),
  
  # Engine size in liters (1.0 to 5.0)
  EngineSize = runif(n_train, 1.0, 5.0),
  
  # Brand rating (1 to 10, higher is more premium)
  BrandRating = sample(1:10, n_train, replace = TRUE),
  
  # Condition score (1 to 10, higher is better condition)
  Condition = sample(1:10, n_train, replace = TRUE)
)

# Create the ValueBenchmark using logarithmic transformation
# This is the "secret formula" we're trying to predict
# Lower values indicate better deals
train_data$ValueBenchmark_Score <- 
  log(train_data$Price) - 
  0.3 * log(train_data$Mileage + 1) - 
  0.5 * log(train_data$Age + 1) + 
  0.2 * train_data$BrandRating + 
  0.3 * train_data$Condition

# Convert to binary classification: Deal or Not Deal
# If the score is below the median, it's a "Deal"
median_score <- median(train_data$ValueBenchmark_Score)
train_data$ValueBenchmark <- ifelse(
  train_data$ValueBenchmark_Score <= median_score, 
  "Deal", 
  "Not_Deal"
)
train_data$ValueBenchmark <- as.factor(train_data$ValueBenchmark)

# Remove the score column (we only want to predict the binary outcome)
train_data <- train_data %>% select(-ValueBenchmark_Score)

# Create synthetic testing dataset (200 observations)
n_test <- 200

test_data <- data.frame(
  Price = runif(n_test, 20000, 200000),
  Mileage = runif(n_test, 5000, 200000),
  Age = runif(n_test, 0, 15),
  EngineSize = runif(n_test, 1.0, 5.0),
  BrandRating = sample(1:10, n_test, replace = TRUE),
  Condition = sample(1:10, n_test, replace = TRUE)
)

# Create actual labels for test data (normally hidden in competition)
test_data$ValueBenchmark_Score <- 
  log(test_data$Price) - 
  0.3 * log(test_data$Mileage + 1) - 
  0.5 * log(test_data$Age + 1) + 
  0.2 * test_data$BrandRating + 
  0.3 * test_data$Condition

test_data$ValueBenchmark <- ifelse(
  test_data$ValueBenchmark_Score <= median_score, 
  "Deal", 
  "Not_Deal"
)
test_data$ValueBenchmark <- as.factor(test_data$ValueBenchmark)
test_data <- test_data %>% select(-ValueBenchmark_Score)
```

**Explanation:**
- We create features that would be typical for used car data
- The `ValueBenchmark` is created using logarithms (as hinted), combining price, mileage, age, and quality factors
- We convert the continuous score to a binary classification: "Deal" vs "Not Deal"
- The `set.seed()` function ensures reproducibility

---

### Step 2: Exploratory Data Analysis

```r
# View the first few rows of training data
head(train_data)

# Check the structure of the data
str(train_data)

# Summary statistics for all variables
summary(train_data)

# Check the distribution of our target variable
table(train_data$ValueBenchmark)
prop.table(table(train_data$ValueBenchmark))
```

**Explanation:**
- `head()` shows us the first 6 rows to get a sense of the data
- `str()` displays the structure: data types and dimensions
- `summary()` provides statistical summaries (min, max, mean, median, etc.)
- `table()` counts how many "Deal" vs "Not_Deal" we have
- `prop.table()` converts counts to proportions (percentages)

---

### Step 3: Feature Engineering

Since `ValueBenchmark` uses logarithms, let's create log-transformed features:

```r
# Create logarithmic transformations of key features
train_data$log_Price <- log(train_data$Price)
train_data$log_Mileage <- log(train_data$Mileage + 1)  # +1 to avoid log(0)
train_data$log_Age <- log(train_data$Age + 1)

# Do the same for test data
test_data$log_Price <- log(test_data$Price)
test_data$log_Mileage <- log(test_data$Mileage + 1)
test_data$log_Age <- log(test_data$Age + 1)

# View the updated data
head(train_data)
```

**Explanation:**
- Logarithmic transformations help with skewed data
- We add 1 before taking log to avoid issues with zero values: `log(0)` is undefined
- This creates new features that might help our model better understand the patterns
- These transformations often help machine learning models perform better

---

### Step 4: Build the Classification Model

```r
# Build a decision tree model using rpart
# The formula says: "Predict ValueBenchmark using all other variables"
# The ~ symbol means "is predicted by"
# The . means "all other variables in the dataset except the target"

model <- rpart(
  formula = ValueBenchmark ~ .,  # Predict ValueBenchmark using all features
  data = train_data,              # Use training data
  method = "class",               # "class" for classification
  control = rpart.control(
    cp = 0.01,                    # Complexity parameter (lower = more complex tree)
    minsplit = 20,                # Minimum observations needed to split a node
    maxdepth = 10                 # Maximum depth of tree
  )
)

# View the model summary
print(model)

# Visualize the decision tree
rpart.plot(model, 
           type = 4,                  # Type of plot
           extra = 101,               # Show class probabilities
           fallen.leaves = TRUE,      # Align leaf nodes
           main = "Decision Tree for Dubai Car Deals")
```

**Explanation:**
- `rpart()` builds a decision tree (like a flowchart of yes/no questions)
- The tree learns patterns by asking questions like "Is Price > 100,000?"
- `method = "class"` tells R this is classification (not regression)
- `cp` (complexity parameter): Controls tree complexity. Lower values = more complex trees
- `minsplit`: Minimum number of observations required to attempt a split
- `maxdepth`: Prevents the tree from growing too deep (prevents overfitting)
- `rpart.plot()` creates a visual representation of the decision tree

---

### Step 5: Make Predictions on Test Data

```r
# Generate predictions on the test dataset
# type = "class" gives us the predicted class labels
predictions <- predict(model, newdata = test_data, type = "class")

# View first few predictions
head(predictions)

# If you want probabilities instead:
predictions_prob <- predict(model, newdata = test_data, type = "prob")
head(predictions_prob)
```

**Explanation:**
- `predict()` applies our trained model to new data
- `type = "class"` returns the predicted category (Deal or Not_Deal)
- `type = "prob"` returns probabilities for each class
- The model uses the decision tree rules learned from training data

---

### Step 6: Evaluate Model Performance

```r
# Create a confusion matrix to see how well we did
confusion_matrix <- table(Predicted = predictions, Actual = test_data$ValueBenchmark)
print(confusion_matrix)

# Calculate accuracy manually
accuracy <- sum(predictions == test_data$ValueBenchmark) / length(predictions)
cat("Model Accuracy:", round(accuracy * 100, 2), "%\n")

# Use caret for more detailed metrics
confusionMatrix(predictions, test_data$ValueBenchmark)
```

**Explanation:**
- **Confusion Matrix**: Shows correct and incorrect predictions
  - True Positives (TP): Correctly predicted "Deal"
  - True Negatives (TN): Correctly predicted "Not_Deal"
  - False Positives (FP): Predicted "Deal" but was "Not_Deal"
  - False Negatives (FN): Predicted "Not_Deal" but was "Deal"
- **Accuracy**: Percentage of correct predictions
- `confusionMatrix()` from caret provides additional metrics like sensitivity and specificity

---

### Step 7: Cross-Validation

Cross-validation helps us understand how well our model generalizes by training on different subsets of data.

```r
# Method 1: Manual K-Fold Cross-Validation Function
perform_cv <- function(data, k = 5) {
  # k = number of folds (we'll divide data into k parts)
  
  # Shuffle the data randomly
  data <- data[sample(nrow(data)), ]
  
  # Create k equally sized folds
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  
  # Store accuracy for each fold
  accuracies <- numeric(k)
  
  # Perform k-fold cross-validation
  for(i in 1:k) {
    # Split data: test fold i, train on all others
    test_indices <- which(folds == i)
    cv_train <- data[-test_indices, ]
    cv_test <- data[test_indices, ]
    
    # Train model on training fold
    cv_model <- rpart(ValueBenchmark ~ ., 
                      data = cv_train, 
                      method = "class",
                      control = rpart.control(cp = 0.01))
    
    # Make predictions on test fold
    cv_predictions <- predict(cv_model, newdata = cv_test, type = "class")
    
    # Calculate accuracy for this fold
    accuracies[i] <- sum(cv_predictions == cv_test$ValueBenchmark) / nrow(cv_test)
    
    cat("Fold", i, "Accuracy:", round(accuracies[i] * 100, 2), "%\n")
  }
  
  # Return mean accuracy across all folds
  mean_accuracy <- mean(accuracies)
  cat("\n--- Cross-Validation Results ---\n")
  cat("Mean Accuracy:", round(mean_accuracy * 100, 2), "%\n")
  cat("Standard Deviation:", round(sd(accuracies) * 100, 2), "%\n")
  
  return(list(accuracies = accuracies, mean_accuracy = mean_accuracy))
}

# Perform 5-fold cross-validation
cv_results <- perform_cv(train_data, k = 5)
```

**Explanation:**
- **K-Fold Cross-Validation**: Divides data into k parts
- For each fold:
  1. Use k-1 parts for training
  2. Use 1 part for testing
  3. Calculate accuracy
- Repeat k times so each part serves as test set once
- Average all k accuracies to get overall performance
- This gives us a more reliable estimate of model performance than a single train/test split

```r
# Method 2: Using caret's built-in cross-validation
train_control <- trainControl(
  method = "cv",          # Cross-validation
  number = 5,             # 5 folds
  savePredictions = TRUE  # Save predictions
)

# Train model with cross-validation
cv_model_caret <- train(
  ValueBenchmark ~ .,
  data = train_data,
  method = "rpart",
  trControl = train_control,
  tuneLength = 5  # Try 5 different complexity parameters
)

# View results
print(cv_model_caret)
plot(cv_model_caret)
```

**Explanation:**
- `trainControl()` sets up the cross-validation method
- `train()` from caret automatically performs cross-validation
- `tuneLength = 5` tells caret to try 5 different model configurations
- Caret automatically finds the best configuration
- The plot shows accuracy vs. complexity parameter

---

## 3. Final Model Summary and Insights

```r
# Get variable importance
variable_importance <- model$variable.importance
print("Variable Importance:")
print(sort(variable_importance, decreasing = TRUE))

# Create a bar plot of variable importance
barplot(sort(variable_importance, decreasing = TRUE),
        las = 2,
        col = "steelblue",
        main = "Feature Importance in Predicting Car Deals",
        ylab = "Importance",
        cex.names = 0.8)
```

**Explanation:**
- Variable importance tells us which features matter most
- Features with higher importance have more influence on predictions
- This helps us understand what makes a car a "deal"

---

## 4. Key Takeaways for Beginners

### What We Learned:

1. **Classification vs Regression**: 
   - Classification predicts categories (Deal/Not Deal)
   - Regression predicts numbers (Price)

2. **Decision Trees**: 
   - Easy to understand and visualize
   - Make decisions through a series of yes/no questions
   - Can overfit if not controlled properly

3. **Cross-Validation**:
   - Essential for understanding model performance
   - Prevents overfitting
   - Gives more reliable accuracy estimates

4. **Feature Engineering**:
   - Creating new features (like log transformations) can improve model performance
   - Domain knowledge helps (knowing that price/value ratios often use logs)

5. **The Hint About log()**:
   - Logarithms are commonly used in financial/pricing models
   - They help normalize skewed distributions
   - Our engineered log features likely improved model performance

### Next Steps:
- Try different models (Random Forest, Logistic Regression)
- Tune hyperparameters more carefully
- Explore more feature engineering
- Analyze misclassified examples to improve the model

---

## 5. Complete Code for Kaggle Submission

```r
# After training your best model, create predictions for submission:

# Load test data (replace with actual Kaggle test file)
# kaggle_test <- read.csv("test_data.csv")

# Make predictions
# final_predictions <- predict(model, newdata = kaggle_test, type = "class")

# Create submission file
# submission <- data.frame(
#   ID = 1:nrow(kaggle_test),
#   ValueBenchmark = final_predictions
# )

# Write to CSV
# write.csv(submission, "dubai_cars_submission.csv", row.names = FALSE)
```

Good luck with your Kaggle competition! Remember to experiment with different models and features to improve your score.

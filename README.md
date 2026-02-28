# Credit Score Classification 

A machine learning project to **predict individual credit score categories** — **Good / Standard / Poor** — based on demographic and financial behavior features.

## Overview
Credit scores are widely used by banks and financial institutions to assess credit risk (loan approvals, credit limits, interest rates, etc.).  
This project formulates credit score prediction as a **multi-class classification** task and builds several models to automate and improve decision-making efficiency.

## Dataset
- **Source:** Kaggle – *Credit Score Classification* dataset  
- **Size:** ~100,000 rows, 28 columns  
- **Target:** `Credit_Score` (3 classes: Good / Standard / Poor)
- **Features:** a mix of numerical and categorical attributes (e.g., `Age`, `Annual_Income`, `Num_Bank_Accounts`, `Interest_Rate`, `Num_Credit_Inquiries`, `Credit_Utilization_Ratio`, etc.)

## Data Cleaning & Preprocessing
Key preprocessing steps:
- **Dropped non-informative / identifier-like columns** (e.g., `ID`, `Customer_ID`, `Month`, `Name`, `SSN`, etc.)
- **Fixed malformed string values** in numeric columns (e.g., underscores or random symbols → converted to `NA`)
- **Handled invalid values** (e.g., meaningless negative values in `Age`, `Num_Bank_Accounts`, `Num_of_Loan`)
- **Outlier treatment (IQR rule):** values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` were replaced as `NA`, then imputed with **median**
- Ensured correct **data types** across features

## Exploratory Data Analysis (IDA)
- Checked missingness and confirmed missing ratio was manageable
- Visualized numeric distributions (density plots)
- Re-checked distributions with boxplots after outlier handling to ensure values fall within reasonable ranges

## Models
We trained and evaluated 4 classification models using an **80/20 train-test split**:

1. **K-Nearest Neighbors (KNN)**
   - Cross-validation grid search over `k`
   - Best `k = 20`

2. **Logistic Regression (Multinomial via glmnet)**
   - Hyperparameter tuning via **grid search** over `(alpha, lambda)`
   - **10-fold cross-validation**
   - Strong interpretability for financial feature effects

3. **Random Forest**
   - `ntree = 100`
   - Strong performance on mixed feature types and non-linear patterns

4. **Linear Discriminant Analysis (LDA)**
   - Repeated cross-validation training

## Results (Test Set Accuracy)
| Model | Accuracy |
|------|----------|
| Random Forest | **0.787** |
| KNN (best k=20) | **0.720** |
| Logistic Regression | **0.674** |
| LDA | **0.670** |

✅ **Best overall model:** **Random Forest** (highest accuracy and balanced per-class metrics)

## Tech Stack
- **R**
- Key libraries (typical): `tidyverse`, `caret`, `glmnet`, `randomForest`, `MASS`, `ggplot2`, `reshape2`, etc.


1. Install dependencies in R:
```r
install.packages(c("tidyverse", "caret", "glmnet", "randomForest", "MASS", "reshape2", "ggplot2"))

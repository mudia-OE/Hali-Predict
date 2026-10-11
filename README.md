# Hali Predict 🏀
### Predicting NBA Player Performance Following an Achilles Injury

Hali Predict is an NBA analytics project investigating how an Achilles injury may affect a player's statistical performance upon returning to the court. Using historical player statistics and regression analysis, the project explores how comparable players performed before and after similar injuries to estimate potential post-injury outcomes for Indiana Pacers guard Tyrese Haliburton.

The project combines sports analytics, statistical modelling, and data-driven investigation to explore a challenging question in basketball: how much can historical player data tell us about an individual player's return from a major injury?

## Project Objectives

- Analyze historical NBA player performance before and after comparable Achilles injuries.
- Identify statistical patterns that may help explain post-injury performance changes.
- Apply regression modelling to estimate player performance using historical data.
- Evaluate predictive accuracy using mean absolute error (MAE).
- Explore the potential and limitations of using historical player comparisons to inform basketball analysis.

## Research Question

**What can the historical performance of NBA players returning from comparable injuries tell us about Tyrese Haliburton's potential statistical performance following an Achilles injury?**

Rather than assuming every player recovers in the same way, the project uses historical performance data to investigate patterns and generate a data-informed estimate.

## Methodology

### 1. Historical Player Analysis
Collected and examined player statistics from before and after comparable injuries to investigate changes in performance following an injury.

### 2. Data Preparation
Prepared player statistics for comparison, including per-36-minute metrics to help account for differences in playing time.

### 3. Regression Modelling
Applied regression analysis to model statistical performance using historical player data.

### 4. Model Evaluation
Evaluated predictive performance using mean absolute error (MAE), which measures the average absolute difference between predicted and actual values.

**Reported MAE: approximately 1.67** on the evaluated data.

*Note: The exact prediction target, units, evaluation procedure, and test-set setup should be documented alongside the final model results.*

## Tools & Technologies

- **Python** for data analysis and modelling
- **pandas** for data manipulation
- **scikit-learn** for regression modelling and evaluation
- **NumPy** for numerical computation
- **NBA player statistics** for historical performance comparisons

*The tools and data sources listed above should be confirmed against the final code and project files.*

## Key Findings

The project investigates whether historical injury-return performance can provide useful estimates of future player production.

The model's reported MAE of approximately 2.61 provides an initial measure of prediction error on the evaluated data. Its practical interpretation depends on the specific statistic being predicted and how the model was tested.

The project also highlights an important limitation of sports prediction: individual recovery, age, playing style, role, minutes, and team context can all affect performance. Historical comparisons cannot guarantee an individual player's outcome.

## Limitations

- The number of comparable players may be limited.
- Players can differ substantially in age, role, physical characteristics, and recovery circumstances.
- Per-36-minute statistics help standardize playing time but do not eliminate differences in usage or playing role.
- Historical relationships do not establish that an injury alone caused a change in performance.
- Model accuracy depends on data quality, feature selection, and evaluation methodology.

## Project Structure

```text
hali-predict/
├── README.md
├── achilles_dataset/             # Dataset files
├── train_haliburton_model.py/              # Analysis and modelling scripts
```

*Adjust this structure to match the actual repository.*

## Future Improvements

- Expand the sample of comparable players where reliable data are available.
- Compare regression with alternative modelling approaches.
- Evaluate the model on held-out data and document the validation strategy.
- Examine prediction errors across different performance metrics.
- Improve visualizations to communicate pre-injury and post-injury trends.
- Incorporate relevant contextual variables where sufficient data are available.

## Purpose

Hali Predict is an independent sports analytics project exploring the intersection of basketball, injury recovery, and statistical modelling. It demonstrates an approach to turning a basketball question into a quantitative investigation while recognizing the uncertainty involved in predicting individual performance.

---

**Author:** Divine Osas-Egbe  
**Project:** Hali Predict  
**Focus:** NBA Analytics · Regression · Sports Data Science

# Life Expectancy Prediction with Multiple Linear Regression

Predicts national life expectancy from WHO health, economic, and social
indicators, with a Streamlit web app for interactive prediction.

Course project for SEP 786 (Artificial Intelligence and Machine Learning
Fundamentals), W Booth School of Engineering Practice and Technology,
McMaster University — November 2023.

## Results

| Model | MAE | MSE | RMSE |
|-------|-----|-----|------|
| Linear Regression | 1.360 | 5.868 | 2.422 |
| Ridge (RidgeCV) | 1.362 | 5.854 | 2.420 |
| Lasso (LassoCV) | 3.253 | 18.489 | 4.300 |
| Elastic Net (ElasticNetCV) | 1.393 | 6.133 | 2.477 |

Plain linear regression reached an R² of 0.937 on the held-out test set,
predicting life expectancy to within about 1.4 years on average.

Regularization did not help. Ridge matched the unregularized model almost
exactly, Elastic Net was marginally worse, and Lasso was substantially worse —
its L1 penalty zeroes out coefficients, and with 192 one-hot country columns
that discards most of the model's information.

## Dataset

WHO Global Health Observatory data joined with UN economic data, covering 193
countries from 2000 to 2015. 2,938 rows, 22 columns (20 numeric, 2
categorical). Sourced from Kaggle.

Target: `life_expectancy` (years).

Predictors span four groups:

- **Immunization** — hepatitis B, polio, diphtheria (DTP3) coverage
- **Mortality** — adult mortality, infant deaths, under-five deaths, HIV/AIDS,
  measles
- **Economic** — GDP per capita, percentage expenditure, total expenditure,
  income composition of resources
- **Social** — schooling, BMI, alcohol, population, thinness (5–9 and 10–19),
  country, developed/developing status

## Preprocessing

**Missing values.** Ten columns had nulls, the worst being population (652),
hepatitis B (553), and GDP (448). Mean imputation was rejected because of
outliers. Country-wise interpolation failed for two reasons: some countries
are null across all 16 years, and interpolation cannot fill a leading null.
Final approach was year-wise median imputation.

**Outliers.** IQR capping was tested and discarded — it cut the dataset from
2,938 rows to 1,132. Since the model predicts per country, extreme values for
a given country are legitimate observations rather than noise. Low-schooling
and low-income-composition points with high life expectancy were kept for the
same reason.

**Redundancy.** `under-five deaths` and `infant deaths` correlate at 1.00, so
one was dropped.

**Categoricals.** Country (193 levels) and status (2 levels) one-hot encoded,
producing 192 country columns. Status encoded with `drop_first=True`, leaving
a single "Developing" indicator.

## Modeling

- 80/20 train/test split, `random_state=0`
- Features standardized with `StandardScaler`
- `LinearRegression` baseline, then `RidgeCV`, `LassoCV`, and `ElasticNetCV`
  with cross-validated alpha selection

## Exploratory Findings

- Adult mortality shows the strongest negative relationship with life
  expectancy; developed countries cluster at low mortality and high longevity
- Schooling correlates at 0.71 with life expectancy, the strongest single
  positive predictor
- Health expenditure as a share of government spending correlates at only
  0.22 — spending more on healthcare, by itself, does not track with longer
  life
- Alcohol consumption shows little relationship; developed countries consume
  more and still live longer

## Deployment

The trained model is pickled to `.sav` and served through a Streamlit app that
takes the predictor values as form inputs and returns a life expectancy
estimate.

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn plotly
streamlit run app.py
```

## Repository Layout

```
notebooks/
  Life_Expectancy_FINAL_Group_6.ipynb
data/
  Life Expectancy Data.csv
models/
  general_life_expectancy_ui.sav
app.py
docs/
  project_report.pdf
```

## Known Limitations

- One-hot encoding all 193 countries means the model largely learns a per-
  country baseline. The 0.937 R² is optimistic: with 16 rows per country and
  a random split, most test rows have sibling rows from the same country in
  training. A country-held-out split would be the honest test of whether the
  health and economic features generalize.
- Year-wise median imputation ignores country-level structure, so imputed
  values for outlier countries are pulled toward the global middle.
- Linear regression assumes independent observations, but the data is a panel
  — 16 correlated years per country.
- Everything here is correlational. The coefficients do not license causal
  claims about what raises life expectancy.

## Team

Harikrishna Ramanathan Balakrishnan, Lavanya Singh, Suraj Ramesh, and Vipin
Chandran Muthirikkaparambil. Course instructor: Sayyed Faridoddin Afzali.

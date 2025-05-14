---

# Weather Forecasting Model (Prophet + Optuna)

## 📊 Overview

This project uses **Prophet** for time series forecasting, enhanced with **Optuna** for hyperparameter optimization. It predicts the temperature (in Celsius) for the next 7 days in **Gauteng, South Africa**, using historical temperature and humidity data.

Improved and extended the original `forecast.py` script to address bugs, improve accuracy, increase flexibility, and implement CLI-based usability.

---

## 🔧 Features Implemented

### ✅ Bug Fixes

* Proper exception handling for:

  * Missing or unreadable CSV file.
  * Invalid/missing columns (`ds`, `temperature_celsius`, `humidity`).
  * Invalid datetime parsing.
* Ensured output forecast includes required columns: `ds`, `yhat`, `yhat_lower`, `yhat_upper`.

### ⚡ Optimization

* Refactored code for readability and modularity.
* Improved variable names for clarity (e.g. `df` → `weather_df`, `e` → `forecast_error`).
* Deprecated methods like `suggest_loguniform` replaced with `suggest_float(..., log=True)`.
* Fixed `FutureWarning` by replacing `.fillna(method=...)` with `.bfill()` and `.ffill()`.

### ✨ Feature Additions

* **Command Line Interface (CLI)** with:

  * `--input`: path to input CSV file
  * `--periods`: number of days to forecast (default: 7)
  * `--output`: optional path to save the forecast CSV file
* **Humidity** used as a regressor for improved accuracy.
* **Additional Regressors**:

  * `humidity_lag1` (1-day lag)
  * `temp_rolling3` (3-day rolling average of temperature)
* **Dark Mode Forecast Plots** with clear color scheme and labelling.
* Two plots:

  * Full timeline forecast (`Historical and Future Forecast.png`)
  * Zoomed-in 7-day forecast (`Next 7 days Forecast.png`)

### 🤖 Prophet Enhancements

* Used **Optuna** for automatic hyperparameter tuning:

  * `changepoint_prior_scale`
  * `seasonality_prior_scale`
  * `holidays_prior_scale`
  * `changepoint_range`
  * `seasonality_mode`
* Cross-validation added to evaluate model performance using RMSE.
* Seasonal and holiday parameter tuning boosts prediction accuracy around seasonal shifts (e.g. summer/winter).

### 🧪 Unit Tests

* Unit tests added in `tests.py` using `pytest`:

  * ✅ Data loading and validation
  * ✅ Model training
  * ✅ Future dataframe generation
  * ✅ Forecast output structure

---

## ✅ Submission Checklist

* [x] Updated `forecast.py` with bug fixes, enhancements, and CLI support
* [x] CLI arguments: `--input`, `--periods`, `--output`
* [x] Forecast plots generated (`Historical and Future Forecast.png`, `Next 7 days Forecast.png`)
* [x] Forecast output CSV supported (via `--output`)
* [x] Graceful error handling and data validation
* [x] Modular, readable, well-commented code
* [x] Optimized and tuned model via Optuna
* [x] Unit tests implemented using `pytest`
* [x] Documentation written in `README.md`
* [x] All work organized under `forecast.py`
* [x] GitHub branch `/submissions/<candidate_name>` used
* [x] Commit messages are clear and descriptive

---

## 🌐 Usage Instructions

### ➤ Run Forecast from Command Line

```bash
python forecast.py --input weather.csv --periods 7 --output forecast_output.csv
```

### ➤ Run Tests

```bash
pytest tests.py
```

---

## ✨ Final Notes

This project demonstrates:

* A complete ML workflow: ingestion → processing → training → tuning → forecasting → output
* Thoughtful use of time series modeling, cross-validation, and AI-based hyperparameter tuning
* A clean, modular, and production-friendly architecture with test coverage and CLI

---

⚡ By **Mitheel Ramdaw**
✉️ [mitheelramdaw@gmail.com](mailto:mitheelramdaw@gmail.com)

---

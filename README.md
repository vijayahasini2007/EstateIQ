# 🏙️ EstateIQ – Real Estate Market Analytics & Insights

## 🚀 Live Demo

**Explore EstateIQ:** [Open Live Dashboard](https://estateiq-5znnyjpdcs4uf5h4ayadqe.streamlit.app/)

Explore interactive real estate analytics, property price trends, location comparisons, rental insights, and ML-based price estimation.

EstateIQ is an interactive Streamlit dashboard for exploring property prices, location-level differences, property characteristics, rental metrics and indicative price estimates.

> **Important:** The included sample dataset is synthetic demonstration data generated for this project. It is not actual property listings or verified market prices. Use a properly licensed real dataset before making real-world market claims.

## Features
- Market overview with property count, median price, price per sq.ft. and median area
- City and locality comparisons
- Property price distributions by BHK and property type
- Area-versus-price analysis
- Rental metrics and gross rental-yield calculation when rent data is available
- Random-forest price estimator with holdout R² and MAE
- City, BHK and property-type filters
- Upload your own CSV dataset
- Download filtered records as CSV

## Technology
Python, Streamlit, Pandas, NumPy, Plotly and Scikit-learn.

## Project structure
```text
EstateIQ/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── raw/
│       └── estateiq_sample_synthetic.csv
├── src/
│   ├── __init__.py
│   └── analytics.py
└── notebooks/
```

## Run locally
Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install and run:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Use your own dataset
Upload a CSV from the sidebar. The required columns are:
- `city`
- `locality`
- `property_type`
- `bhk`
- `area_sqft`
- `price_inr`

Optional columns:
- `monthly_rent_inr`
- `property_age_years`
- `amenities_count`
- `furnishing`
- `listing_date`
- `price_per_sqft`

The app normalizes column names to lowercase and replaces spaces with underscores. Ensure numeric fields contain numeric values and prices are consistently denominated in INR.

## Analytical methodology
- Median property price and median area are used to reduce sensitivity to extreme values.
- Price per square foot is calculated as `price_inr / area_sqft` if not provided.
- Gross rental yield is `monthly_rent_inr × 12 / price_inr × 100`. It excludes vacancy, taxes, maintenance, fees and financing costs.
- The estimator uses a Random Forest regression pipeline with one-hot encoding for categorical features and median/mode imputation. It uses a random 80/20 train/test split and reports R² and mean absolute error.
- Model quality depends on dataset representativeness, data quality and sample size. The score is not a guarantee of future accuracy.

## Data and limitations
The bundled dataset is synthetic and is included only to demonstrate the application's functionality. It must not be presented as real market evidence. Replace it with a dataset whose license permits your intended use. Record the dataset source, access date, geographic coverage, unit definitions and license in your project documentation.

## Disclaimer
EstateIQ is an educational analytics project, not a property valuation service, investment recommendation or financial/legal advice. Any price estimate is an experimental model output, not a formal valuation.

## Deploy on Streamlit Community Cloud
1. Push the project to a GitHub repository.
2. Sign in at https://share.streamlit.io/.
3. Create an app using your repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Deploy. Dependencies are installed from `requirements.txt`.

## Future enhancements
- Integrate a verified, licensed real-estate dataset
- Add map visualizations after sourcing reliable latitude/longitude
- Add time-series analysis using verified listing dates
- Add benchmark comparisons and model cross-validation
- Add automated tests for data cleaning and analytics

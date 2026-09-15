# Dataset

This project uses the **Rain in Australia (WeatherAUS)** dataset for predicting whether it will rain the following day.

## Dataset Source

The dataset is available on Kaggle:

**Rain in Australia**
- Dataset: Rain in Australia / WeatherAUS
- Target variable: `RainTomorrow`
- Observations: 145,460
- Features: 23 columns

The dataset contains daily weather observations collected from multiple locations across Australia.

## Files Used

The original dataset contains the following files:

- `weatherAUS.csv` — Main dataset containing weather observations.

The CSV file is **not included in this repository** because it is large and is available publicly through Kaggle.

## How to Get the Dataset

1. Download the dataset from Kaggle.
2. Extract `weatherAUS.csv`.
3. Place it inside this `data/` directory.

The expected structure is:

```text
weather-australia-rain-prediction/
│
├── data/
│   └── README.md
│
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── weather.ipynb
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md

# Q * model

## Description

This project focuses on using machine learning (ML) models to identify outperforming stocks in a portfolio before they peak, allowing for strategic fund allocation. Additionally, it employs InterpretML to provide explanations for the model's predictions using techniques such as LIME and SHAP.

This project is supported by QuantX and Quant Illinois.

## How to Find Outperforming Stock

Predicting profit directly can be challenging, so we explore categorical labels and temporal models (K bars) considering margin balance and short balance. The trend is more important than the actual numbers.

## Data
https://uillinoisedu-my.sharepoint.com/:f:/g/personal/boyusu2_illinois_edu/EjNQqI4VOoxMneCp1kiJASgBqBwuI5NpXtaXKNb_zWujJw

Using the top_100_preprocessed_data/result.csv file as the dataset to train and test the machine learning model.

## Features

### Technical Indicators (TA-Lib)

#### Goals:
1. Filter out indicators that make no sense.
2. If an indicator can be divided by moving average (MA), keep it and normalize by MA.

#### Criteria for Selecting Features:
1. Alignment across different stocks.
2. Entropy: Categorical features should have enough information.
3. Stability over different periods (from training to testing set).

### Momentum Indicators (Josh)
- **Drop**: `'adx'`, `'adxr'`, `'apo'`, `'bop'`, `'cci'`, `'cmo'`, `'dx'`, `'macd'`, `'macd_x'`, `'mfi'`, `'minus_di'`, `'minus_dm'`, `'plus_di'`, `'plus_dm'`, `'ppo'`, `'roc'`, `'rocr'`, `'rocp'`, `'rocr100'`, `'rsi'`, `'trix'`, `'ultosc'`, `'willr'`
- **Keep**: `'aroonup'`, `'aroondown'`, `'aroonosc'`, `'mom'`

### Overlap Studies (Josh)
- **Drop**: `'dema'`, `'ema'`, `'kama'`, `'ma'`, `'mama'`, `'sma'`, `'t3'`, `'tema'`, `'trima'`, `'wma'`, `'upperband'`, `'lowerband'`, `'ht_trendline'`, `'midpoint'`, `'midprice'`, `'sar'`, `'sarext'`
- **Keep**: `'middleband'`, `'midpoint'`, `'midprice'`

### Volume Indicators (Wenxuan)
- **Drop**: Usually use the trend, not the absolute value; normalize by volume.

### Volatility Indicators (Wenxuan)
- **Drop**: TR, ATR
- **Keep**: NATR (normalized), useful for estimating volatility
- **Description**: [NATR Indicator](https://tulipindicators.org/natr)

### Price Transform (Josh)
- **Drop**: `'avgprice'`, `'medprice'`, `'typprice'`
- **Keep**: `'wclprice'`

### Cycle Indicators (Wenxuan)
- **Keep**: HT_DCPHASE (-45 to 315)
- **Drop**: HT_PHASOR (proportional to stock price)
- **Pending**: HT_DCPERIOD, HT_SINE (similar to HT_DCPHASE), HT_TRENDMODE (always returns 1 or 0 instead of -1 as documented)

### Pattern Recognition (Tony, James)
- **Drop**: `'cdl3starsinsouth'`, `'cdlabandonedbaby'`, `'cdlconcealbabyswall'`
- **Almost all 0, count(-100) < 50**: `'cdlbreakaway'`, `'cdl2crows'`, `'cdl3blackcrows'`, `'cdleveningdojistar'`
- **Almost all 0, count(100) < 50**: `'cdl3linestrike'`, `'cdl3whitesoldiers'`
- **Pending**: `'cdladvanceblock'`, `'cdldarkcloudcover'`, `'cdleveningstar'`, `'cdlhangingman'`
- **No +100**: `'cdldoji'`, `'cdldragonflydoji'`, `'cdlgravestonedoji'`, `'cdlhammer'`

### Margin Data
1. **Explosive Volume of Daily Margin Balance Change**:
    - Compare with the past 20 days for sudden volume increases.
    - Largest volume in the past 20 days.
    - Volume increase > 5 times the average of the past 20 days.
2. **Explosive Volume of Daily Short Balance Change**:
    - Similar criteria as margin balance.
3. **Stock Price Consolidation at Low Level** (optional):
    - Price range within ±5% for the past 20 days.

## Label

### Classifier
1. **Binary Classification**:
    - PR 97 (above 24%) → 0
    - PR 97 (below 24%) → 1
2. **4-Class Classification**:
    - PR 97 (above 24%) → 0
    - PR 90 (13%-24%) → 1
    - 1.08%-13% → 2
    - Below 1.08% → 3

#### Profit Statistics (Top 100 Raw Data)
- count: 12019
- mean: 1.081814
- std: 11.182851
- min: -46.226415
- 25%: -4.851116
- 50%: 0.423131
- 75%: 5.688666
- max: 142.248062

### Percentile Thresholds
- PR_50: 0.423131
- PR_60: 2.163202
- PR_70: 4.341017
- PR_80: 7.407407
- PR_90: 13.062045
- PR_97: 24.036847
- PR_99: 36.552946

## Model

- **Classification**: xgboost, lgbm
- **Regression**

## Portfolio Optimization Method

- **Initial Portfolio**: $10k across 10 stocks, e.g., 0001: $1k, 0002: $1k
- **Classification Models**:
    - Allocate funds based on class predictions, e.g., 0001: 0:70%, 1:30%
- **Regression Models**:
    - Allocate based on expected return and variance, e.g., 0001: expected return 25%, variance 10% → $0.5k

### Allocation Strategy
1. Stocks with > 13% expected return: 20% of the portfolio.
2. Remaining stocks: average allocation.

## Backtesting

Integrate the ML model and allocation strategy into the original quantx trading engine backtest system

## Contribution

The QStar model utilizing machine learning algorithms (XGBoost, LGBM) to identify outperforming equities and rebalance the fund allocation within the QuantX portfolio, increasing return by 2.32% and lowering drawdown by 3.72%.

## Explainable AI (XAI)

[InterpretML](https://github.com/interpretml/interpret?source=post_page-----7c7c37ae30f7--------------------------------')

### SHAP (SHapley Additive exPlanations)
- Analyzes model predictions into the contribution of each factor.
- Visual analysis suite in Python.
- [SHAP Example](https://blog.infuseai.io/use-pokemon-dataset-to-explin-the-ml-model-explainable-ai-7c7c37ae30f7)

### LIME
- Provides explanations for complex models.
- [LIME Example](https://medium.com/sherry-ai/xai-%E9%80%8F%E9%81%8E-lime-%E8%A7%A3%E9%87%8B%E8%A4%87%E9%9B%9C%E9%9B%A3%E6%87%82%E7%9A%84%E6%A8%A1%E5%9E%8B-23898753bea5)


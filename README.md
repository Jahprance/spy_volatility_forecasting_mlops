# SPY Next-Day Volatility Forecasting MLOps Service

This project forecasts **next-day SPY Parkinson variance** using daily OHLCV data.

The main goal was not to build a trading system. The purpose is to develop and
package a simple, leakage-safe volatility forecasting model with practical MLOps
components such as experiment tracking, model artifacts, API, testing, Docker
and CI.

> This project is not financial advice and I make no claims reg profitability

Daily SPY OHLCV data was obtained using the Twelve Data API. The API key is entered
interactively in the notebook and is not included in this repository.

## Project goal

The target is next-day Parkinson variance:

\[
\text{target\_var}(t) = \hat{\sigma}^{2}_{P,t+1}
\]

Features are calculated using information available by the close of trading day
\(t\), then used to forecast the next trading day \(t+1\).

The selected model is a scikit-learn pipeline--

```text
StandardScaler → Ridge(alpha=1.0)
```

## Features

The final HAR-style Ridge model uses 7 features:

- `har_d`: current-day Parkinson variance
- `har_w`: 5-trading-day average Parkinson variance
- `har_m`: 22-trading-day average Parkinson variance
- `rv_lag1`: previous-day Parkinson variance
- `abs_return`: absolute daily log return
- `hl_range`: high-low range scaled by close
- `log_volume`: log trading volume

The daily, weekly and monthly HAR-style features represent short, medium and
longer volatility history.

## Evaluation

Used chronological train, validation and test splits. I also compared the model
against a persistence baseline, where tomorrow's variance is predicted as today's
variance.

| Period | Persistence RMSE | HAR-Ridge RMSE | Improvement |
|---|---:|---:|---:|
| Validation | 0.000214 | 0.000194 | 9.34% lower |
| Final test | 0.000062 | 0.000050 | 19.76% lower |

I also used four expanding-window walk-forward folds. HAR-Ridge had lower RMSE than
persistence in all four folds, with a median improvement of **17.44%**.

The test-period visual comparison is available in:

```text
reports/figures/test_har_var_interactive.html
```

## MLOps components

- MLflow logs model parameters, metrics, artifacts and the promotion decision.
- The trained `StandardScaler + Ridge` pipeline is saved as a Joblib artifact.
- Model metadata includes feature order and a SHA-256 artifact checksum.
- FastAPI exposes `/health`, `/predict`, and `/monitor` endpoints.
- Pydantic rejects invalid negative variance-style inputs.
- `/monitor` compares request features against training-only q10-q90 reference
  ranges. This is a simple input-reference warning, not formal drift detection.
- Pytest covers health, valid prediction, invalid input and input monitoring.
- Docker packages the API and model artifact.
- GitHub Actions is configured to run tests and build the Docker image.

## Repository structure

```text
app/                 FastAPI service, schema, and monitoring logic
models/              Saved model, training reference, and metadata
notebooks/           Experiment and evaluation notebook
reports/             Evaluation tables and interactive test plot
tests/               Pytest API tests
.github/workflows/   CI workflow
Dockerfile           Container definition
requirements.txt     Pinned dependencies
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -q
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -i http://127.0.0.1:8000/health
```

Example prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "har_d": 0.00003,
    "har_w": 0.00004,
    "har_m": 0.00005,
    "rv_lag1": 0.00003,
    "abs_return": 0.01,
    "hl_range": 0.016,
    "log_volume": 18.0
  }'
```

## Docker

```bash
docker build -t spy-volatility-mlops:local .
docker run --rm -p 8000:8000 spy-volatility-mlops:local
```

Then call:

```bash
curl -i http://127.0.0.1:8000/health
```

## Current status

Verified in Colab:

- Repository FastAPI `/health`, `/predict`, and `/monitor` checks.
- Negative input rejection with HTTP 422.
- `pytest -q` passed with 4 tests.

Still to verify:

- Docker build and running-container health check.
- GitHub Actions workflow after the first push.
- Optional public cloud deployment.

## Limitations

- The API accepts precomputed features. It does not fetch live market data or create
  features from raw OHLCV requests.
- Parkinson variance does not fully capture overnight price gaps.
- This is a point-forecasting model. It does not provide uncertainty intervals.
- The input q10-q90 check is not a full drift-detection system.

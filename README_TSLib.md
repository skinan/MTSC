# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

```
Time-Series-Library/
├── README.md                     # Official README with tasks, leaderboard, usage
├── requirements.txt              # pip dependency list for quick environment setup
├── LICENSE / CONTRIBUTING.md     # Upstream license and contribution guide
├── run.py                        # Unified entry that parses args and dispatches tasks
├── exp/                          # Task pipelines wrapping train/val/test
│   ├── exp_basic.py              # Experiment base class, registers models, builds flows
│   ├── exp_long_term_forecasting.py    # Long-term forecasting logic
│   ├── exp_short_term_forecasting.py   # Short-term forecasting logic
│   ├── exp_imputation.py               # Missing-value imputation
│   ├── exp_anomaly_detection.py        # Anomaly detection
│   ├── exp_classification.py           # Classification
│   └── exp_zero_shot_forecasting.py    # LTSM zero-shot evaluation
├── data_provider/                # Dataset loaders and splits
│   ├── data_factory.py           # Chooses the proper DataLoader per task
│   ├── data_loader.py            # Generic TS reader with sliding-window logic
│   ├── uea.py / m4.py            # Parsers for UEA, M4 and other formats
│   └── __init__.py               # Exposes factory interfaces upward
├── models/                       # All model implementations
│   ├── TimesNet.py, TimeMixer.py # Main forecasting models
│   ├── Chronos2.py, TiRex.py     # LTSM zero-shot models
│   └── __init__.py               # Enables name-based instantiation inside exp
├── layers/                       # Reusable attention / conv / embedding blocks
│   ├── Transformer_EncDec.py     # Transformer stacks
│   ├── AutoCorrelation.py        # Auto-correlation operator
│   ├── MultiWaveletCorrelation.py# Frequency-domain unit
│   └── Embed.py etc.             # Shared primitives
├── utils/                        # Utility toolbox
│   ├── metrics.py                # MSE / MAE / DTW and other metrics
│   ├── tools.py                  # General helpers such as EarlyStopping
│   ├── augmentation.py           # Augmentations for classification / detection
│   ├── print_args.py             # Unified argument printer
│   └── masking.py / losses.py    # Task-specific helpers
├── scripts/                      # Bash recipes for reproducible experiments
│   ├── long_term_forecast/       # Long-term forecasting per dataset/model
│   ├── short_term_forecast/      # M4 and other short-term scripts
│   ├── imputation/               # Imputation scripts
│   ├── anomaly_detection/        # SMD / SMAP / SWAT detection scripts
│   ├── classification/           # UEA classification scripts
│   └── exogenous_forecast/       # TimeXer exogenous forecasting flow
├── tutorial/                     # TimesNet tutorial notebook and figures
└── pic/                          # README figures (dataset overview, etc.)
```

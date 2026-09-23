# Data

This project uses the Kaggle "Home Credit Default Risk" dataset.

## How to download
To download the data, ensure you have the Kaggle CLI installed and authenticated (e.g. `~/.kaggle/kaggle.json` is set up with an API token from your Kaggle account settings).

```bash
uv pip install kaggle
kaggle competitions download -c home-credit-default-risk -p data
unzip data/home-credit-default-risk.zip -d data/
```

Required files:
- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`

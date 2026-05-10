# BOT-YEDA Trading Bot

Bot de trading algorítmico modular para investigación, backtesting y simulación. Por seguridad, el proyecto inicia en `backtest` y no opera en cuenta real por defecto.

## Seguridad

- Modos soportados: `backtest`, `paper`, `demo`, `real`.
- `real` está bloqueado salvo que `ENABLE_REAL_TRADING=true`.
- No hay credenciales hardcodeadas.
- Las credenciales futuras de brokers deben ir en `.env`.
- No se implementa martingala agresiva.
- El risk manager bloquea operaciones con baja confianza, más de 3 pérdidas consecutivas, pérdida diaria mayor al 5%, datos insuficientes, mercado lateral o volatilidad anormal.

## Formato CSV

El archivo debe tener al menos:

```csv
timestamp,open,high,low,close,volume
2026-01-01T00:00:00Z,100,101,99,100.5,1000
```

El archivo base esperado es `data/raw/candles.csv` y debe incluir todas esas columnas.

## Instalación

```bash
cd bot-trading
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Backtest por CLI

```bash
python -m app.main backtest --csv data/raw/candles.csv --output data/processed/trades.csv
```

## Entrenar modelo ML

```bash
python -m app.main train --csv data/raw/candles.csv --output models/best_model.joblib
```

## Comparar reglas vs ML

```bash
python -m app.main compare --csv data/raw/candles.csv --model models/model.joblib
```

## API

```bash
uvicorn app.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /backtest`
- `POST /train`
- `POST /compare-strategies`

Ejemplo:

```json
{
  "csv_path": "data/raw/candles.csv",
  "initial_balance": 1000,
  "save_to_db": true
}
```

## Docker

```bash
copy .env.example .env
docker compose up
```

## Tests

```bash
pytest
```

## Estado de brokers

`PaperBroker` está disponible para simulación. `IQOptionBroker` y `ExnovaBroker` son placeholders seguros para una futura integración en demo, sin operaciones reales.

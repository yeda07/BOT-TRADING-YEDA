# BOT-YEDA Trading Bot

BOT-YEDA es un bot de trading algorítmico modular para investigación, backtesting, entrenamiento de modelos de Machine Learning y simulación en modo paper/demo.

El proyecto está diseñado para trabajar primero con datos históricos en CSV, validar estrategias de forma reproducible y operar únicamente en entornos simulados. El sistema analiza velas de mercado, calcula indicadores técnicos, genera señales por reglas o ML, ejecuta backtests de opciones binarias, controla riesgo y deja preparada una arquitectura extensible para brokers, sin habilitar operación real por defecto.

## Advertencia de Riesgo

Este software es educativo. No garantiza ganancias. El trading con dinero real implica riesgo de pérdida total. Usar primero en cuenta demo.

No uses este proyecto para operar dinero real sin auditoría técnica, validación estadística, pruebas fuera de muestra, control de riesgo estricto y autorización explícita. El modo `real` está bloqueado por defecto.

## Instalación

Desde la raíz del repositorio:

```bash
cd bot-trading
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Para verificar la instalación:

```bash
pytest
```

## Configuración

La configuración se carga desde `.env` usando `pydantic-settings`. Copia `.env.example` a `.env` y ajusta los valores según tu entorno.

Variables principales:

```env
BOT_MODE=backtest
BROKER=paper
ASSET=EURUSD-OTC
TIMEFRAME_SECONDS=60
EXPIRATION_CANDLES=1
INITIAL_BALANCE=3000
PAYOUT=0.87
RISK_PER_TRADE=0.01
MAX_DAILY_LOSS=0.05
MAX_CONSECUTIVE_LOSSES=3
MIN_CONFIDENCE=0.58
ENABLE_REAL_TRADING=false
```

Reglas de seguridad:

- `BOT_MODE` solo puede ser `backtest`, `paper`, `demo` o `real`.
- Si `BOT_MODE=real` y `ENABLE_REAL_TRADING=false`, el sistema falla al iniciar.
- `RISK_PER_TRADE` no puede ser mayor a `0.02`.
- `MAX_DAILY_LOSS` no puede ser mayor a `0.10`.
- No hay credenciales hardcodeadas.
- Las credenciales futuras de brokers deben vivir en `.env`.

## Formato del CSV

El archivo base esperado es:

```text
data/raw/candles.csv
```

Debe incluir estas columnas:

```csv
timestamp,open,high,low,close,volume
2026-01-01T00:00:00Z,100,101,99,100.5,1000
```

Validaciones aplicadas:

- Todas las columnas requeridas deben existir.
- `timestamp` debe ser una fecha válida.
- Los datos deben estar ordenados por fecha ascendente.
- `open`, `high`, `low` y `close` no pueden estar vacíos.
- Debe haber mínimo 200 velas para calcular indicadores y ejecutar estrategias con datos suficientes.

## Como Correr Backtest

Ejecuta un backtest con la estrategia por reglas:

```bash
python -m app.main backtest
```

También puedes pasar un CSV específico:

```bash
python -m app.main backtest --csv data/raw/candles.csv
```

Para exportar las operaciones a un archivo:

```bash
python -m app.main backtest --csv data/raw/candles.csv --output data/logs/backtest_results.csv
```

El backtester simula opciones binarias:

- `BUY` gana si el precio futuro sube.
- `SELL` gana si el precio futuro baja.
- `HOLD` no abre operación.

Métricas calculadas:

- Balance inicial y final.
- Profit neto.
- Total de trades.
- Wins y losses.
- Win rate.
- Profit factor.
- Max drawdown.
- Máximas pérdidas consecutivas.
- Equity curve.

## Como Entrenar Modelo

Entrena y compara modelos de Machine Learning:

```bash
python -m app.main train
```

También puedes definir el CSV y la ruta de salida:

```bash
python -m app.main train --csv data/raw/candles.csv --output models/best_model.joblib
```

Modelos evaluados:

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`

El split es temporal: los datos antiguos se usan para entrenamiento y los datos recientes para prueba. No se usa `shuffle=True`, para evitar mezclar el orden natural de la serie temporal.

Métricas:

- Accuracy.
- Precision.
- Recall.
- F1.
- ROC AUC.
- Matriz de confusión.
- Win rate simulado.
- Profit simulado.

El mejor modelo se guarda en:

```text
models/best_model.joblib
```

El artefacto guardado incluye el modelo, la lista de features, el nombre del mejor modelo y sus métricas.

## Como Ejecutar Paper Trading

Simula una operación usando las últimas velas del CSV:

```bash
python -m app.main paper
```

Con un CSV específico:

```bash
python -m app.main paper --csv data/raw/candles.csv
```

Este modo usa `PaperBroker`. No envía órdenes reales, no automatiza navegador y no interactúa con cuentas reales.

## Comparar Reglas vs ML

Compara la estrategia por reglas contra la estrategia ML usando el mismo motor de backtesting:

```bash
python -m app.main compare --csv data/raw/candles.csv --model models/best_model.joblib
```

## API Local

Servidor local:

```bash
uvicorn app.main:app --reload
```

Endpoints disponibles:

- `GET /health`
- `POST /backtest`
- `POST /train`
- `POST /compare-strategies`

Ejemplo de payload:

```json
{
  "csv_path": "data/raw/candles.csv",
  "initial_balance": 3000,
  "save_to_db": true
}
```

## Arquitectura

```text
app/
  brokers/
    base.py              Interfaz abstracta BrokerBase
    paper_broker.py      Broker simulado funcional
    iqoption_broker.py   Adaptador preparado, ejecución deshabilitada
    exnova_broker.py     Adaptador preparado, ejecución deshabilitada
  execution/
    backtester.py        Motor de backtesting de opciones binarias
    executor.py          Ejecución paper/demo sobre BrokerBase
    comparison.py        Comparación reglas vs ML
  market/
    candles.py           Carga y validación de CSV
    indicators.py        Indicadores técnicos
    features.py          Features y target para ML
  ml/
    train.py             Entrenamiento y selección de modelo
    predict.py           Predicción con modelo guardado
    evaluate.py          Métricas de clasificación
  risk/
    risk_manager.py      Control de riesgo y bloqueo de operaciones
  strategies/
    rule_based.py        Estrategia por reglas
    ml_strategy.py       Estrategia basada en modelo ML
  storage/
    database.py          Persistencia SQLite de trades
  utils/
    logger.py            Configuración de logs
```

Flujo principal:

1. `market/candles.py` carga y valida el CSV.
2. `market/features.py` calcula features e indicadores.
3. `strategies/` genera señales `BUY`, `SELL` o `HOLD`.
4. `risk/risk_manager.py` decide si una operación puede ejecutarse.
5. `execution/backtester.py` simula resultados y calcula métricas.
6. `ml/train.py` entrena modelos y guarda el mejor artefacto.
7. `brokers/paper_broker.py` permite simular ejecución sin operar dinero real.

## Brokers

`PaperBroker` está disponible para simulación local.

`IQOptionBroker` y `ExnovaBroker` están preparados como adaptadores futuros, pero la operación real/demo está deshabilitada. El proyecto no automatiza clics de navegador, no hace bypass de seguridad y no intenta saltarse términos del broker.

## Docker

```bash
copy .env.example .env
docker compose up
```

## Próximos Pasos

- Agregar validación walk-forward para modelos ML.
- Incorporar costos, slippage y latencia en el backtest.
- Crear reportes HTML con métricas, equity curve y matriz de confusión.
- Añadir monitoreo de drift de datos y performance.
- Implementar conexión demo mediante APIs oficiales y permitidas por cada broker.
- Agregar CI para ejecutar `pytest` en cada cambio.

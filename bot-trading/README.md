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
LIVE_MAX_STEPS=100
LIVE_SLEEP_SECONDS=1
TRADE_LOG_PATH=data/logs/live_trades.csv
TRADES_DB_PATH=data/logs/trades.db
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

Si ejecutas `python -m app.main backtest` o `python -m app.main train` y ese archivo no existe o no tiene velas suficientes, el CLI crea un dataset demo determinístico para que la primera ejecución sea funcional. Para usar tus propios datos, pasa la ruta con `--csv`.

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

## Machine Learning

La fase de Machine Learning entrena modelos para estimar si la vela futura cerrará arriba o abajo según `EXPIRATION_CANDLES`. El flujo usa split temporal, sin `shuffle`, para reducir data leakage:

- 70% de datos antiguos para entrenamiento.
- 15% para validación y selección del modelo.
- 15% de datos recientes para prueba.

Modelos comparados:

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `ExtraTreesClassifier`

Comandos principales:

```bash
pip install -r requirements.txt
python -m app.main train
python -m app.main compare
python -m app.main live-paper
python -m app.main summary
python -m app.main predict-latest
pytest
```

El entrenamiento guarda:

```text
models/best_model.joblib
data/logs/model_metrics.csv
```

La comparación de estrategias guarda:

```text
data/logs/ml_backtest_results.csv
data/logs/ml_equity_curve.csv
data/logs/strategy_comparison.csv
```

Interpretación de métricas:

- `AUC`: mide qué tan bien el modelo separa velas alcistas y bajistas en distintos umbrales.
- `F1`: balancea precision y recall; ayuda cuando accuracy puede ser engañoso.
- `win rate`: porcentaje de operaciones ganadas en la simulación binaria.
- `profit factor`: ganancia bruta dividida por pérdida bruta. Valores mayores a `1` son preferibles.
- `max drawdown`: caída máxima desde un pico de equity; mide riesgo y estabilidad.

El `breakeven win rate` es el porcentaje mínimo de aciertos requerido para no perder dinero según el payout:

```text
breakeven = 1 / (1 + payout)
```

Con `PAYOUT=0.87`, el modelo necesita superar aproximadamente `53.48%` de win rate antes de costos, slippage o latencia.

Un modelo con buen accuracy puede perder dinero si no supera el win rate mínimo requerido por el payout del broker. Por eso el mejor modelo no se elige solo por accuracy; se priorizan `roc_auc`, `f1`, win rate simulado sobre breakeven, `profit_factor` mayor a `1` y drawdown razonable.

`predict-latest` carga el mejor modelo y genera una señal para la última vela disponible:

```bash
python -m app.main predict-latest
```

Ejemplo:

```text
===== LATEST SIGNAL =====
Asset: EURUSD-OTC
Signal: BUY
Confidence: 0.61
Probability Up: 0.61
Decision: Trade allowed only in paper/demo mode
```

Las señales ML deben usarse primero en backtesting y paper/demo. Este proyecto no habilita operación real, no automatiza clicks de navegador y no guarda credenciales en código.

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

## Live Paper Trading

`live-paper` ejecuta el bot como si estuviera en vivo, pero usando velas de `data/raw/candles.csv`, `PaperBroker`, el modelo ML guardado y el `RiskManager`. El feed simula la llegada de velas una por una y el motor evita mirar datos futuros.

Flujo:

- Carga una ventana reciente de velas.
- Calcula indicadores y features.
- Predice con `models/best_model.joblib`.
- Valida confianza, riesgo y vela duplicada.
- Abre una orden binaria paper.
- Espera la vela de expiracion dentro del CSV.
- Resuelve `WON`, `LOST` o `PENDING`.
- Registra la operacion en CSV y SQLite.

Comandos:

```bash
python -m app.main train
python -m app.main compare
python -m app.main live-paper
python -m app.main summary
python -m app.main collect-data
python -m app.main data-quality
pytest
```

Con rutas explicitas:

```bash
python -m app.main live-paper --csv data/raw/candles.csv --model models/best_model.joblib
```

Archivos generados:

```text
data/logs/live_trades.csv
data/logs/trades.db
```

Resumen:

```bash
python -m app.main summary
```

Dashboard opcional:

```bash
streamlit run app/dashboard/streamlit_app.py
```

Aunque el bot gane en backtest o paper trading, eso no garantiza ganancias en cuenta real. La ejecucion real puede tener latencia, diferencias de precio, restricciones del broker y riesgo de perdida total. El modo `real` sigue bloqueado por defecto y esta fase no implementa operacion con dinero real.

## Real-Time Data Feed / Demo Data Connection

La fuente de velas se selecciona con `DATA_FEED_SOURCE` en `.env`. Todas las fuentes se normalizan al formato interno:

```text
timestamp,open,high,low,close,volume,asset,timeframe_seconds,source
```

Fuentes disponibles:

- `csv`: lee `CANDLES_CSV_PATH` y entrega velas desde archivo.
- `mock_realtime`: usa un CSV historico, pero entrega las velas una a una como si fueran tiempo real.
- `iqoption_demo`: adaptador preparado para una API demo autorizada; todavia no descarga velas.
- `exnova_demo`: adaptador preparado para una API demo autorizada; todavia no descarga velas.

Ejemplo de `.env`:

```env
BOT_MODE=paper
DATA_FEED_SOURCE=mock_realtime
CANDLES_CSV_PATH=data/raw/candles.csv
COLLECTED_CANDLES_PATH=data/raw/collected_candles.csv
ASSET=EURUSD-OTC
TIMEFRAME_SECONDS=60
LIVE_MAX_STEPS=100
ENABLE_REAL_TRADING=false
```

Recolectar velas:

```bash
python -m app.main collect-data
```

Validar calidad:

```bash
python -m app.main data-quality
```

Correr live paper con la fuente configurada:

```bash
python -m app.main live-paper
python -m app.main summary
pytest
```

La conexion a brokers reales debe hacerse unicamente mediante APIs, websockets o metodos permitidos por el proveedor. El proyecto no usa automatizacion de pantalla, no automatiza clicks, no hace scraping agresivo, no intenta saltarse restricciones del broker y no activa trading real. Las credenciales futuras deben vivir en `.env` y no se imprimen en logs.

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
    live_engine.py       Motor live controlado para paper/demo
    order_manager.py     Validacion y envio de ordenes paper
    trade_logger.py      Registro CSV de operaciones live
  market/
    candles.py           Carga y validación de CSV
    data_feed.py         Feed CSV vela por vela sin mirar futuro
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
    trades_repository.py Repositorio SQLite para live trading
  dashboard/
    streamlit_app.py     Dashboard opcional de operaciones paper
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
8. `execution/live_engine.py` ejecuta paper trading controlado con logs y persistencia.

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

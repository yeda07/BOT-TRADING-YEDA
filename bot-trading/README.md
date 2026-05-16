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

## Demo Execution

El proyecto distingue tres superficies de ejecucion:

- `paper`: simulacion interna con `PaperBroker`.
- `demo`: ejecucion contra un broker demo autorizado, o `demo_stub` para probar el flujo completo sin proveedor externo.
- `real`: bloqueado por defecto y no usado por esta fase.

`demo_stub` simula una cuenta demo externa y permite probar guardas, ordenes pendientes, resolucion, reconciliacion, SQLite, logs, dashboard y kill switch sin operar dinero real.

Ejemplo de `.env` para demo controlada:

```env
BOT_MODE=demo
BROKER=demo_stub
DATA_FEED_SOURCE=mock_realtime
ENABLE_REAL_TRADING=false
KILL_SWITCH_PATH=data/logs/kill_switch.json
EXECUTION_STATE_PATH=data/logs/execution_state.json
```

Comandos:

```bash
python -m app.main healthcheck
python -m app.main demo
python -m app.main reconcile
python -m app.main summary
python -m app.main kill-status
python -m app.main kill-on "riesgo alto"
python -m app.main kill-off
pytest
```

El kill switch bloquea cualquier nueva orden demo/paper protegida por `ExecutionGuard`:

```bash
python -m app.main kill-on "riesgo alto"
python -m app.main kill-status
python -m app.main kill-off
```

`healthcheck` revisa configuracion, modelo, fuente de datos, base SQLite, kill switch y bloqueo de real trading:

```bash
python -m app.main healthcheck
```

`reconcile` busca ordenes `PENDING` en SQLite y consulta el broker demo para actualizar resultado, profit y balance:

```bash
python -m app.main reconcile
```

IQ Option y Exnova quedan como adaptadores preparados, pero sin ejecucion demo autorizada. Para habilitarlos haria falta una API, websocket o metodo permitido por el proveedor. Este proyecto no usa Selenium, PyAutoGUI, automatizacion visual, capturas de pantalla ni clicks sobre botones `SUBE` o `BAJA`.

La ejecucion demo no garantiza resultados reales. En cuenta real existen latencia, slippage, restricciones del broker, cambios de payout, errores de conexion y riesgo de perdida total. Real trading sigue bloqueado.

## Advanced Validation

La validacion avanzada existe para detectar sobreajuste antes de pensar en cualquier conexion real. Un backtest ganador no es suficiente: el modelo puede haber aprendido ruido, condiciones muy especificas del pasado o una fuga accidental de informacion futura.

Conceptos clave:

- `Walk-forward validation`: entrena en una ventana antigua y evalua en una ventana futura. Luego mueve la ventana hacia adelante sin mezclar datos.
- `Out-of-sample testing`: mide el modelo en datos no usados para entrenar ni seleccionar parametros.
- `Time-series split`: respeta el orden temporal y usa `gap`; no usa `shuffle`.
- `Threshold optimization`: busca el umbral de confianza que equilibra calidad de senal y cantidad minima de operaciones.
- `Monte Carlo`: reordena operaciones para estimar riesgo, drawdown y probabilidad de terminar en ganancia. No garantiza rentabilidad.
- `Stress testing`: simula payout menor, slippage, latencia, ruido y fallos de conexion.
- `Stability score`: resume consistencia entre folds, thresholds y escenarios de stress.

Comandos:

```bash
python -m app.main optimize
python -m app.main validate
python -m app.main walk-forward
python -m app.main threshold-optimize
python -m app.main monte-carlo
python -m app.main stress-test
python -m app.main leakage-audit
python -m app.main validation-report
pytest
```

Archivos generados:

```text
data/logs/walk_forward_results.csv
data/logs/walk_forward_summary.json
data/logs/hyperparameter_results.csv
data/logs/threshold_optimization.csv
data/logs/best_threshold.json
data/logs/monte_carlo_results.csv
data/logs/monte_carlo_summary.json
data/logs/stress_test_results.csv
data/logs/stress_test_summary.json
data/logs/overfitting_report.json
data/logs/model_stability_report.json
data/logs/data_leakage_audit.json
data/logs/final_validation_report.json
data/logs/final_validation_report.md
models/optimized_model.joblib
```

Recomendaciones posibles:

- `REJECTED`: leakage critico, sobreajuste alto o estabilidad muy mala.
- `PAPER_ONLY`: puede seguirse observando, pero no cumple criterios de demo.
- `DEMO_ALLOWED`: cumple criterios estadisticos minimos para demo controlada, nunca real.
- `NEEDS_MORE_DATA`: faltan folds, operaciones o muestra suficiente.

`DEMO_ALLOWED` solo puede aparecer si no hay leakage critico, los folds rentables son suficientes, el profit factor promedio supera `1`, el win rate promedio supera breakeven, el drawdown esta controlado, el riesgo de overfitting no es `HIGH`, Monte Carlo tiene probabilidad de ganancia suficiente y stress moderado no falla.

Un resultado positivo en validacion avanzada no garantiza ganancias reales. Solo reduce el riesgo de sobreajuste y ayuda a decidir si el sistema merece continuar en paper/demo.

## MLOps and Continuous Demo Operation

La fase MLOps permite correr sesiones largas en paper/demo con estado persistente, monitoreo, reportes diarios y ciclo controlado de modelos. No activa trading real y no promueve modelos automaticamente.

Una sesion paper/demo guarda:

- `data/logs/current_session.json`
- `data/logs/runtime_state.json`
- heartbeat de sesion
- ultimo error
- ultimo balance
- ultimo resultado de operacion

Comandos principales:

```bash
python -m app.main run-paper-session
python -m app.main run-demo-session
python -m app.main runtime-status
python -m app.main drift-check
python -m app.main retrain
python -m app.main models
python -m app.main promote MODEL_ID
python -m app.main rollback MODEL_ID
python -m app.main daily-report
pytest
```

`runtime-status` muestra sesion actual, kill switch, estado runtime, metricas del bot y metricas basicas del sistema.

`drift-check` revisa degradacion del modelo: caida de win rate, profit factor, confianza promedio, exceso de HOLD y perdida de edge.

`retrain` usa `data/raw/collected_candles.csv`, valida calidad, entrena un candidato y lo registra en `models/model_registry.json`. El candidato queda como `CANDIDATE`; no reemplaza `models/best_model.joblib`.

`models` lista el registro de modelos. Los estados son:

- `CANDIDATE`
- `STAGING`
- `PRODUCTION`
- `REJECTED`
- `ARCHIVED`

`promote MODEL_ID` solo promueve si el candidato cumple reglas de validacion, estabilidad, drawdown, profit factor, win rate, Monte Carlo y ausencia de leakage critico. Antes de reemplazar el modelo productivo se genera backup.

`rollback MODEL_ID` restaura un modelo registrado anterior hacia `models/best_model.joblib`.

`daily-report` genera:

```text
data/reports/daily_report_YYYYMMDD.json
data/reports/daily_report_YYYYMMDD.md
```

Docker:

```bash
docker compose up --build
```

Servicios:

- `bot`: ejecuta `python -m app.main healthcheck` por defecto.
- `dashboard`: ejecuta Streamlit en `http://localhost:8501`.

Los volumenes montan `./data` y `./models`. Docker no inicia trading real automaticamente.

Un sistema estable en demo no garantiza ganancias reales. Antes de considerar cualquier entorno real se requieren semanas de ejecucion controlada, revision humana, cumplimiento de reglas del broker y analisis de riesgo.

## Troubleshooting run-paper-session

Comando recomendado:

```bash
python -m app.main kill-off
python -m app.main register-current-model
python -m app.main models
python -m app.main healthcheck
python -m app.main run-paper-session
python -m app.main runtime-status
python -m app.main summary
```

Problemas comunes:

- `Kill switch ON`: desactivalo solo si entiendes la causa.

```bash
python -m app.main kill-status
python -m app.main kill-off
```

- `Mode: backtest` en una sesion paper: revisa `.env`. `run-paper-session` fuerza `BOT_MODE=paper` internamente, pero `healthcheck` avisara si `.env` sigue en `backtest` o tiene claves duplicadas como `BOT_MODE` repetido.

```env
BOT_MODE=paper
BROKER=paper
DATA_FEED_SOURCE=mock_realtime
```

- `Not enough candles for live decision`: el motor hace warm-up antes de generar señales. Por defecto requiere:

```env
MIN_CANDLES=200
FEATURE_WINDOW_SIZE=300
LIVE_MAX_STEPS=400
```

Si `LIVE_MAX_STEPS < FEATURE_WINDOW_SIZE`, la sesion terminara antes de poder operar. Sube `LIVE_MAX_STEPS` o usa un CSV con mas velas.

- `total_trades = 0`: puede ser warm-up incompleto, todas las señales `HOLD`, confianza baja, risk manager bloqueando, kill switch activo o modelo no registrado/aprobado.

- `No models registered`: registra el modelo actual sin reemplazar nada:

```bash
python -m app.main register-current-model
python -m app.main models
```

- `NEEDS_MORE_DATA` en `retrain`: es aceptable cuando `data/raw/collected_candles.csv` no tiene suficientes velas nuevas. Recolecta mas datos paper/demo antes de reentrenar.

```bash
python -m app.main collect-data
python -m app.main retrain
```

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
## Session-based metrics

Paper/demo execution now stores a `session_id` in both `data/logs/live_trades.csv` and `data/logs/trades.db`. This prevents mixing a visible session balance with historical profit from older sessions.

Use:

```bash
python -m app.main summary
python -m app.main summary-current
python -m app.main sessions
python -m app.main summary-session SESSION_ID
```

`summary` reports all historical executed trades. `summary-current` reports the active session, or the latest session if the bot is stopped. `summary-session SESSION_ID` reports only that session. Win rate and net profit count only resolved trades with `WON` or `LOST`; `HOLD`, `SKIPPED`, `BLOCKED` and `ERROR` are not counted as executed trades.

If you want to start a clean paper test, archive logs explicitly:

```bash
python -m app.main reset-paper-logs --confirm
```

This archives `live_trades.csv` and `trades.db` with timestamped `.bak` names, then creates fresh files. It never deletes logs automatically.

## Feed cursor and replay protection

`run-paper-session` with `DATA_FEED_SOURCE=mock_realtime` now persists progress in `data/logs/feed_cursor.json`. A new paper session continues from the last consumed candle instead of replaying the same CSV window.

Useful commands:

```bash
python -m app.main feed-status
python -m app.main append-candles path/to/new_candles.csv
python -m app.main reset-feed-cursor --confirm
python -m app.main sessions
```

Relevant `.env` options:

```env
FEED_CURSOR_PATH=data/logs/feed_cursor.json
RESET_FEED_CURSOR=false
RANDOM_FEED_START=false
FEED_START_INDEX=
ALLOW_REPLAY_SAME_WINDOW=false
```

Keep `ALLOW_REPLAY_SAME_WINDOW=false` for normal paper testing. Enable replay only when intentionally debugging the same historical slice; otherwise historical summaries will accumulate duplicated, non-independent trades.

When `feed-status` reports `END_OF_FEED`, add more historical candles with `append-candles` or intentionally reset the cursor with `reset-feed-cursor --confirm`. A second paper session will not start if there are no remaining candles.

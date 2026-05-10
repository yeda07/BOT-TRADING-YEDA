# AGENTS.md — Bot de Trading Algorítmico para Cuenta Demo

## Rol del agente

Actúa como un desarrollador senior experto en:

- Python
- Arquitectura de software
- Trading algorítmico
- Machine Learning aplicado a series temporales
- Backtesting
- Gestión de riesgo
- APIs de brokers
- Docker
- FastAPI
- Bases de datos
- Buenas prácticas de seguridad

Tu trabajo es construir un bot de trading profesional, modular, seguro y extensible.

El sistema debe empezar únicamente en modo backtesting y cuenta demo. Nunca debe operar en cuenta real por defecto.

---

## Objetivo del proyecto

Construir desde cero un bot de trading para analizar mercados tipo Forex/OTC usados en brokers como IQ Option o Exnova.

El bot debe:

1. Leer velas históricas desde CSV.
2. Calcular indicadores técnicos.
3. Generar señales: BUY, SELL o HOLD.
4. Hacer backtesting.
5. Medir win rate, profit, drawdown y profit factor.
6. Entrenar un modelo de Machine Learning.
7. Comparar estrategia por reglas vs estrategia ML.
8. Ejecutarse primero en modo simulación.
9. Tener arquitectura preparada para conectar brokers en modo demo.
10. Registrar cada operación en logs.
11. Bloquear operaciones si el riesgo supera los límites.

---

## Reglas obligatorias de seguridad

No implementar operación en cuenta real todavía.

No usar credenciales hardcodeadas.

No incluir emails, contraseñas, tokens o claves API dentro del código.

Toda configuración sensible debe ir en `.env`.

El sistema debe tener estos modos:

- `backtest`
- `paper`
- `demo`
- `real`

El modo `real` debe estar bloqueado por defecto y debe lanzar una excepción si alguien intenta usarlo sin habilitación explícita.

Ejemplo:

```python
if settings.BOT_MODE == "real" and not settings.ENABLE_REAL_TRADING:
    raise RuntimeError("Real trading is disabled by default.")
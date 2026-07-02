# apple-health-explorer

Analisis completo de tu exportacion de Apple Health: pasos, sueno, frecuencia cardiaca, HRV, actividad, ruido ambiental y correlaciones cruzadas entre metricas.

Complemento de [redmoon](https://github.com/aroaxinping/redmoon) (ciclo menstrual vs sueno). Este paquete cubre todo lo demas.

```bash
pip install apple-health-explorer
```

## Uso

```python
from apple_health_explorer import HealthExport

health = HealthExport("export.xml")
health.steps_analysis()
health.sleep_analysis()
health.heart_analysis()
health.activity_analysis()
health.noise_analysis()
health.correlation_matrix()
```

## Visualizaciones

```python
from apple_health_explorer import HealthExport

health = HealthExport("export.xml")

# Genera graficas con tema oscuro (#0f0f0f fondo, #e85d04 acento)
health.plot_steps()
health.plot_sleep()
health.plot_heart()
health.plot_correlation_matrix()
```

## Que datos analiza

| Metrica | HK Identifier |
|---|---|
| Pasos | StepCount |
| Sueno | SleepAnalysis |
| Frecuencia cardiaca en reposo | RestingHeartRate |
| HRV (SDNN) | HeartRateVariabilitySDNN |
| Energia activa | ActiveEnergyBurned |
| Minutos de ejercicio | AppleExerciseTime |
| Horas de pie | AppleStandHour |
| Ruido ambiental | EnvironmentalAudioExposure |

## Exportar datos de Apple Health

En tu iPhone: Salud -> foto de perfil -> Exportar datos de salud -> descomprime el zip -> `export.xml`.

## Desarrollo

```bash
git clone https://github.com/aroaxinping/apple-health-explorer.git
cd apple-health-explorer
pip install -e ".[all]"
pytest tests/ -v
```

## Licencia

MIT

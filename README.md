# Estadística Core · FastAPI

**Tamara Tapia Leturné** — Statistical Insight · Agile Execution

---

## Estructura del proyecto

```
estadistica_fastapi/
├── main.py                  ← FastAPI app principal
├── routers/
│   ├── upload.py            ← POST /api/upload · GET /api/profile · POST /api/clean
│   ├── descriptiva.py       ← GET /api/descriptiva/{col} · GET /api/descriptiva-cat/{col}
│   ├── inferencia.py        ← POST /api/ttest-1 · /ttest-2 · /ztest-prop · /paired · /ic-media
│   └── visualizaciones.py   ← GET /api/viz/histograma|boxplot|scatter|barras|grupos
├── static/
│   └── index.html           ← Frontend completo (HTML + CSS + JS)
├── requirements.txt
├── Procfile                 ← Para Railway / Render
└── README.md
```

---

## Correr localmente


# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar servidor
uvicorn main:app --reload --port 8000

# 4. Abrir en el navegador
# http://localhost:8000
```

La documentación automática de la API está en:
- `http://localhost:8000/docs`  (Swagger UI)
- `http://localhost:8000/redoc` (ReDoc)

---

## Deployment en Railway (gratis)

1. Crear cuenta en [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Subir este proyecto a un repo de GitHub
4. Railway detecta el `Procfile` automáticamente
5. Variables de entorno: ninguna necesaria
6. Deploy → URL pública lista en ~2 minutos

## Deployment en Render (gratis)

1. Crear cuenta en [render.com](https://render.com)
2. New → Web Service → conectar repo de GitHub
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Plan: Free → Create Web Service

---

## Endpoints disponibles

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/upload` | Carga CSV/Excel, devuelve preview + perfil |
| GET  | `/api/profile` | Perfil del dataset en sesión |
| GET  | `/api/diagnostics/{col}` | Diagnóstico de calidad de una variable |
| POST | `/api/clean` | Aplica limpieza a una columna |
| GET  | `/api/export/csv` | Descarga dataset limpio como CSV |
| GET  | `/api/columns` | Lista columnas con tipo inferido |
| GET  | `/api/descriptiva/{col}` | Estadísticos descriptivos numéricos |
| GET  | `/api/descriptiva-cat/{col}` | Frecuencias de variable categórica |
| POST | `/api/ic-media` | Intervalo de confianza para la media (t) |
| POST | `/api/ttest-1` | t-test una muestra |
| POST | `/api/ttest-2` | t-test dos muestras independientes |
| POST | `/api/ztest-prop` | z-test para proporción |
| POST | `/api/paired` | Prueba pareada (t o Wilcoxon automático) |
| GET  | `/api/viz/histograma` | Histograma → base64 PNG |
| GET  | `/api/viz/boxplot` | Boxplot → base64 PNG |
| GET  | `/api/viz/scatter` | Dispersión + tendencia → base64 PNG |
| GET  | `/api/viz/barras` | Barras de frecuencia → base64 PNG |
| GET  | `/api/viz/grupos` | Boxplot por grupo → base64 PNG |

---

## Nota sobre sesión

La sesión actual es **en memoria por proceso** — funciona perfectamente para uso individual o demostraciones. Para producción multi-usuario, reemplaza el dict `_session` en `routers/upload.py` por Redis o una base de datos por usuario (UUID en cookie).

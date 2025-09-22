## Laboratorio 1 — CBDE Vector Databases: Guía de Ejecución paso a paso
* Bilal Moussaoui El Azrak
* Anouar El Haddadi El Yahyati

Los scripts están organizados en tres grupos:
- `P0.py`, `P1.py`, `P2.py`, `P2_posgres.sql` → PostgreSQL
- `C0.py`, `C1.py`, `C2.py` → ChromaDB
- `G0.py`, `G1.py`, `G2.py` → PostgreSQL + pgvector

Los resultados de ejemplo se incluyen en:
- `Resultados_PostgreSQL.txt`
- `Resultados_Chroma.txt`
- `Resultados_PgVector_Sin_Indices.txt`, `Resultados_PgVector_Con_Indices_IVFFlat.txt`, `Resultados_PgVector_Con_Indices_HNSW.txt`


Dependencias Python principales (se instalan más abajo):
- `psycopg2-binary`, `datasets`, `sentence-transformers`, `tqdm`, `chromadb`, `numpy`

Notas:
- Los scripts descargan 10.000 frases del dataset `PatrickHaller/wiki-and-book-corpus-10M` de Hugging Face en el primer uso (requiere Internet).
- `sentence-transformers` descargará el modelo `all-MiniLM-L6-v2` en el primer uso.

### Preparación del entorno (PowerShell)
Ubíquese en la carpeta del laboratorio y active el entorno virtual incluido (o cree uno nuevo si lo prefiere):

```powershell
cd ".\Lab1_CBDE"

# Opción A: usar venv incluido
./venv/Scripts/Activate.ps1

# Opción B: crear venv nuevo (si lo prefieres)
# python -m venv .venv
# ./.venv/Scripts/Activate.ps1

python -m pip install --upgrade pip
pip install psycopg2-binary datasets sentence-transformers tqdm chromadb numpy
```


### Configuración de PostgreSQL
1) Asegúrese de que el servicio PostgreSQL esté iniciado y que exista la base `cbde_database` y el rol `postgres` con contraseña `postgres` (ajuste si usa otros datos para establecer la conexión).

```powershell
# Abrir consola psql (ajusta ruta si es necesario)
psql -U postgres -h localhost -p 5432 -c "CREATE DATABASE cbde_database;"
```

2) (Solo para bloque G) Instale y habilite la extensión `pgvector` en su servidor. En Windows normalmente se instala desde el StackBuilder o paquete del proveedor. Luego, en la base:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Los scripts `G0.py`/`G1.py` ejecutan `CREATE EXTENSION IF NOT EXISTS vector;` automáticamente, pero la extensión debe estar disponible en el servidor.

### Parámetros de conexión y tamaño de datos
- Todos los scripts usan por defecto: `host=localhost`, `database=cbde_database`, `user=postgres`, `password=postgres`, `port=5432`.
  - Si necesita cambiarlos, edite las constantes de conexión al inicio de `P0.py`, `P1.py`, `P2.py`, `G0.py`, `G1.py`, `G2.py`.
- Número de frases del dataset: por defecto `train[:10000]`. Puede ajustar ese tramo en `P0.py`, `C0.py`, `G0.py` y `C1.py` si desea más/menos datos.
- Tamaño de lote por defecto: 1000.

## Ejecución por bloques

### A) PostgreSQL «clásico» (arrays REAL[] + funciones SQL personalizadas)
Orden recomendado:
1. `P0.py` — Inserta 10.000 `chunks` de texto en `chunks_table`.
2. `P1.py` — Calcula embeddings con `SentenceTransformer` y los guarda en `embeddings_table` (`REAL[]`).
3. `P2_posgres.sql` — Crea funciones de distancia `l1/l2` y wrappers `top2_*`.
4. `P2.py` — Ejecuta 10 consultas de similitud con métricas L1/L2 y muestra resultados/tiempos.

Comandos (PowerShell):
```powershell
python P0.py | tee ../Resultados_PostgreSQL.txt
python P1.py | tee -a ../Resultados_PostgreSQL.txt
psql -U postgres -h localhost -d cbde_database -f P2_posgres.sql | tee -a ../Resultados_PostgreSQL.txt
python P2.py | tee -a ../Resultados_PostgreSQL.txt
```

Salida esperada: métricas por lote y estadísticas de consulta (ver `Resultados_PostgreSQL.txt`).

### B) ChromaDB (persistencia local en `./chroma_persist`)
Orden recomendado:
1. `C0.py` — Crea dos colecciones persistentes (`lab1_chunks_euclidean`, `lab1_chunks_cosine`) e inserta documentos con embeddings «dummy» para medir solo inserción.
2. `C1.py` — Calcula embeddings reales con `SentenceTransformer` y los actualiza en ambas colecciones.
3. `C2.py` — Genera 10 embeddings de consulta y hace `query` top-2 en ambas colecciones, con estadísticas de tiempo.

Comandos (PowerShell):
```powershell
python C0.py | tee ../Resultados_Chroma.txt
python C1.py | tee -a ../Resultados_Chroma.txt
python C2.py | tee -a ../Resultados_Chroma.txt
```

Notas:
- Los datos persisten en la carpeta `chroma_persist/`. Borrarla reinicia el estado de Chroma.

### C) PostgreSQL + pgvector (vectores nativos + índices)
Orden recomendado:
1. `G0.py` — Crea `chunks_table_pgvector` e inserta los 10.000 `chunks`.
2. `G1.py` — Calcula embeddings y los inserta en `embeddings_table_pgvector (vector(384))`. Crea índices IVFFlat por defecto. (Índices HNSW están en el script como líneas comentadas: descoméntalas si quieres medir HNSW.)
3. `G2.py` — Ejecuta 10 consultas top-2 usando operadores de `pgvector` con distancias euclidiana y coseno, mostrando tiempos y vecinos.

Comandos (PowerShell):
```powershell
python G0.py | tee ../Resultados_PgVector_Sin_Indices.txt
python G1.py | tee ../Resultados_PgVector_Con_Indices_IVFFlat.txt
python G2.py | tee -a ../Resultados_PgVector_Con_Indices_IVFFlat.txt
```

Para probar HNSW (opcional):
1) Edite `G1.py` y descomente el bloque marcado como «Índices HNSW». Comente el bloque de IVFFlat si quiere aislar HNSW.
2) Vuelva a ejecutar `G1.py` y después `G2.py`.
3) Guarde la salida como referencia, por ejemplo en `Resultados_PgVector_Con_Indices_HNSW.txt`.

## Resolución de problemas (FAQ breve)
- No conecta a PostgreSQL: verifique servicio activo, puerto 5432, credenciales y que la base `cbde_database` exista.
- Falta extensión `vector`: instale `pgvector` en su servidor, luego ejecute `CREATE EXTENSION vector;` en la base.
- Error instalando `torch`: use el índice CPU indicado arriba o instale una versión compatible con su Python.
- Descargas lentas de datasets/modelos: el primer uso puede tardar; se cachea en disco para ejecuciones posteriores.
- Chroma no encuentra colecciones: ejecute primero `C0.py` para crear la persistencia inicial.

## Dónde mirar los resultados
- La consola imprimirá métricas y vecinos top-2.
- Los archivos `Resultados_*.txt` en esta carpeta muestran ejecuciones de ejemplo comparables.

## Cambios típicos si desea experimentar
- Cambiar tamaño de dataset: modifique `split="train[:N]"` en `P0.py`, `C0.py`, `G0.py`, `C1.py`.
- Cambiar batch size: variable `batch_size`/`BATCH_SIZE` en los scripts.
- Ajustar conexión: constantes al inicio de cada script `P*` y `G*`.

import time, psycopg2

postgres_connection = psycopg2.connect(
    host="localhost",
    database="cbde_database",
    user="postgres",
    password="postgres",
    port="5432"
)
cursor = postgres_connection.cursor()

chunk_ids = [11,12,13,14,15,16,17,18,19,20]

def run_top2(metric_func, qid):
    t0 = time.perf_counter()
    cursor.execute(f"SELECT neighbor_id, distance FROM {metric_func}(%s);", (qid,))
    neighbors = cursor.fetchall()  # [(neighbor_id, distance), (neighbor_id, distance)]
    dt = time.perf_counter() - t0
    return neighbors, dt

results = []
for qid in chunk_ids:
    # obtener el texto del query_id
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (qid,))
    chunk_text = cursor.fetchone()[0]

    eucl, t_eucl = run_top2('top2_euclidean', qid)
    manh, t_manh = run_top2('top2_manhattan', qid)

    results.append({
        "chunk_id": qid,
        "chunk": chunk_text,
        "euclidean": eucl,  "t_euclidean": t_eucl,
        "manhattan": manh,  "t_manhattan": t_manh
    })

for r in results:
    print(f"\nQuery {r['chunk_id']}: {r['chunk']}...")
    
    print("  Euclidean:", r["euclidean"], f"({r['t_euclidean']:.4f}s)")
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["euclidean"][0][0],))
    eucl_first_chunk = cursor.fetchone()[0]
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["euclidean"][1][0],))
    eucl_second_chunk = cursor.fetchone()[0]
    print("    First neighbor:", eucl_first_chunk)
    print("    Second neighbor:", eucl_second_chunk)

    print(" <-------------------------------->")

    print("  Manhattan:", r["manhattan"], f"({r['t_manhattan']:.4f}s)")
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["manhattan"][0][0],))
    manh_first_chunk = cursor.fetchone()[0]
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["manhattan"][1][0],))
    manh_second_chunk = cursor.fetchone()[0]
    print("    First neighbor:", manh_first_chunk)
    print("    Second neighbor:", manh_second_chunk)

cursor.close()
postgres_connection.close()

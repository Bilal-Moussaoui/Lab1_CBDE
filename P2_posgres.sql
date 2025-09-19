-- Distancia Euclídea (L2)
CREATE OR REPLACE FUNCTION l2_distance(a REAL[], b REAL[])
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
  s DOUBLE PRECISION := 0;
  i INT;
  len INT;
  d DOUBLE PRECISION;
BEGIN
  len := array_length(a, 1);
  IF len IS NULL OR len <> array_length(b, 1) THEN
    RAISE EXCEPTION 'El tamaño de los vectores NO es el mismo: % vs %', len, array_length(b,1);
  END IF;
  FOR i IN 1..len LOOP
    d := (a[i] - b[i]);
    s := s + d*d;
  END LOOP;
  RETURN sqrt(s);
END;
$$;

-- Distancia Manhattan (L1)
CREATE OR REPLACE FUNCTION l1_distance(a REAL[], b REAL[])
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
  s DOUBLE PRECISION := 0;
  i INT;
  len INT;
BEGIN
  len := array_length(a, 1);
  IF len IS NULL OR len <> array_length(b, 1) THEN
    RAISE EXCEPTION 'El tamaño de los vectores NO es el mismo: % vs %', len, array_length(b,1);
  END IF;
  FOR i IN 1..len LOOP
    s := s + abs(a[i] - b[i]);
  END LOOP;
  RETURN s;
END;
$$;


-- Top-2 por Euclídea
CREATE OR REPLACE FUNCTION top2_euclidean(query_embedding REAL[], query_id INT)
RETURNS TABLE(neighbor_id INT, distance DOUBLE PRECISION)
LANGUAGE sql
STABLE
AS $$
  SELECT e.id AS neighbor_id,
         l2_distance(query_embedding, e.embedding) AS distance
  FROM embeddings_table e
  WHERE e.id <> query_id
  ORDER BY distance ASC
  LIMIT 2;
$$;

-- Top-2 por Manhattan
CREATE OR REPLACE FUNCTION top2_manhattan(query_embedding REAL[], query_id INT)
RETURNS TABLE(neighbor_id INT, distance DOUBLE PRECISION)
LANGUAGE sql
STABLE
AS $$
  SELECT e.id AS neighbor_id,
         l1_distance(query_embedding, e.embedding) AS distance
  FROM embeddings_table e
  WHERE e.id <> query_id
  ORDER BY distance ASC
  LIMIT 2;
$$;

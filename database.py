import sqlite3

DB_PATH = "./films.db"

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS films (
            id  INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fiche_technique (
            film_id INTEGER NOT NULL REFERENCES films(id),
            cle     TEXT NOT NULL,
            valeur  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS distribution (
            film_id INTEGER NOT NULL REFERENCES films(id),
            entree  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS distinctions (
            film_id INTEGER NOT NULL REFERENCES films(id),
            entree  TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def save_film(conn: sqlite3.Connection, sql_data: dict) -> None:
    """Insère les données d'un film (fiche, distribution, distinctions).
    Idempotent : si l'URL existe déjà, on ne réinsère pas."""
    url = sql_data["url"]
    cur = conn.cursor()

    cur.execute("SELECT id FROM films WHERE url = ?", (url,))
    row = cur.fetchone()
    if row:
        return  # déjà présent

    cur.execute("INSERT INTO films (url) VALUES (?)", (url,))
    film_id = cur.lastrowid

    cur.executemany(
        "INSERT INTO fiche_technique (film_id, cle, valeur) VALUES (?, ?, ?)",
        [(film_id, k, v) for k, v in sql_data.get("fiche_technique", {}).items()],
    )
    cur.executemany(
        "INSERT INTO distribution (film_id, entree) VALUES (?, ?)",
        [(film_id, e) for e in sql_data.get("distribution", [])],
    )
    cur.executemany(
        "INSERT INTO distinctions (film_id, entree) VALUES (?, ?)",
        [(film_id, e) for e in sql_data.get("distinctions", [])],
    )
    conn.commit()

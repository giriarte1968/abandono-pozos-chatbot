# rag_llm_explica_chatbot.py
import streamlit as st
import oracledb
from sentence_transformers import SentenceTransformer

# ========================================
# CONFIGURACIÓN
# ========================================
st.set_page_config(page_title="Abandono de Pozos - RAG Chatbot", layout="centered")
st.title("Chatbot RAG + LLM: Abandono de Pozos")
st.markdown("### Pregunta sobre el documento técnico (5 secciones)")

# --- CREDENCIALES ---
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
CONNECT_ALIAS = st.secrets["CONNECT_ALIAS"]

# --- CARPETA WALLET (en GitHub) ---
WALLET_DIR = "Wallet_RAGTEST"

# --- INICIALIZAR CLIENTE ORACLE (Thin + Wallet) ---
try:
    oracledb.init_oracle_client(
        config_dir=WALLET_DIR,
        wallet_password=DB_PASSWORD
    )
    st.success("Oracle Wallet cargado (Thin mode + TCPS)")
except Exception as e:
    st.error(f"Error Wallet: {e}")

# --- CONFIG RAG ---
TABLE_NAME = "faqs"

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

encoder = load_model()

# ========================================
# CONEXIÓN A ORACLE
# ========================================
@st.cache_resource
def get_connection():
    try:
        conn = oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=CONNECT_ALIAS
        )
        st.success("Conectado a Oracle ADB")
        return conn
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

def rag_search(query: str):
    conn = get_connection()
    if not conn:
        return None
    vec = encoder.encode([query])[0].tolist()
    try:
        with conn.cursor() as cursor:
            cursor.setinputsizes(vector=oracledb.DB_TYPE_VECTOR)
            cursor.execute(f"""
                SELECT JSON_VALUE(payload, '$.question') AS q,
                       JSON_VALUE(payload, '$.answer') AS a,
                       JSON_VALUE(payload, '$.source') AS src,
                       VECTOR_DISTANCE(vector, TO_VECTOR(:vector), COSINE) AS dist
                FROM {TABLE_NAME}
                ORDER BY dist
                FETCH FIRST 1 ROW ONLY
            """, {'vector': vec})
            row = cursor.fetchone()
            if row:
                q, a

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
        wallet_location=WALLET_DIR,
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
                q, a, src, dist = row
                return {"pregunta": q, "respuesta": a, "fuente": src, "distancia": round(dist, 4)}
            return None
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

# ========================================
# LLM (cambiar por Groq después)
# ========================================
def generate_explanation(query: str, context: str, source: str, distance: float):
    prompt = f"""
    Eres un experto en abandono de pozos. Responde claro y educativo.

    1. Respuesta directa
    2. Contexto
    3. Importancia
    4. Fuente: {source}

    Pregunta: {query}
    Texto: {context}
    """
    try:
        import ollama
        response = ollama.chat(model="gpt-oss:20b", messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        return f"LLM no disponible (Ollama solo local): {e}"

# ========================================
# CHAT
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fuente"):
            st.caption(f"Fuente: {msg['fuente']} | Dist: {msg['distancia']}")

if prompt := st.chat_input("Ej: ¿Qué es el DTM?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en Oracle..."):
            result = rag_search(prompt)
            if result and result["distancia"] <= 0.7:
                respuesta = generate_explanation(prompt, result["respuesta"], result["fuente"], result["distancia"])
            else:
                respuesta = "No se encontró información relevante."

            st.markdown(respuesta)
            if result and result["distancia"] <= 0.7:
                st.caption(f"Fuente: {result['fuente']} | Dist: {result['distancia']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": respuesta,
                "fuente": result["fuente"] if result else None,
                "distancia": result["distancia"] if result else None
            })

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.header("Config")
    st.write("**Doc:** Abandono de Pozo")
    st.write("**RAG:** MiniLM-L6-v2")
    st.write("**LLM:** gpt-oss:20b")
    st.write("**Umbral:** ≤ 0.7")
    st.divider()
    st.markdown("### Preguntas:")
    for q in ["¿Qué es el DTM?", "¿Cuántas cementadoras?", "¿Cuántos operarios?", "¿De dónde viene el cemento?", "¿Qué equipos se usan?"]:
        st.code(q)

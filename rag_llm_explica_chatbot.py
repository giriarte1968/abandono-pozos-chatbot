# rag_llm_explica_chatbot.py
import streamlit as st
import oracledb
from sentence_transformers import SentenceTransformer
import ollama

# ========================================
# ACTIVAR THICK MODE (SOLUCIONA DPY-4011)
# ========================================
try:
    oracledb.init_oracle_client()  # Usa Oracle Instant Client (instalado por packages.txt)
    st.success("Oracle Thick Mode activado → Soporte para NNE en ADB")
except Exception as e:
    st.error(f"Error al activar Thick Mode: {e}")

# ========================================
# CONFIGURACIÓN
# ========================================
st.set_page_config(page_title="Abandono de Pozos - RAG Chatbot", layout="centered")
st.title("Chatbot RAG + LLM: Abandono de Pozos")
st.markdown("### Pregunta sobre el documento técnico (5 secciones)")

DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DSN = st.secrets["DSN"]

MODEL_NAME = "gpt-oss:20b"
TABLE_NAME = "faqs"

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

encoder = load_model()

# ========================================
# CONEXIÓN A ORACLE (Thick Mode + NNE)
# ========================================
@st.cache_resource
def get_connection():
    try:
        conn = oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DSN
        )
        st.success("Conectado a Oracle ADB (NNE activado)")
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
                SELECT
                    JSON_VALUE(payload, '$.question') AS q,
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
        st.error(f"Error Oracle: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

# ========================================
# GENERAR RESPUESTA
# ========================================
def generate_explanation(query: str, context: str, source: str, distance: float):
    prompt = f"""
    Eres un asistente técnico especializado en abandono de pozos petroleros.
    Explica de forma clara, profesional y educativa.

    1. Respuesta directa
    2. Contexto breve
    3. Importancia operativa
    4. Fuente

    Texto: {context}
    Pregunta: {query}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error LLM: {e}"

# ========================================
# CHAT
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.success(f"**LLM:** `{MODEL_NAME}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("fuente"):
            st.caption(f"**Fuente:** {message['fuente']} | Dist: {message['distancia']}")

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
                st.caption(f"**Fuente:** {result['fuente']} | Dist: {result['distancia']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": respuesta,
                "fuente": result["fuente"] if result and result["distancia"] <= 0.7 else None,
                "distancia": result["distancia"] if result and result["distancia"] <= 0.7 else None
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

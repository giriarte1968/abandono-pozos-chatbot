# rag_chatbot.py
import streamlit as st
import oracledb
import os
from sentence_transformers import SentenceTransformer
import ollama

# ========================================
# CONFIGURACIÓN
# ========================================
st.set_page_config(page_title="Abandono de Pozos - RAG Chatbot", layout="centered")
st.title("Chatbot RAG + LLM: Abandono de Pozos")
st.markdown("### Pregunta sobre el documento técnico (5 secciones)")

# --- Configuración de conexión ---
DB_USER = st.secrets.get("DB_USER", "vector")
DB_PASSWORD = st.secrets.get("DB_PASSWORD", "AXPHAXPHAXPH777a.")
CONNECT_ALIAS = st.secrets.get("CONNECT_ALIAS", "ragtest_high")
WALLET_PATH = st.secrets.get("WALLET_PATH", r"C:\Users\Gustavo\pruebaonedrive\Wallet_RAGTEST")
MODEL_NAME = "gpt-oss:20b"  # ← MODELO EXACTO QUE TIENES DESCARGADO
TABLE_NAME = "faqs"
os.environ['TNS_ADMIN'] = WALLET_PATH

# --- Inicializar modelo (CPU) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
encoder = load_model()

# --- Inicializar Oracle Client ---
try:
    oracledb.init_oracle_client()
except:
    pass

# ========================================
# FUNCIÓN: BUSCAR EN ORACLE
# ========================================
def rag_search(query: str):
    vec = encoder.encode([query])[0].tolist()
    try:
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=CONNECT_ALIAS,
            config_dir=WALLET_PATH,
            wallet_location=WALLET_PATH,
            wallet_password=DB_PASSWORD
        )
        with connection.cursor() as cursor:
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
        if 'connection' in locals():
            connection.close()

# ========================================
# FUNCIÓN: EXTRAER DATO CON LLM (gpt-oss:20b)
# ========================================
def extract_with_llm(query: str, context: str):
    prompt = f"""
    Responde SOLO con el dato exacto (número, nombre, sí/no).
    No expliques. No inventes. No uses comillas.

    Texto del documento:
    {context}

    Pregunta: {query}
    Respuesta:
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error LLM: {e}"

# ========================================
# INTERFAZ DE CHAT
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar modelo usado
st.sidebar.success(f"**LLM en uso:** `{MODEL_NAME}`")

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("fuente"):
            st.caption(f"**Fuente:** {message['fuente']} | Distancia COSINE: {message['distancia']}")

# Input del usuario
if prompt := st.chat_input("Ej: ¿Cuántas cementadoras hay?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner(f"RAG + {MODEL_NAME}..."):
            result = rag_search(prompt)

            if result and result["distancia"] <= 0.7:
                llm_answer = extract_with_llm(prompt, result["respuesta"])
                respuesta = llm_answer
            else:
                respuesta = "No se encontró información relevante en el documento."

            st.markdown(respuesta)
            if result and result["distancia"] <= 0.7:
                st.caption(f"**Fuente:** {result['fuente']} | Distancia COSINE: {result['distancia']}")

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
    st.header("Configuración")
    st.write("**Documento:** Proceso de Abandono de Pozo (Versión Consolidada)")
    st.write("**Modelo RAG:** `all-MiniLM-L6-v2` (CPU)")
    st.write("**LLM:** `gpt-oss:20b` (local, 20B parámetros)")
    st.write("**Umbral:** ≤ 0.7 (relevante)")
    st.divider()
    st.markdown("### Preguntas válidas:")
    st.code("¿Cuántas cementadoras hay?")
    st.code("¿Qué es el DTM?")
    st.code("¿Cuántos operarios directos?")
    st.code("¿De dónde viene el cemento?")
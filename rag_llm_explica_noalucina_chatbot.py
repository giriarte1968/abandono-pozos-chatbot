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
MODEL_NAME = "gpt-oss:20b"
TABLE_NAME = "faqs"
os.environ['TNS_ADMIN'] = WALLET_PATH

# --- Inicializar modelo ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
encoder = load_model()

# --- Inicializar modo thick ---
try:
    oracledb.init_oracle_client()
except:
    pass  # Ya está en PATH

# ========================================
# FUNCIÓN: BUSCAR EN ORACLE (RAG)
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
        st.error(f"Error: {e}")
        return None
    finally:
        if 'connection' in locals():
            connection.close()

# ========================================
# FUNCIÓN: GENERAR RESPUESTA EXPLICATIVA (CON ANTI-ALUCINACIÓN)
# ========================================
def generate_explanation(query: str, context: str, source: str, distance: float):
    prompt = f"""
    ERES UN ASISTENTE TÉCNICO QUE SOLO USA EL TEXTO DEL DOCUMENTO PDF.
    NO TIENES ACCESO A INTERNET, FECHAS ACTUALES, NOTICIAS NI CONOCIMIENTO EXTERNO.

    REGLAS ESTRICTAS:
    - SI LA RESPUESTA NO ESTÁ EN EL TEXTO → DILO CLARAMENTE.
    - NUNCA inventes fechas, nombres, números, costos o hechos.
    - NUNCA uses información de internet o de tu entrenamiento.
    - SI NO SABES → responde: "No hay información sobre esto en el documento técnico."

    Explica de forma clara, profesional y educativa para un usuario administrativo que está aprendiendo el tema.
    Usa lenguaje sencillo, evita tecnicismos sin explicación, y estructura la respuesta así:
    1. Respuesta directa al usuario
    2. Explicación breve del contexto
    3. Importancia operativa
    4. Fuente del documento

    Texto del documento:
    {context}

    Pregunta del usuario: {query}

    Respuesta (sigue el formato exacto):
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}  # ← 0.0 = DETERMINISTA (NO ALUCINA)
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error al generar explicación: {e}"

# ========================================
# INTERFAZ DE CHAT
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.success(f"**LLM en uso:** `{MODEL_NAME}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("fuente"):
            st.caption(f"**Fuente:** {message['fuente']} | Distancia COSINE: {message['distancia']}")

if prompt := st.chat_input("Ej: ¿Qué es el DTM?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Generando explicación técnica (anti-alucinación)..."):
            result = rag_search(prompt)

            if result and result["distancia"] <= 0.7:
                explanation = generate_explanation(
                    query=prompt,
                    context=result["respuesta"],
                    source=result["fuente"],
                    distance=result["distancia"]
                )
                respuesta = explanation
            else:
                respuesta = "No se encontró información relevante en el documento técnico."

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
    st.write("**Umbral:** ≤ 0.7 (ajustado para precisión)")
    st.write("**Anti-alucinación:** Activada (solo PDF)")
    st.divider()
    st.markdown("### Preguntas válidas:")
    st.code("¿Qué es el DTM?")
    st.code("¿Cuántas cementadoras hay?")
    st.code("¿Cuántos operarios se necesitan?")
    st.code("¿De dónde viene el cemento?")
    st.code("¿Qué equipos se usan?")
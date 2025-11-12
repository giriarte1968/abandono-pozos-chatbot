# rag_llm_explica_chatbot.py
import streamlit as st
import oracledb
from sentence_transformers import SentenceTransformer
import ollama

# ========================================
# CONFIGURACIÓN
# ========================================
st.set_page_config(page_title="Abandono de Pozos - RAG Chatbot", layout="centered")
st.title("Chatbot RAG + LLM: Abandono de Pozos")
st.markdown("### Pregunta sobre el documento técnico (5 secciones)")

# --- CREDENCIALES ORACLE (desde Secrets) ---
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DSN = st.secrets["DSN"]  # TU DSN: adb.eu-frankfurt-1.oraclecloud.com:1522/...

# --- CONFIGURACIÓN RAG ---
MODEL_NAME = "gpt-oss:20b"
TABLE_NAME = "faqs"

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

encoder = load_model()

# ========================================
# CONEXIÓN A ORACLE (DSN directo, sin Wallet)
# ========================================
@st.cache_resource
def get_connection():
    try:
        conn = oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DSN
        )
        st.success("Conectado a Oracle (ADB) correctamente")
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
# GENERAR RESPUESTA EXPLICATIVA
# ========================================
def generate_explanation(query: str, context: str, source: str, distance: float):
    prompt = f"""
    Eres un asistente técnico especializado en abandono de pozos petroleros.
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
            messages=[{"role": "user", "content": prompt}]
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
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Generando explicación técnica..."):
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
    st.write("**Modo:** Explicativo educativo")
    st.divider()
    st.markdown("### Preguntas válidas:")
    st.code("¿Qué es el DTM?")
    st.code("¿Cuántas cementadoras hay?")
    st.code("¿Cuántos operarios se necesitan?")
    st.code("¿De dónde viene el cemento?")
    st.code("¿Qué equipos se usan?")

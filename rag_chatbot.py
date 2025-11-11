# rag_chatbot.py
import streamlit as st
import oracledb
import os
import json
from sentence_transformers import SentenceTransformer

# ========================================
# CONFIGURACIÓN
# ========================================
st.set_page_config(page_title="Abandono de Pozos - RAG Chatbot", layout="centered")
st.title("Chatbot RAG: Abandono de Pozos")
st.markdown("### Pregunta sobre el documento técnico (5 secciones)")

# --- Configuración de conexión ---
DB_USER       = st.secrets.get("DB_USER", "vector")
DB_PASSWORD   = st.secrets.get("DB_PASSWORD", "AXPHAXPHAXPH777a.")
CONNECT_ALIAS = st.secrets.get("CONNECT_ALIAS", "ragtest_high")
WALLET_PATH   = st.secrets.get("WALLET_PATH", r"C:\Users\Gustavo\pruebaonedrive\Wallet_RAGTEST")
MODEL_NAME = "qwen:3.8b"  # ← TU MODELO
TABLE_NAME    = "faqs"

os.environ['TNS_ADMIN'] = WALLET_PATH

# --- Inicializar modelo ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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
            config_dir=WALLET_PATH,        # ← NECESARIO
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
        	    VECTOR_DISTANCE(vector, TO_VECTOR(:vector), COSINE) AS dist  -- ← AÑADIDO
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
# INTERFAZ DE CHAT
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("fuente"):
            st.caption(f"**Fuente:** {message['fuente']} | Distancia: {message['distancia']}")

# Input del usuario
if prompt := st.chat_input("Ej: ¿Qué es el DTM?"):
    # Mostrar pregunta
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Buscar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el documento..."):
            result = rag_search(prompt)
            
        if result:
            respuesta = result["respuesta"]
            st.markdown(respuesta)
            st.caption(f"**Fuente:** {result['fuente']} | Distancia COSINE: {result['distancia']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": respuesta,
                "fuente": result["fuente"],
                "distancia": result["distancia"]
            })
        else:
            st.markdown("No se encontró información relevante en el documento.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "No se encontró información relevante."
            })

# ========================================
# SIDEBAR: INFO
# ========================================
with st.sidebar:
    st.header("Información")
    st.write("**Documento cargado:** 5 secciones")
    st.write("**Modelo:** `all-MiniLM-L6-v2`")
    st.write("**Base de datos:** Oracle 23ai ADB")
    st.write("**Tabla:** `faqs`")
    st.write("**Métrica:** Distancia COSINE")
    st.divider()
    st.markdown("### Preguntas de ejemplo:")
    st.code("¿Qué es el DTM?")
    st.code("¿Cuántas cementadoras hay?")
    st.code("¿De dónde viene el cemento?")
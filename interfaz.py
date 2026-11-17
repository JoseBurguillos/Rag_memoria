import streamlit as st
import requests

st.title("Asistente de VelozNet")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
    
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["texto"])
        
pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    # 1. Mostrar la pregunta del usuario y guardarla
    with st.chat_message("user"):
        st.write(pregunta)
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})

    # 2. Mandar la pregunta a la API
    respuesta = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"cliente": "user", "pregunta":pregunta}
    )
    texto = respuesta.json()["respuesta"]

    # 3. Mostrar la respuesta del bot y guardarla
    with st.chat_message("assistant"):
        st.write(texto)
    st.session_state.mensajes.append({"rol": "assistant", "texto": texto})
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Test Supabase", page_icon="🔐")

# Conexión (lee secretos desde Streamlit Cloud)
sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])

st.title("🔐 Test conexión con Supabase")
modo = st.sidebar.radio("Menú", ["Registro", "Login"])

if modo == "Registro":
    email = st.text_input("Email")
    password = st.text_input("Contraseña", type="password")
    if st.button("Crear cuenta"):
        try:
            sb.auth.sign_up({"email": email, "password": password})
            st.success("✅ Usuario creado (revisa correo si pide confirmación)")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    email = st.text_input("Email")
    password = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        try:
            res = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["user"] = res.user
            st.success(f"Bienvenido {res.user.email} 🎉")
        except Exception as e:
            st.error(f"Error: {e}")

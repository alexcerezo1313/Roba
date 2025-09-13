import streamlit as st
import pandas as pd

st.set_page_config(page_title="👕 Armario Digital", page_icon="🧥", layout="wide")

# --- Inicializar almacenamiento en sesión ---
if "armario" not in st.session_state:
    st.session_state["armario"] = pd.DataFrame(columns=["Tipo", "Categoria", "Largo", "Capa"])

st.title("👕 Armario Digital")
st.write("Añade tus prendas y organízalas por categorías.")

# --- Formulario para añadir prenda ---
with st.form("nueva_prenda", clear_on_submit=True):
    tipo = st.selectbox("Tipo de prenda", ["Parte de arriba", "Parte de abajo", "Zapatos"])
    
    # Subcategorías
    if tipo == "Parte de arriba":
        categoria = st.selectbox("Categoria", ["Camiseta", "Camisa", "Sudadera"])
        capa = st.selectbox("Capa", ["Primera piel", "Segunda capa", "Tercera capa"])
    elif tipo == "Parte de abajo":
        categoria = st.selectbox("Categoria", ["Pantalón", "Short", "Falda"])
        capa = "-"
    else:  # Zapatos
        categoria = st.selectbox("Categoria", ["Zapatillas", "Botas", "Sandalias"])
        capa = "-"
    
    largo = st.selectbox("Largo", ["Corto", "Largo"])
    enviado = st.form_submit_button("➕ Añadir prenda")

    if enviado:
        nueva = pd.DataFrame([[tipo, categoria, largo, capa]], columns=st.session_state["armario"].columns)
        st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
        st.success(f"{categoria} añadida al armario ✅")

# --- Mostrar el armario ---
st.subheader("🗂 Tu Armario")
if st.session_state["armario"].empty:
    st.info("Aún no has añadido ninguna prenda.")
else:
    st.dataframe(st.session_state["armario"], use_container_width=True)

# --- Filtros para buscar prendas ---
st.subheader("🔍 Buscar por filtros")
col1, col2, col3 = st.columns(3)

with col1:
    filtro_tipo = st.selectbox("Filtrar por tipo", ["Todos"] + st.session_state["armario"]["Tipo"].unique().tolist())
with col2:
    filtro_largo = st.selectbox("Filtrar por largo", ["Todos"] + st.session_state["armario"]["Largo"].unique().tolist())
with col3:
    filtro_capa = st.selectbox("Filtrar por capa", ["Todos"] + [c for c in st.session_state["armario"]["Capa"].unique() if c != "-"])

# Aplicar filtros
df_filtrado = st.session_state["armario"]
if filtro_tipo != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Tipo"] == filtro_tipo]
if filtro_largo != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Largo"] == filtro_largo]
if filtro_capa != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Capa"] == filtro_capa]

st.write("### 👕 Resultado de la búsqueda")
if df_filtrado.empty:
    st.warning("No hay prendas que coincidan con esos filtros.")
else:
    st.dataframe(df_filtrado, use_container_width=True)

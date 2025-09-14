import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
import io
from datetime import datetime

st.set_page_config(page_title="👕 Armario Digital (con XML)", page_icon="🧥", layout="wide")

# -------------------- Estado inicial --------------------
COLUMNS = ["Tipo", "Categoria", "Largo", "Capa"]
if "armario" not in st.session_state:
    st.session_state["armario"] = pd.DataFrame(columns=COLUMNS)

st.title("👕 Armario Digital")
st.caption("Añade prendas y guarda/recupera tu armario con un archivo XML.")

# -------------------- Utilidades XML --------------------
SCHEMA_VERSION = "1.0"

def df_to_xml_bytes(df: pd.DataFrame) -> bytes:
    root = Element("wardrobe", attrib={"version": SCHEMA_VERSION})
    for _, row in df.iterrows():
        item = SubElement(root, "item")
        SubElement(item, "type").text = str(row.get("Tipo", ""))
        SubElement(item, "category").text = str(row.get("Categoria", ""))
        SubElement(item, "length").text = str(row.get("Largo", ""))
        SubElement(item, "layer").text = str(row.get("Capa", ""))
    buf = io.BytesIO()
    ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()

def xml_bytes_to_df(xml_bytes: bytes) -> pd.DataFrame:
    try:
        root = fromstring(xml_bytes)
        records = []
        for item in root.findall("item"):
            records.append({
                "Tipo": (item.findtext("type") or "").strip(),
                "Categoria": (item.findtext("category") or "").strip(),
                "Largo": (item.findtext("length") or "").strip(),
                "Capa": (item.findtext("layer") or "").strip(),
            })
        df = pd.DataFrame(records, columns=COLUMNS)
        # sane defaults
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception as e:
        st.error(f"XML no válido: {e}")
        return pd.DataFrame(columns=COLUMNS)

# -------------------- Formulario: añadir prenda --------------------
with st.form("nueva_prenda", clear_on_submit=True):
    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        tipo = st.selectbox("Tipo", ["Parte de arriba", "Parte de abajo", "Zapatos"])
    with colB:
        if tipo == "Parte de arriba":
            categoria = st.selectbox("Categoría", ["Camiseta", "Camisa", "Sudadera"])
        elif tipo == "Parte de abajo":
            categoria = st.selectbox("Categoría", ["Pantalón", "Short", "Falda"])
        else:
            categoria = st.selectbox("Categoría", ["Zapatillas", "Botas", "Sandalias"])
    with colC:
        largo = st.selectbox("Largo", ["Corto", "Largo"])
    with colD:
        capa = st.selectbox(
            "Capa",
            ["Primera piel", "Segunda capa", "Tercera capa", "-"] if tipo == "Parte de arriba" else ["-"]
        )

    enviar = st.form_submit_button("➕ Añadir prenda")
    if enviar:
        nueva = pd.DataFrame([[tipo, categoria, largo, capa]], columns=COLUMNS)
        st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
        st.success(f"{categoria} añadida ✅")

# -------------------- Exportar / Importar XML --------------------
st.subheader("💾 Guardar / Cargar tu armario (XML)")
col1, col2 = st.columns([1,1])

with col1:
    if st.session_state["armario"].empty:
        st.button("Descargar XML", disabled=True)
        st.info("Añade alguna prenda para poder descargar tu XML.")
    else:
        xml_bytes = df_to_xml_bytes(st.session_state["armario"])
        filename = f"armario_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.xml"
        st.download_button(
            "⬇️ Descargar XML",
            data=xml_bytes,
            file_name=filename,
            mime="application/xml",
            help="Guarda este archivo. Luego podrás cargarlo para recuperar tu armario."
        )

with col2:
    uploaded = st.file_uploader("⬆️ Cargar XML de tu armario", type=["xml"])
    modo = st.radio("Cómo cargar", ["Añadir a lo existente", "Reemplazar todo"], horizontal=True)
    if uploaded is not None:
        df_import = xml_bytes_to_df(uploaded.read())
        if not df_import.empty:
            if modo == "Reemplazar todo":
                st.session_state["armario"] = df_import
            else:
                st.session_state["armario"] = pd.concat([st.session_state["armario"], df_import], ignore_index=True)
            st.success("XML cargado correctamente ✅")

# -------------------- Filtros y vista --------------------
st.subheader("🗂 Tu Armario")
if st.session_state["armario"].empty:
    st.info("Aún no has añadido ninguna prenda.")
else:
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        f_tipo = st.selectbox("Filtrar por tipo", ["Todos"] + sorted(st.session_state["armario"]["Tipo"].unique().tolist()))
    with fcol2:
        f_largo = st.selectbox("Filtrar por largo", ["Todos"] + sorted(st.session_state["armario"]["Largo"].unique().tolist()))
    with fcol3:
        capas = [c for c in st.session_state["armario"]["Capa"].unique().tolist()]
        if "-" not in capas: capas.append("-")
        f_capa = st.selectbox("Filtrar por capa", ["Todos"] + sorted(capas))

    df = st.session_state["armario"].copy()
    if f_tipo != "Todos":
        df = df[df["Tipo"] == f_tipo]
    if f_largo != "Todos":
        df = df[df["Largo"] == f_largo]
    if f_capa != "Todos":
        df = df[df["Capa"] == f_capa]

    st.dataframe(df, use_container_width=True)

# -------------------- Nota --------------------
st.caption("💡 Consejo: guarda tu XML tras añadir prendas. Cuando vuelvas, súbelo para recuperar tu armario. No se guarda en servidor.")

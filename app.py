import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io

st.set_page_config(page_title="👕 Armario Digital", page_icon="🧥", layout="wide")

# ---------- Config ----------
COLUMNS = ["Categoria", "Tipo", "ColorNombre", "ColorHex", "FotoBase64"]
SCHEMA_VERSION = "4.0"

CATEGORIAS = [
    "Camiseta", "Camisa", "Sudadera",   # Arriba
    "Pantalón", "Short", "Falda",       # Abajo
    "Zapatillas", "Botas", "Sandalias"  # Calzado
]
TIPOS = ["Corto", "Largo"]

# Paleta de colores habituales en ropa
OPCIONES_COLOR = {
    "Negro": "#000000",
    "Blanco": "#FFFFFF",
    "Gris": "#808080",
    "Beige": "#F5F5DC",
    "Marrón": "#8B4513",
    "Azul marino": "#000080",
    "Azul claro": "#87CEEB",
    "Rojo": "#FF0000",
    "Verde": "#008000",
    "Amarillo": "#FFFF00",
    "Rosa": "#FFC0CB"
}

# ---------- Estado ----------
if "armario" not in st.session_state:
    st.session_state["armario"] = pd.DataFrame(columns=COLUMNS)

st.title("👕 Armario Digital")
st.caption("Añade prendas, color y (opcional) fotografía. Exporta/Importa tu armario en XML.")

# ---------- Utils ----------
def file_to_b64(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    data = uploaded_file.getvalue()
    return base64.b64encode(data).decode("utf-8")

def b64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64)
    except Exception:
        return b""

def df_to_xml_bytes(df: pd.DataFrame) -> bytes:
    root = Element("wardrobe", attrib={"version": SCHEMA_VERSION})
    for _, row in df.iterrows():
        item = SubElement(root, "item")
        SubElement(item, "category").text = str(row.get("Categoria", ""))
        SubElement(item, "type").text = str(row.get("Tipo", ""))
        SubElement(item, "color_name").text = str(row.get("ColorNombre", ""))
        SubElement(item, "color_hex").text = str(row.get("ColorHex", ""))
        foto_b64 = (row.get("FotoBase64") or "").strip()
        if foto_b64:
            SubElement(item, "photo_b64").text = foto_b64
    buf = io.BytesIO()
    ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()

def xml_bytes_to_df(xml_bytes: bytes) -> pd.DataFrame:
    try:
        root = fromstring(xml_bytes)
        records = []
        for item in root.findall("item"):
            records.append({
                "Categoria": (item.findtext("category") or "").strip(),
                "Tipo": (item.findtext("type") or "").strip(),
                "ColorNombre": (item.findtext("color_name") or "").strip(),
                "ColorHex": (item.findtext("color_hex") or "").strip(),
                "FotoBase64": (item.findtext("photo_b64") or "").strip(),
            })
        df = pd.DataFrame(records, columns=COLUMNS)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception as e:
        st.error(f"XML no válido: {e}")
        return pd.DataFrame(columns=COLUMNS)

# ---------- Formulario: añadir prenda ----------
with st.form("nueva_prenda", clear_on_submit=True):
    c1, c2, c3 = st.columns([1, 1, 1.2])

    with c1:
        categoria = st.selectbox("Categoría", CATEGORIAS)

    with c2:
        tipo = st.selectbox("Tipo (corto/largo)", TIPOS)

    with c3:
        color_nombre = st.selectbox("Color", list(OPCIONES_COLOR.keys()))
        color_hex = OPCIONES_COLOR[color_nombre]
        st.markdown(
            f"<div style='background:{color_hex};width:100%;height:25px;border:1px solid #000;border-radius:5px'></div>",
            unsafe_allow_html=True
        )

    foto = st.file_uploader("Fotografía (opcional)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    enviado = st.form_submit_button("➕ Añadir prenda")
    if enviado:
        nueva = pd.DataFrame([{
            "Categoria": categoria,
            "Tipo": tipo,
            "ColorNombre": color_nombre,
            "ColorHex": color_hex,
            "FotoBase64": file_to_b64(foto)
        }], columns=COLUMNS)
        st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
        st.success(f"{categoria} ({color_nombre}) añadida ✅")

# ---------- Exportar / Importar XML ----------
st.subheader("💾 Guardar / Cargar tu armario (XML)")
cc1, cc2 = st.columns(2)

with cc1:
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

with cc2:
    uploaded = st.file_uploader("⬆️ Cargar XML", type=["xml"])
    modo = st.radio("Cómo cargar", ["Añadir a lo existente", "Reemplazar todo"], horizontal=True)
    if uploaded is not None:
        df_import = xml_bytes_to_df(uploaded.read())
        if not df_import.empty:
            if modo == "Reemplazar todo":
                st.session_state["armario"] = df_import
            else:
                st.session_state["armario"] = pd.concat([st.session_state["armario"], df_import], ignore_index=True)
            st.success("XML cargado correctamente ✅")

# ---------- Vista ----------
st.subheader("🗂 Tu Armario")
if st.session_state["armario"].empty:
    st.info("Aún no has añadido ninguna prenda.")
else:
    f1, f2 = st.columns(2)
    with f1:
        f_cat = st.selectbox("Filtrar por categoría", ["Todos"] + sorted(st.session_state["armario"]["Categoria"].unique().tolist()))
    with f2:
        f_tipo = st.selectbox("Filtrar por tipo", ["Todos"] + sorted(st.session_state["armario"]["Tipo"].unique().tolist()))

    df = st.session_state["armario"].copy()
    if f_cat != "Todos":
        df = df[df["Categoria"] == f_cat]
    if f_tipo != "Todos":
        df = df[df["Tipo"] == f_tipo]

    st.dataframe(df.drop(columns=["FotoBase64"]), use_container_width=True)

    thumbs = df[df["FotoBase64"].str.len() > 0]
    if not thumbs.empty:
        st.write("### 📸 Miniaturas")
        cols = st.columns(6)
        i = 0
        for _, row in thumbs.iterrows():
            img_bytes = b64_to_bytes(row["FotoBase64"])
            if img_bytes:
                with cols[i % 6]:
                    st.image(img_bytes, caption=f"{row['Categoria']} ({row['ColorNombre']})", use_container_width=True)
                i += 1

st.caption("💡 Consejo: descarga tu XML tras añadir prendas, y vuelve a cargarlo cuando regreses a la web.")

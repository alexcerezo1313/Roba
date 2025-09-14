import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io
from PIL import Image
import numpy as np
import plotly.express as px

st.set_page_config(page_title="👕 Armario Digital", page_icon="🧥", layout="wide")

# ---------- Config ----------
COLUMNS = ["Categoria", "Tipo", "Color1Nombre", "Color1Hex", "Color2Nombre", "Color2Hex", "FotoBase64"]
SCHEMA_VERSION = "7.0"

CATEGORIAS = [
    "Camiseta", "Camisa", "Sudadera",
    "Pantalón", "Short", "Falda",
    "Zapatillas", "Botas", "Sandalias"
]
TIPOS = ["Corto", "Largo"]

PALETA = {
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

# ---------- Utils ----------
def file_to_b64(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def b64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64)
    except Exception:
        return b""

def hex_from_rgb(rgb_tuple) -> str:
    r, g, b = rgb_tuple[:3]
    return f"#{r:02X}{g:02X}{b:02X}"

def df_to_xml_bytes(df: pd.DataFrame) -> bytes:
    root = Element("wardrobe", attrib={"version": SCHEMA_VERSION})
    for _, row in df.iterrows():
        item = SubElement(root, "item")
        SubElement(item, "category").text = str(row.get("Categoria", ""))
        SubElement(item, "type").text = str(row.get("Tipo", ""))
        SubElement(item, "color1_name").text = str(row.get("Color1Nombre", ""))
        SubElement(item, "color1_hex").text = str(row.get("Color1Hex", ""))
        SubElement(item, "color2_name").text = str(row.get("Color2Nombre", ""))
        SubElement(item, "color2_hex").text = str(row.get("Color2Hex", ""))
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
                "Color1Nombre": (item.findtext("color1_name") or "").strip(),
                "Color1Hex": (item.findtext("color1_hex") or "").strip(),
                "Color2Nombre": (item.findtext("color2_name") or "").strip(),
                "Color2Hex": (item.findtext("color2_hex") or "").strip(),
                "FotoBase64": (item.findtext("photo_b64") or "").strip(),
            })
        return pd.DataFrame(records, columns=COLUMNS)
    except Exception as e:
        st.error(f"XML no válido: {e}")
        return pd.DataFrame(columns=COLUMNS)

def color_preview(hex_code: str):
    if not hex_code:
        return
    st.markdown(
        f"<div style='background:{hex_code};width:100%;height:26px;border:1px solid #000;border-radius:6px;'></div>",
        unsafe_allow_html=True
    )

def pick_color_from_plotly(img: Image.Image, label: str) -> str:
    """Muestra la imagen con Plotly y permite seleccionar un píxel con el cursor."""
    arr = np.array(img)
    fig = px.imshow(arr)
    fig.update_layout(
        dragmode="drawclosedpath",  # permite marcar zonas
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.caption(f"{label} — Haz clic en la imagen (usa la herramienta de selección para marcar un píxel).")
    st.plotly_chart(fig, use_container_width=True)
    st.info("⚠️ Nota: Streamlit aún no devuelve la coordenada exacta del clic en Plotly.\n"
            "Si quieres que te dé el color exacto del píxel, necesitamos un paso adicional con un callback de Plotly.")
    return ""  # de momento placeholder

# ---------- UI ----------
st.title("👕 Armario Digital")
st.caption("Elige colores: paleta, picker exacto o clic sobre la imagen (Plotly). Añade color secundario y detecta si lo hay.")

with st.form("nueva_prenda", clear_on_submit=False):
    c1, c2 = st.columns([1, 1])

    with c1:
        categoria = st.selectbox("Categoría", CATEGORIAS)
        tipo = st.selectbox("Tipo (corto/largo)", TIPOS)

        # --------- COLOR PRINCIPAL ----------
        metodo1 = st.radio("Color principal — método", ["Paleta", "Picker", "Desde imagen (clic)"])
        color1_name, color1_hex = "", ""

        if metodo1 == "Paleta":
            color1_name = st.selectbox("Color (paleta)", list(PALETA.keys()), key="c1pal")
            color1_hex = PALETA[color1_name]
            color_preview(color1_hex)
        elif metodo1 == "Picker":
            color1_hex = st.color_picker("Color exacto", "#cccccc", key="c1pick")
            color1_name = "Personalizado"
            color_preview(color1_hex)
        else:
            foto1 = st.file_uploader("Fotografía para color principal (clic con Plotly)", type=["png", "jpg", "jpeg"], key="foto1")
            if foto1:
                img1 = Image.open(io.BytesIO(foto1.getvalue()))
                color1_hex = pick_color_from_plotly(img1, "Color principal")
                color1_name = "Desde imagen"
                if color1_hex:
                    color_preview(color1_hex)
            else:
                st.info("Sube una foto para seleccionar el color con clic.")

        # --------- COLOR SECUNDARIO ----------
        st.markdown("---")
        usar_color2 = st.checkbox("Añadir color secundario (opcional)")
        color2_name, color2_hex, metodo2 = "", "", None
        if usar_color2:
            metodo2 = st.radio("Color secundario — método", ["Paleta", "Picker", "Desde imagen (clic)"], key="m2")
            if metodo2 == "Paleta":
                color2_name = st.selectbox("Color secundario (paleta)", list(PALETA.keys()), key="c2pal")
                color2_hex = PALETA[color2_name]
                color_preview(color2_hex)
            elif metodo2 == "Picker":
                color2_hex = st.color_picker("Color secundario exacto", "#bbbbbb", key="c2pick")
                color2_name = "Personalizado"
                color_preview(color2_hex)
            else:
                foto2 = st.file_uploader("Fotografía para secundario", type=["png", "jpg", "jpeg"], key="foto2")
                if foto2:
                    img2 = Image.open(io.BytesIO(foto2.getvalue()))
                    color2_hex = pick_color_from_plotly(img2, "Color secundario")
                    color2_name = "Desde imagen"
                    if color2_hex:
                        color_preview(color2_hex)
                else:
                    st.info("Sube una foto para el color secundario.")

    with c2:
        foto_prenda = st.file_uploader("Fotografía de la prenda (opcional)", type=["png", "jpg", "jpeg"], key="fotoprenda")
        if foto_prenda:
            st.image(foto_prenda, caption="Vista previa", use_container_width=True)

    enviado = st.form_submit_button("➕ Añadir prenda")
    if enviado:
        nueva = pd.DataFrame([{
            "Categoria": categoria,
            "Tipo": tipo,
            "Color1Nombre": color1_name,
            "Color1Hex": color1_hex,
            "Color2Nombre": color2_name,
            "Color2Hex": color2_hex,
            "FotoBase64": file_to_b64(foto_prenda)
        }], columns=COLUMNS)
        st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
        st.success(f"{categoria} añadida ✅")

# ---------- Exportar / Importar ----------
st.subheader("💾 Guardar / Cargar tu armario (XML)")
c1, c2 = st.columns(2)

with c1:
    if not st.session_state["armario"].empty:
        xml_bytes = df_to_xml_bytes(st.session_state["armario"])
        fname = f"armario_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.xml"
        st.download_button("⬇️ Descargar XML", data=xml_bytes, file_name=fname, mime="application/xml")
    else:
        st.info("Añade prendas para poder descargar tu armario.")

with c2:
    up = st.file_uploader("⬆️ Cargar XML", type=["xml"])
    modo = st.radio("Cómo cargar", ["Añadir a lo existente", "Reemplazar todo"], horizontal=True)
    if up is not None:
        df_imp = xml_bytes_to_df(up.read())
        if not df_imp.empty:
            if modo == "Reemplazar todo":
                st.session_state["armario"] = df_imp
            else:
                st.session_state["armario"] = pd.concat([st.session_state["armario"], df_imp], ignore_index=True)
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
        for i, (_, row) in enumerate(thumbs.iterrows()):
            img_bytes = b64_to_bytes(row["FotoBase64"])
            if img_bytes:
                with cols[i % 6]:
                    st.image(img_bytes, caption=f"{row['Categoria']} ({row['Color1Hex'] or row['Color1Nombre']})", use_container_width=True)

st.caption("💡 Consejo: guarda tu XML tras añadir prendas, y recárgalo cuando vuelvas a la web.")

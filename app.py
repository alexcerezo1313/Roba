import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io
from PIL import Image

st.set_page_config(page_title="👕 Armario Digital", page_icon="🧥", layout="wide")

# ---------- Config ----------
COLUMNS = ["Categoria", "Tipo", "ColorNombre", "ColorHex", "FotoBase64"]
SCHEMA_VERSION = "5.1"

CATEGORIAS = [
    "Camiseta", "Camisa", "Sudadera",
    "Pantalón", "Short", "Falda",
    "Zapatillas", "Botas", "Sandalias"
]
TIPOS = ["Corto", "Largo"]

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
st.caption("Elige color: paleta, picker o desde imagen (con sliders). Exporta/Importa tu armario en XML.")

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

def hex_from_rgb(rgb_tuple) -> str:
    r, g, b = rgb_tuple[:3]
    return f"#{r:02X}{g:02X}{b:02X}"

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
        return df
    except Exception as e:
        st.error(f"XML no válido: {e}")
        return pd.DataFrame(columns=COLUMNS)

# ---------- Formulario ----------
with st.form("nueva_prenda", clear_on_submit=False):
    c1, c2 = st.columns([1, 1])

    with c1:
        categoria = st.selectbox("Categoría", CATEGORIAS)
        tipo = st.selectbox("Tipo (corto/largo)", TIPOS)
        metodo_color = st.radio(
            "Cómo elegir el color",
            ["Paleta (rápido)", "Picker (preciso)", "Desde imagen (sliders)"]
        )

    with c2:
        foto = st.file_uploader("Fotografía (opcional, necesaria si eliges 'desde imagen')",
                                type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    color_nombre, color_hex = "", ""

    if metodo_color == "Paleta (rápido)":
        color_nombre = st.selectbox("Color", list(OPCIONES_COLOR.keys()))
        color_hex = OPCIONES_COLOR[color_nombre]
        st.markdown(f"<div style='background:{color_hex};width:100%;height:25px;border:1px solid #000'></div>", unsafe_allow_html=True)

    elif metodo_color == "Picker (preciso)":
        color_hex = st.color_picker("Elige color exacto", "#cccccc")
        color_nombre = "Personalizado"
        st.markdown(f"<div style='background:{color_hex};width:100%;height:25px;border:1px solid #000'></div>", unsafe_allow_html=True)

    else:  # Desde imagen con sliders
        if foto is None:
            st.warning("Sube una foto para poder elegir el color con sliders.")
        else:
            img = Image.open(io.BytesIO(foto.getvalue())).convert("RGB")
            w, h = img.size
            st.image(img, caption=f"Imagen ({w}×{h}px)", use_container_width=True)
            x = st.slider("X", 0, w - 1, w // 2)
            y = st.slider("Y", 0, h - 1, h // 2)
            rgb = img.getpixel((x, y))
            color_hex = hex_from_rgb(rgb)
            color_nombre = "Desde imagen"
            st.write(f"Pixel ({x},{y}) → {color_hex}")
            st.markdown(f"<div style='background:{color_hex};width:100%;height:25px;border:1px solid #000'></div>", unsafe_allow_html=True)

    enviado = st.form_submit_button("➕ Añadir prenda")
    if enviado:
        if metodo_color == "Desde imagen (sliders)" and (not color_hex or foto is None):
            st.error("Para 'Desde imagen' necesitas subir foto y mover sliders para elegir píxel.")
        else:
            nueva = pd.DataFrame([{
                "Categoria": categoria,
                "Tipo": tipo,
                "ColorNombre": color_nombre,
                "ColorHex": color_hex,
                "FotoBase64": file_to_b64(foto)
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
                    st.image(img_bytes, caption=f"{row['Categoria']} ({row['ColorNombre'] or row['ColorHex']})", use_container_width=True)

st.caption("💡 Consejo: guarda tu XML tras añadir prendas, y recárgalo cuando vuelvas a la web.")

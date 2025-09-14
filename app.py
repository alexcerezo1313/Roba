import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io

st.set_page_config(page_title="👕 Armario Digital", page_icon="🧥", layout="wide")

# ---------- Config ----------
COLUMNS = ["Tipo", "Categoria", "Largo", "ColorHex", "FotoBase64"]
SCHEMA_VERSION = "2.0"

TIPO_CATS = {
    "Parte de arriba": ["Camiseta", "Camisa", "Sudadera"],
    "Parte de abajo": ["Pantalón", "Short", "Falda"],
    "Zapatos": ["Zapatillas", "Botas", "Sandalias"]
}
LARGOS = ["Corto", "Largo"]

# ---------- Estado ----------
if "armario" not in st.session_state:
    st.session_state["armario"] = pd.DataFrame(columns=COLUMNS)

# Guardamos el último tipo para resetear categoría cuando cambie
if "ultimo_tipo" not in st.session_state:
    st.session_state["ultimo_tipo"] = None

st.title("👕 Armario Digital")
st.caption("Añade prendas, color y (opcional) fotografía. Exporta/Importa tu armario en XML.")

# ---------- Utils: imágenes & XML ----------
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
        SubElement(item, "type").text = str(row.get("Tipo", ""))
        SubElement(item, "category").text = str(row.get("Categoria", ""))
        SubElement(item, "length").text = str(row.get("Largo", ""))
        SubElement(item, "color_hex").text = str(row.get("ColorHex", ""))
        # Foto opcional como base64
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
                "Tipo": (item.findtext("type") or "").strip(),
                "Categoria": (item.findtext("category") or "").strip(),
                "Largo": (item.findtext("length") or "").strip(),
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
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])

    with c1:
        tipo = st.selectbox("Tipo", list(TIPO_CATS.keys()), key="tipo_sel")

    # Si el tipo cambió respecto al render anterior, reseteamos la categoría almacenada
    if st.session_state["ultimo_tipo"] != st.session_state["tipo_sel"]:
        st.session_state.pop("categoria_sel", None)  # forzamos que tome el primer valor por defecto
        st.session_state["ultimo_tipo"] = st.session_state["tipo_sel"]

    opciones_categoria = TIPO_CATS[st.session_state["tipo_sel"]]
    with c2:
        # Si no hay categoría guardada o no está en las opciones actuales, elegimos la primera
        default_index = 0
        if "categoria_sel" in st.session_state and st.session_state["categoria_sel"] in opciones_categoria:
            default_index = opciones_categoria.index(st.session_state["categoria_sel"])
        categoria = st.selectbox("Categoría", opciones_categoria, index=default_index, key="categoria_sel")

    with c3:
        largo = st.selectbox("Largo", LARGOS)

    with c4:
        color_hex = st.color_picker("Color", "#cccccc", help="Selecciona el color de la prenda")

    foto = st.file_uploader("Fotografía (opcional)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    enviado = st.form_submit_button("➕ Añadir prenda")
    if enviado:
        nueva = pd.DataFrame([{
            "Tipo": tipo,
            "Categoria": categoria,
            "Largo": largo,
            "ColorHex": color_hex,
            "FotoBase64": file_to_b64(foto)
        }], columns=COLUMNS)
        st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
        st.success(f"{categoria} añadida ✅")

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

# ---------- Filtros y vista ----------
st.subheader("🗂 Tu Armario")
if st.session_state["armario"].empty:
    st.info("Aún no has añadido ninguna prenda.")
else:
    f1, f2, f3 = st.columns(3)
    with f1:
        tipos = ["Todos"] + sorted(st.session_state["armario"]["Tipo"].unique().tolist())
        f_tipo = st.selectbox("Filtrar por tipo", tipos)
    with f2:
        largos = ["Todos"] + sorted(st.session_state["armario"]["Largo"].unique().tolist())
        f_largo = st.selectbox("Filtrar por largo", largos)
    with f3:
        # No filtramos por capa (ya no existe), solo mostramos color
        st.write(" ")


    df = st.session_state["armario"].copy()
    if f_tipo != "Todos":
        df = df[df["Tipo"] == f_tipo]
    if f_largo != "Todos":
        df = df[df["Largo"] == f_largo]

    st.dataframe(df.drop(columns=["FotoBase64"]), use_container_width=True)

    # Vista de miniaturas (si hay fotos)
    thumbs = df[df["FotoBase64"].str.len() > 0]
    if not thumbs.empty:
        st.write("### 📸 Miniaturas")
        cols = st.columns(6)
        i = 0
        for _, row in thumbs.iterrows():
            img_bytes = b64_to_bytes(row["FotoBase64"])
            if img_bytes:
                with cols[i % 6]:
                    st.image(img_bytes, caption=f"{row['Categoria']} ({row['ColorHex']})", use_container_width=True)
                i += 1

st.caption("💡 Consejo: descarga tu XML tras añadir prendas, y vuelve a cargarlo cuando regreses a la web.")

import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="👕 Armario Digital", page_icon="🧥", layout="wide")

# ---------- Config ----------
COLUMNS = ["Categoria", "Tipo", "Color1Nombre", "Color1Hex", "Color2Nombre", "Color2Hex", "FotoBase64"]
SCHEMA_VERSION = "6.0"

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

def pick_color_from_canvas(img: Image.Image, label: str) -> str:
    """Muestra la imagen en un canvas; el usuario hace clic (modo 'point') y devolvemos el color del píxel."""
    w, h = img.size
    max_width = min(600, w)  # no gigante
    scale = max_width / w
    disp_w, disp_h = int(w * scale), int(h * scale)

    st.caption(f"{label} — Haz clic en la imagen para elegir un píxel")
    canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",  # sin relleno
        stroke_width=8, stroke_color="#00FF00",
        background_image=img.resize((disp_w, disp_h)),
        update_streamlit=True, height=disp_h, width=disp_w,
        drawing_mode="point", key=f"canvas_{label}"
    )
    if canvas.json_data and canvas.json_data.get("objects"):
        # Tomamos el último punto
        obj = canvas.json_data["objects"][-1]
        # Coordenadas del lienzo (escala aplicada)
        cx, cy = obj.get("left", 0), obj.get("top", 0)
        # Convertir a coords originales de la imagen
        ox = int(round(cx / scale))
        oy = int(round(cy / scale))
        ox = max(0, min(ox, w - 1))
        oy = max(0, min(oy, h - 1))
        rgb = img.convert("RGB").getpixel((ox, oy))
        return hex_from_rgb(rgb)
    return ""

def detectar_color_secundario(img: Image.Image, n_colors: int = 3, min_dist: int = 24) -> tuple[str, bool]:
    """
    Usa cuantización para encontrar colores dominantes.
    Devuelve (hex_sugerido, hay_secundario)
    """
    # Reducimos tamaño para velocidad
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    # Cuantizar a n colores
    q = thumb.convert("RGB").quantize(colors=n_colors, method=Image.MEDIANCUT)
    pal = q.getpalette()[:n_colors * 3]  # lista RGB aplanada
    counts = q.getcolors() or []  # [(count, index), ...]
    if not counts:
        return "", False
    # Ordenar por frecuencia (desc)
    counts.sort(reverse=True, key=lambda t: t[0])
    # Color principal
    _, idx0 = counts[0]
    r0, g0, b0 = pal[idx0 * 3: idx0 * 3 + 3]
    # Buscar secundario suficientemente distinto
    for _, idx in counts[1:]:
        r, g, b = pal[idx * 3: idx * 3 + 3]
        # distancia Manhattan simple
        if abs(r - r0) + abs(g - g0) + abs(b - b0) >= min_dist:
            return hex_from_rgb((r, g, b)), True
    return "", False

# ---------- UI ----------
st.title("👕 Armario Digital")
st.caption("Elige colores: paleta, picker exacto o clic sobre la imagen (canvas). Añade color secundario y detecta si lo hay.")

with st.form("nueva_prenda", clear_on_submit=False):
    c1, c2 = st.columns([1, 1])

    with c1:
        categoria = st.selectbox("Categoría", CATEGORIAS)
        tipo = st.selectbox("Tipo (corto/largo)", TIPOS)

        # --------- COLOR PRINCIPAL (3 métodos) ----------
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
            foto1 = st.file_uploader("Fotografía para color principal (necesaria para clic)", type=["png", "jpg", "jpeg"], key="foto1")
            if foto1:
                img1 = Image.open(io.BytesIO(foto1.getvalue()))
                color1_hex = pick_color_from_canvas(img1, "Color principal")
                color1_name = "Desde imagen"
                if color1_hex:
                    color_preview(color1_hex)
            else:
                st.info("Sube una foto para seleccionar el color con clic.")

        # --------- COLOR SECUNDARIO (opcional, mismos 3 métodos) ----------
        st.markdown("---")
        usar_color2 = st.checkbox("Añadir color secundario (opcional)")
        color2_name, color2_hex = "", ""
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
                foto2 = st.file_uploader("Fotografía para secundario (o reutiliza la de arriba)", type=["png", "jpg", "jpeg"], key="foto2")
                if not foto2 and 'foto1' in st.session_state and st.session_state['foto1'] is not None:
                    # si no sube segunda, intentamos usar la primera (si existe en session, depende de Streamlit)
                    foto2 = st.session_state['foto1']
                if foto2:
                    img2 = Image.open(io.BytesIO(foto2.getvalue()))
                    color2_hex = pick_color_from_canvas(img2, "Color secundario")
                    color2_name = "Desde imagen"
                    if color2_hex:
                        color_preview(color2_hex)
                else:
                    st.info("Sube una foto para el secundario o usa la misma del principal.")

        # --------- Botón: ¿Hay color secundario? (detección automática) ----------
        st.markdown("---")
        st.write("🔎 Detección automática de color secundario (opcional)")
        foto_auto = st.file_uploader("Imagen para analizar (puede ser la misma)", type=["png", "jpg", "jpeg"], key="foto_auto")
        if st.button("¿Hay color secundario?"):
            if not foto_auto:
                st.warning("Sube una imagen para analizar.")
            else:
                img_auto = Image.open(io.BytesIO(foto_auto.getvalue()))
                sugerido_hex, hay = detectar_color_secundario(img_auto)
                if hay:
                    st.success(f"Sí, parece haber un color secundario: **{sugerido_hex}**")
                    color_preview(sugerido_hex)
                else:
                    st.info("No se detecta un color secundario claro.")

    with c2:
        # Foto de la prenda (opcional). Se guarda tal cual en el XML.
        foto_prenda = st.file_uploader("Fotografía de la prenda (opcional)", type=["png", "jpg", "jpeg"], key="fotoprenda")
        if foto_prenda:
            st.image(foto_prenda, caption="Vista previa", use_container_width=True)

    enviado = st.form_submit_button("➕ Añadir prenda")
    if enviado:
        # Validaciones básicas para métodos "Desde imagen"
        if (metodo1 == "Desde imagen (clic)" and not color1_hex):
            st.error("Selecciona el color principal clicando en la imagen.")
        elif (usar_color2 and getattr(st, 'metodo2', None) == "Desde imagen (clic)" and not color2_hex):
            st.error("Selecciona el color secundario clicando en la imagen.")
        else:
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

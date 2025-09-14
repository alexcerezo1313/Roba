import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io
from PIL import Image
import numpy as np

st.set_page_config(page_title="👕 Armario Digital — Colores por Paleta/Hex/Auto", page_icon="🧥", layout="wide")

# ---------------- Config ----------------
CATEGORIAS = ["Camiseta", "Camisa", "Sudadera", "Pantalón", "Short", "Falda", "Zapatillas", "Botas", "Sandalias"]
TIPOS = ["Corto", "Largo"]
COLUMNS = ["Categoria", "Tipo", "Color1Hex", "Color2Hex", "FotoBase64"]
SCHEMA_VERSION = "11.0"

# Paleta sencilla de colores comunes en ropa (nombre -> hex)
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

# Parámetros AUTOMÁTICOS optimizados para 1 prenda por foto
AUTO_PARAMS = dict(
    center_keep=0.95,          # Mantener 95% central
    ignore_bg_mode="auto",     # Ignorar fondo claro/oscuro automáticamente
    exclude_skin=False,        # Asumimos solo prenda
    exclude_border=True,       # Excluir color parecido a los bordes (fondo)
    sat_min=0.12,              # Saturación mínima
    val_min=0.12,              # Brillo mínimo
    val_max=0.98,              # Brillo máximo
    border_sim_thresh=0.18,    # Tolerancia similitud a bordes (HSV)
    k_palette=7,               # Tamaño paleta para cuantización
    min_dist=0.28,             # Separación mínima entre principal/secundario (HSV)
    min_prop_secondary=0.10,   # Proporción mínima del secundario
    user_bg_hex=None           # Sin color de fondo explícito
)

# Estado
if "armario" not in st.session_state:
    st.session_state["armario"] = pd.DataFrame(columns=COLUMNS)

# ---------------- Utilidades ----------------
def file_to_b64(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def b64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64)
    except Exception:
        return b""

def hex_from_rgb(rgb):
    r, g, b = [int(x) for x in rgb[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"

def rgb_to_hsv_vec(rgb_arr_uint8: np.ndarray) -> np.ndarray:
    """RGB [0..255] -> HSV [0..1]. rgb_arr_uint8 shape: (N,3) uint8."""
    rgb = rgb_arr_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    diff = mx - mn

    # Hue
    h = np.zeros_like(mx)
    mask_r = (mx == r) & (diff != 0)
    mask_g = (mx == g) & (diff != 0)
    mask_b = (mx == b) & (diff != 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = (h / 6.0) % 1.0

    # Saturation & Value
    s = np.zeros_like(mx)
    s[mx != 0] = diff[mx != 0] / mx[mx != 0]
    v = mx
    return np.stack([h, s, v], axis=1)  # (N,3)

def color_distance_hsv(c1_rgb, c2_rgb):
    """Distancia en HSV perceptual aproximada (0..~2)."""
    def single(rgb):
        arr = np.array(rgb, dtype=np.uint8).reshape(1, 3)
        return rgb_to_hsv_vec(arr)[0]
    h1, s1, v1 = single(c1_rgb)
    h2, s2, v2 = single(c2_rgb)
    dh = min(abs(h1 - h2), 1 - abs(h1 - h2)) * 2.0
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)
    return dh * 0.6 + ds * 0.8 + dv * 0.4

def rgb_to_ycrcb_vec(rgb_arr_uint8: np.ndarray) -> np.ndarray:
    """Convierte RGB a YCrCb aprox (BT.601). shape: (N,3) -> (N,3) en rango 0..255"""
    R = rgb_arr_uint8[:, 0].astype(np.float32)
    G = rgb_arr_uint8[:, 1].astype(np.float32)
    B = rgb_arr_uint8[:, 2].astype(np.float32)
    Y  =  0.299*R + 0.587*G + 0.114*B
    Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
    Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    out = np.stack([Y, Cr, Cb], axis=1)
    out = np.clip(out, 0, 255)
    return out

def skin_mask(rgb_arr_uint8: np.ndarray) -> np.ndarray:
    """Máscara de piel aproximada en YCrCb (umbral clásico). True donde ES piel."""
    ycrcb = rgb_to_ycrcb_vec(rgb_arr_uint8)
    Cr = ycrcb[:, 1]
    Cb = ycrcb[:, 2]
    return (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)

def quantize_colors(arr_rgb_uint8: np.ndarray, k=6):
    """Cuantiza con Pillow (MEDIANCUT) y devuelve [(count, (r,g,b)), ...] desc."""
    if arr_rgb_uint8.size == 0:
        return []
    n = arr_rgb_uint8.shape[0]
    w = int(np.ceil(np.sqrt(n)))
    h = int(np.ceil(n / w))
    pad = w * h - n
    if pad > 0:
        arr_rgb_uint8 = np.vstack([arr_rgb_uint8, np.tile(arr_rgb_uint8[-1], (pad, 1))])
    img = Image.fromarray(arr_rgb_uint8.reshape(h, w, 3), mode="RGB")
    q = img.quantize(colors=max(2, k), method=Image.MEDIANCUT)
    pal = q.getpalette()[:k*3]
    counts = q.getcolors() or []
    results = []
    for count, idx in counts:
        r, g, b = pal[idx*3:idx*3+3]
        results.append((int(count), (int(r), int(g), int(b))))
    results.sort(key=lambda t: t[0], reverse=True)
    return results

def estimate_border_colors(arr_rgb_uint8: np.ndarray, width: int, height: int, border_frac: float = 0.06):
    """Promedios de color en los 4 bordes para excluir fondo similar."""
    bf = max(1, int(min(width, height) * border_frac))
    img2d = arr_rgb_uint8.reshape(height, width, 3)
    top = img2d[:bf, :, :].reshape(-1, 3).mean(0)
    bottom = img2d[-bf:, :, :].reshape(-1, 3).mean(0)
    left = img2d[:, :bf, :].reshape(-1, 3).mean(0)
    right = img2d[:, -bf:, :].reshape(-1, 3).mean(0)
    L = [tuple(int(x) for x in top), tuple(int(x) for x in bottom), tuple(int(x) for x in left), tuple(int(x) for x in right)]
    return L

@st.cache_data(show_spinner=False)
def auto_colors_from_image_bytes(
    image_bytes: bytes,
    params: dict
):
    """Cachea por imagen+parámetros para no recalcular si no cambia nada."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return auto_colors_from_image(img, **params)

def auto_colors_from_image(
    image: Image.Image,
    center_keep: float = AUTO_PARAMS["center_keep"],
    sat_min: float = AUTO_PARAMS["sat_min"],
    val_min: float = AUTO_PARAMS["val_min"],
    val_max: float = AUTO_PARAMS["val_max"],
    ignore_bg_mode: str = AUTO_PARAMS["ignore_bg_mode"],  # "auto" | "ninguno" | "claro" | "oscuro"
    exclude_skin: bool = AUTO_PARAMS["exclude_skin"],
    exclude_border: bool = AUTO_PARAMS["exclude_border"],
    border_sim_thresh: float = AUTO_PARAMS["border_sim_thresh"],
    k_palette: int = AUTO_PARAMS["k_palette"],
    min_dist: float = AUTO_PARAMS["min_dist"],
    min_prop_secondary: float = AUTO_PARAMS["min_prop_secondary"],
    user_bg_hex: str | None = AUTO_PARAMS["user_bg_hex"]
):
    """
    Devuelve (c1_hex, c2_hex, meta_dict) totalmente AUTOMÁTICO para 1 prenda.
    """
    # --- Preproceso y reducción ---
    w0, h0 = image.size
    scale = min(640 / max(w0, h0), 1.0)
    if scale < 1.0:
        image = image.resize((int(w0*scale), int(h0*scale)), Image.LANCZOS)
    w, h = image.size
    arr = np.array(image)                  # (H, W, 3)
    flat = arr.reshape(-1, 3).astype(np.uint8)

    # --- Región central ---
    mask = np.ones((h, w), dtype=bool)
    if center_keep < 1.0:
        keep_w = int(w * center_keep)
        keep_h = int(h * center_keep)
        x0 = (w - keep_w) // 2
        y0 = (h - keep_h) // 2
        central = np.zeros_like(mask)
        central[y0:y0+keep_h, x0:x0+keep_w] = True
        mask &= central

    # --- HSV filtros básicos ---
    hsv = rgb_to_hsv_vec(flat)  # (N,3)
    hsv2d = hsv.reshape(h, w, 3)
    mask &= (hsv2d[:, :, 1] >= sat_min) & (hsv2d[:, :, 2] >= val_min) & (hsv2d[:, :, 2] <= val_max)

    # --- Modo fondo ---
    if ignore_bg_mode == "auto":
        mask &= ~((hsv2d[:, :, 2] > 0.92) & (hsv2d[:, :, 1] < 0.20))
        mask &= ~(hsv2d[:, :, 2] < 0.08)
    elif ignore_bg_mode == "claro":
        mask &= ~((hsv2d[:, :, 2] > 0.90) & (hsv2d[:, :, 1] < 0.25))
    elif ignore_bg_mode == "oscuro":
        mask &= ~(hsv2d[:, :, 2] < 0.12)

    # --- Excluir tono piel (probablemente no necesario) ---
    if exclude_skin:
        skin2d = skin_mask(flat).reshape(h, w)
        mask &= ~skin2d

    # --- Excluir colores similares a los bordes (fondo) ---
    border_rgbs = []
    if exclude_border:
        border_rgbs = estimate_border_colors(flat, width=w, height=h, border_frac=0.06)
        hsv_all = hsv  # ya calculado
        for bg in border_rgbs:
            hsv_bg = rgb_to_hsv_vec(np.array([bg], dtype=np.uint8))[0]
            dh = np.minimum(np.abs(hsv_all[:, 0] - hsv_bg[0]), 1 - np.abs(hsv_all[:, 0] - hsv_bg[0])) * 2.0
            ds = np.abs(hsv_all[:, 1] - hsv_bg[1])
            dv = np.abs(hsv_all[:, 2] - hsv_bg[2])
            dist = dh * 0.6 + ds * 0.8 + dv * 0.4
            mask &= (dist.reshape(h, w) > border_sim_thresh)

    # --- Excluir color de fondo proporcionado (no usado por defecto) ---
    if user_bg_hex:
        bg = tuple(int(user_bg_hex[i:i+2], 16) for i in (1, 3, 5))
        hsv_bg = rgb_to_hsv_vec(np.array([bg], dtype=np.uint8))[0]
        dh = np.minimum(np.abs(hsv[:, 0] - hsv_bg[0]), 1 - np.abs(hsv[:, 0] - hsv_bg[0])) * 2.0
        ds = np.abs(hsv[:, 1] - hsv_bg[1])
        dv = np.abs(hsv[:, 2] - hsv_bg[2])
        dist = dh * 0.6 + ds * 0.8 + dv * 0.4
        mask &= (dist.reshape(h, w) > 0.14)

    # --- Salvaguarda: si queda muy poco, relajamos a solo región central ---
    if mask.sum() < (h * w * 0.02):
        mask = np.ones((h, w), dtype=bool)
        if center_keep < 1.0:
            keep_w = int(w * center_keep)
            keep_h = int(h * center_keep)
            x0 = (w - keep_w) // 2
            y0 = (h - keep_h) // 2
            central = np.zeros_like(mask)
            central[y0:y0+keep_h, x0:x0+keep_w] = True
            mask &= central

    selected = flat[mask.reshape(-1)]
    if selected.size == 0:
        return None, None, {"pixels_used": 0, "border_samples": [hex_from_rgb(c) for c in border_rgbs], "palette": []}

    # --- Cuantización y elección principal/secundario ---
    k_eff = int(np.clip(np.sqrt(selected.shape[0] / 300), 3, k_palette))
    pal = quantize_colors(selected, k=k_eff)  # [(count, (r,g,b)), ...]
    total = sum(c for c, _ in pal) if pal else 1
    palette_hex = [(cnt, hex_from_rgb(rgb), round(cnt/total, 3)) for cnt, rgb in pal]

    c1 = None
    c2 = None
    if pal:
        c1 = pal[0][1]
        for cnt, rgb in pal[1:]:
            prop = cnt / total
            if prop >= min_prop_secondary and color_distance_hsv(c1, rgb) >= min_dist:
                c2 = rgb
                break

    return (
        hex_from_rgb(c1) if c1 else None,
        hex_from_rgb(c2) if c2 else None,
        {"pixels_used": int(selected.shape[0]), "border_samples": [hex_from_rgb(c) for c in border_rgbs], "palette": palette_hex}
    )

def swatch(hex_code, label=None):
    if not hex_code:
        return
    lab = f"&nbsp;{label}" if label else ""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<div style='width:28px;height:28px;border-radius:6px;border:1px solid #000;background:{hex_code};'></div>"
        f"<code>{hex_code}</code>{lab}"
        f"</div>", unsafe_allow_html=True
    )

# ---------------- UI: Crear prenda ----------------
st.title("👕 Armario Digital — Paleta / Hex / Auto (misma foto)")
st.caption("Elige cómo fijar colores. Si usas detección automática, la foto de la prenda es obligatoria y se reutiliza para guardar y para detectar.")

with st.form("nueva_prenda", clear_on_submit=False):
    left, right = st.columns([1, 1])

    with left:
        categoria = st.selectbox("Categoría", CATEGORIAS)
        tipo = st.selectbox("Tipo", TIPOS)

        st.markdown("### 🎨 Método para **color principal**")
        metodo_principal = st.radio(
            "Selecciona método",
            ["Paleta", "Hex (picker)", "Automático desde imagen"],
            index=0
        )

        st.markdown("### 🎨 Método para **color secundario**")
        metodo_secundario = st.radio(
            "Selecciona método",
            ["Ninguno", "Paleta", "Hex (picker)", "Automático desde imagen"],
            index=0
        )

    with right:
        # La MISMA foto sirve para guardar y (si hace falta) para detectar
        foto_prenda = st.file_uploader("📦 Foto de la prenda", type=["png","jpg","jpeg"], key="foto_prenda")
        if foto_prenda:
            st.image(foto_prenda, caption="Vista previa prenda (esta misma se usa para detectar)", use_container_width=True)

    # Campos de color (según método)
    color1_hex = ""
    color2_hex = ""

    # Principal
    if metodo_principal == "Paleta":
        nombre = st.selectbox("Color principal (paleta)", list(PALETA.keys()), key="paleta_p")
        color1_hex = PALETA[nombre]
        swatch(color1_hex, "(paleta)")
    elif metodo_principal == "Hex (picker)":
        color1_hex = st.color_picker("Color principal (hex)", "#cccccc", key="hex_p")
        swatch(color1_hex, "(hex)")
    else:
        if not foto_prenda:
            st.warning("El método 'Automático desde imagen' requiere subir la foto de la prenda.")
        else:
            # Detectamos una sola vez para no recalcular
            img_bytes = foto_prenda.getvalue()
            c1_auto, c2_auto, meta = auto_colors_from_image_bytes(img_bytes, AUTO_PARAMS)
            if c1_auto:
                color1_hex = c1_auto
                swatch(color1_hex, "(auto)")
            else:
                st.info("No se pudo detectar el color principal con esta imagen.")

    # Secundario
    if metodo_secundario == "Ninguno":
        color2_hex = ""
    elif metodo_secundario == "Paleta":
        nombre2 = st.selectbox("Color secundario (paleta)", list(PALETA.keys()), key="paleta_s")
        color2_hex = PALETA[nombre2]
        swatch(color2_hex, "(paleta)")
    elif metodo_secundario == "Hex (picker)":
        color2_hex = st.color_picker("Color secundario (hex)", "#bbbbbb", key="hex_s")
        swatch(color2_hex, "(hex)")
    else:  # Automático desde imagen
        if not foto_prenda:
            st.warning("El método 'Automático desde imagen' (secundario) requiere subir la foto de la prenda.")
        else:
            img_bytes = foto_prenda.getvalue()
            # Si ya hicimos auto arriba y sigue en cache, esto es instantáneo
            c1_auto, c2_auto, meta = auto_colors_from_image_bytes(img_bytes, AUTO_PARAMS)
            if c2_auto:
                color2_hex = c2_auto
                swatch(color2_hex, "(auto)")
            else:
                st.info("No se detecta color secundario claro (prenda monocolor).")

    # Validaciones antes de guardar
    errores = []
    if metodo_principal == "Automático desde imagen" and not foto_prenda:
        errores.append("Falta foto para detectar el color principal.")
    if metodo_secundario == "Automático desde imagen" and not foto_prenda:
        errores.append("Falta foto para detectar el color secundario.")
    if metodo_principal != "Automático desde imagen" and not color1_hex:
        errores.append("Selecciona un color principal válido.")
    if metodo_secundario != "Automático desde imagen" and metodo_secundario != "Ninguno":
        if not color2_hex:
            errores.append("Selecciona un color secundario válido o elige 'Ninguno'.")

    enviado = st.form_submit_button("➕ Añadir prenda")
    if enviado:
        if errores:
            for e in errores:
                st.error(e)
        else:
            nueva = pd.DataFrame([{
                "Categoria": categoria,
                "Tipo": tipo,
                "Color1Hex": color1_hex or "",
                "Color2Hex": color2_hex or "",
                "FotoBase64": file_to_b64(foto_prenda)  # guarda la misma imagen (si hay)
            }], columns=COLUMNS)
            st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
            st.success(f"{categoria} añadida ✅")

# ---------------- Exportar / Importar XML ----------------
st.subheader("💾 Guardar / Cargar tu armario (XML)")
col1, col2 = st.columns(2)

def df_to_xml_bytes(df: pd.DataFrame) -> bytes:
    root = Element("wardrobe", attrib={"version": SCHEMA_VERSION})
    for _, row in df.iterrows():
        item = SubElement(root, "item")
        SubElement(item, "category").text = str(row["Categoria"])
        SubElement(item, "type").text = str(row["Tipo"])
        SubElement(item, "color1_hex").text = str(row.get("Color1Hex",""))
        SubElement(item, "color2_hex").text = str(row.get("Color2Hex",""))
        if row.get("FotoBase64"):
            SubElement(item, "photo_b64").text = row["FotoBase64"]
    buf = io.BytesIO()
    ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()

def xml_bytes_to_df(xml_bytes: bytes) -> pd.DataFrame:
    try:
        root = fromstring(xml_bytes)
        rows = []
        for item in root.findall("item"):
            rows.append({
                "Categoria": item.findtext("category",""),
                "Tipo": item.findtext("type",""),
                "Color1Hex": item.findtext("color1_hex",""),
                "Color2Hex": item.findtext("color2_hex",""),
                "FotoBase64": item.findtext("photo_b64","") or "",
            })
        return pd.DataFrame(rows, columns=COLUMNS)
    except Exception as e:
        st.error(f"XML no válido: {e}")
        return pd.DataFrame(columns=COLUMNS)

with col1:
    if not st.session_state["armario"].empty:
        xml_bytes = df_to_xml_bytes(st.session_state["armario"])
        fname = f"armario_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.xml"
        st.download_button("⬇️ Descargar XML", data=xml_bytes, file_name=fname, mime="application/xml")
    else:
        st.info("Añade prendas para poder descargar tu XML.")

with col2:
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

# ---------------- Vista ----------------
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
                    st.image(
                        img_bytes,
                        caption=f"{row['Categoria']} ({row['Color1Hex']}" + (f", {row['Color2Hex']}" if row['Color2Hex'] else "") + ")",
                        use_container_width=True
                    )

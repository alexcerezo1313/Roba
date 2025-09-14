import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io
from PIL import Image
import numpy as np

st.set_page_config(page_title="👕 Armario Digital", page_icon="🧥", layout="wide")

# ---------------- Config ----------------
CATEGORIAS = ["Camiseta", "Camisa", "Sudadera", "Pantalón", "Short", "Falda", "Zapatillas", "Botas", "Sandalias"]
TIPOS = ["Corto", "Largo"]
COLUMNS = ["Categoria", "Tipo", "Color1Hex", "Color1Name", "Color2Hex", "Color2Name", "FotoBase64"]
SCHEMA_VERSION = "13.2"

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

AUTO_PARAMS = dict(
    center_keep=0.95,
    ignore_bg_mode="auto",
    exclude_skin=False,
    exclude_border=True,
    sat_min=0.12,
    val_min=0.12,
    val_max=0.98,
    border_sim_thresh=0.18,
    k_palette=7,
    min_dist=0.28,
    min_prop_secondary=0.10,
    user_bg_hex=None
)

# Estado
if "armario" not in st.session_state:
    st.session_state["armario"] = pd.DataFrame(columns=COLUMNS)

# ---------------- Utils ----------------
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
    rgb = rgb_arr_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask_r = (mx == r) & (diff != 0)
    mask_g = (mx == g) & (diff != 0)
    mask_b = (mx == b) & (diff != 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = (h / 6.0) % 1.0
    s = np.zeros_like(mx)
    s[mx != 0] = diff[mx != 0] / mx[mx != 0]
    v = mx
    return np.stack([h, s, v], axis=1)

def color_distance_hsv(c1_rgb, c2_rgb):
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
    R = rgb_arr_uint8[:, 0].astype(np.float32)
    G = rgb_arr_uint8[:, 1].astype(np.float32)
    B = rgb_arr_uint8[:, 2].astype(np.float32)
    Y  =  0.299*R + 0.587*G + 0.114*B
    Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
    Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    out = np.stack([Y, Cr, Cb], axis=1)
    return np.clip(out, 0, 255)

def skin_mask(rgb_arr_uint8: np.ndarray) -> np.ndarray:
    ycrcb = rgb_to_ycrcb_vec(rgb_arr_uint8)
    Cr = ycrcb[:, 1]; Cb = ycrcb[:, 2]
    return (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)

def quantize_colors(arr_rgb_uint8: np.ndarray, k=6):
    if arr_rgb_uint8.size == 0:
        return []
    n = arr_rgb_uint8.shape[0]
    w = int(np.ceil(np.sqrt(n))); h = int(np.ceil(n / w))
    pad = w * h - n
    if pad > 0:
        arr_rgb_uint8 = np.vstack([arr_rgb_uint8, np.tile(arr_rgb_uint8[-1], (pad, 1))])
    img = Image.fromarray(arr_rgb_uint8.reshape(h, w, 3), mode="RGB")
    q = img.quantize(colors=max(2, k), method=Image.MEDIANCUT)
    pal = q.getpalette()[:k*3]
    counts = q.getcolors() or []
    res = []
    for count, idx in counts:
        r, g, b = pal[idx*3:idx*3+3]
        res.append((int(count), (int(r), int(g), int(b))))
    res.sort(key=lambda t: t[0], reverse=True)
    return res

def estimate_border_colors(arr_rgb_uint8: np.ndarray, width: int, height: int, border_frac: float = 0.06):
    bf = max(1, int(min(width, height) * border_frac))
    img2d = arr_rgb_uint8.reshape(height, width, 3)
    top = img2d[:bf, :, :].reshape(-1, 3).mean(0)
    bottom = img2d[-bf:, :, :].reshape(-1, 3).mean(0)
    left = img2d[:, :bf, :].reshape(-1, 3).mean(0)
    right = img2d[:, -bf:, :].reshape(-1, 3).mean(0)
    return [tuple(int(x) for x in top), tuple(int(x) for x in bottom),
            tuple(int(x) for x in left), tuple(int(x) for x in right)]

def auto_colors_from_image(image: Image.Image, params: dict):
    # Reducción
    w0, h0 = image.size
    scale = min(640 / max(w0, h0), 1.0)
    if scale < 1.0:
        image = image.resize((int(w0*scale), int(h0*scale)), Image.LANCZOS)
    w, h = image.size
    arr = np.array(image)
    flat = arr.reshape(-1, 3).astype(np.uint8)

    # Región central
    center_keep = params["center_keep"]
    mask = np.ones((h, w), dtype=bool)
    if center_keep < 1.0:
        keep_w = int(w * center_keep); keep_h = int(h * center_keep)
        x0 = (w - keep_w) // 2; y0 = (h - keep_h) // 2
        central = np.zeros_like(mask); central[y0:y0+keep_h, x0:x0+keep_w] = True
        mask &= central

    # HSV filtros
    hsv = rgb_to_hsv_vec(flat); hsv2d = hsv.reshape(h, w, 3)
    mask &= (hsv2d[:, :, 1] >= params["sat_min"]) & (hsv2d[:, :, 2] >= params["val_min"]) & (hsv2d[:, :, 2] <= params["val_max"])

    # Fondo
    mode = params["ignore_bg_mode"]
    if mode == "auto":
        mask &= ~((hsv2d[:, :, 2] > 0.92) & (hsv2d[:, :, 1] < 0.20))
        mask &= ~(hsv2d[:, :, 2] < 0.08)
    elif mode == "claro":
        mask &= ~((hsv2d[:, :, 2] > 0.90) & (hsv2d[:, :, 1] < 0.25))
    elif mode == "oscuro":
        mask &= ~(hsv2d[:, :, 2] < 0.12)

    # Piel (opcional)
    if params["exclude_skin"]:
        skin2d = skin_mask(flat).reshape(h, w)
        mask &= ~skin2d

    # Bordes
    border_rgbs = []
    if params["exclude_border"]:
        border_rgbs = estimate_border_colors(flat, w, h, border_frac=0.06)
        hsv_all = hsv
        for bg in border_rgbs:
            hsv_bg = rgb_to_hsv_vec(np.array([bg], dtype=np.uint8))[0]
            dh = np.minimum(np.abs(hsv_all[:, 0] - hsv_bg[0]), 1 - np.abs(hsv_all[:, 0] - hsv_bg[0])) * 2.0
            ds = np.abs(hsv_all[:, 1] - hsv_bg[1])
            dv = np.abs(hsv_all[:, 2] - hsv_bg[2])
            dist = dh * 0.6 + ds * 0.8 + dv * 0.4
            mask &= (dist.reshape(h, w) > params["border_sim_thresh"])

    # Salvaguarda
    if mask.sum() < (h * w * 0.02):
        mask = np.ones((h, w), dtype=bool)
        if center_keep < 1.0:
            keep_w = int(w * center_keep); keep_h = int(h * center_keep)
            x0 = (w - keep_w) // 2; y0 = (h - keep_h) // 2
            central = np.zeros_like(mask); central[y0:y0+keep_h, x0:x0+keep_w] = True
            mask &= central

    selected = flat[mask.reshape(-1)]
    if selected.size == 0:
        return None, None, {"pixels_used": 0, "border_samples": [hex_from_rgb(c) for c in border_rgbs], "palette": []}

    # Cuantización
    k_eff = int(np.clip(np.sqrt(selected.shape[0] / 300), 3, params["k_palette"]))
    pal = quantize_colors(selected, k=k_eff)
    total = sum(c for c, _ in pal) if pal else 1
    palette_hex = [(cnt, hex_from_rgb(rgb), round(cnt/total, 3)) for cnt, rgb in pal]

    # Elegir (distintos)
    c1 = None; c2 = None
    if pal:
        c1 = pal[0][1]
        for cnt, rgb in pal[1:]:
            prop = cnt / total
            if prop >= params["min_prop_secondary"] and color_distance_hsv(c1, rgb) >= params["min_dist"]:
                c2 = rgb; break

    return (
        hex_from_rgb(c1) if c1 else None,
        hex_from_rgb(c2) if c2 else None,
        {"pixels_used": int(selected.shape[0]), "border_samples": [hex_from_rgb(c) for c in border_rgbs], "palette": palette_hex}
    )

def swatch_with_label(hex_code, title):
    st.markdown(f"**{title}**")
    if hex_code:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:28px;height:28px;border-radius:6px;border:1px solid #000;background:{hex_code};'></div>"
            f"<code>{hex_code}</code>"
            f"</div>", unsafe_allow_html=True
        )
    else:
        st.info("—")

# ---------------- UI: Crear / Detectar ----------------
st.title("👕 Armario Digital — Un método para principal y secundario")
st.caption("Si hay secundario, se usa el mismo método. En **Automático** la foto es obligatoria y el botón solo **detecta colores** (no guarda).")

with st.form("nueva_prenda", clear_on_submit=False):
    left, right = st.columns([1, 1])

    with left:
        categoria = st.selectbox("Categoría", CATEGORIAS)
        tipo = st.selectbox("Tipo", TIPOS)

        metodo = st.radio(
            "Método de color (aplica a principal y secundario):",
            ["Paleta", "Hex (picker)", "Automático desde imagen"],
            index=0
        )
        hay_secundario = st.checkbox("¿La prenda tiene color secundario?", value=False)

    with right:
        foto_prenda = st.file_uploader(
            "📦 Foto de la prenda (obligatoria si usas 'Automático')",
            type=["png","jpg","jpeg"], key="foto_prenda"
        )
        if foto_prenda:
            st.image(foto_prenda, caption="Vista previa (esta misma se usa para detectar si toca)", use_container_width=True)

    # Valores a guardar/mostrar
    color1_hex, color2_hex = "", ""
    color1_name, color2_name = "", ""

    # --- Método Paleta ---
    if metodo == "Paleta":
        col1, col2 = st.columns(2)
        with col1:
            n1 = st.selectbox("Color principal (paleta)", list(PALETA.keys()), key="paleta_p")
            color1_hex = PALETA[n1]
            swatch_with_label(color1_hex, "Principal")
        if hay_secundario:
            with col2:
                n2 = st.selectbox("Color secundario (paleta)", list(PALETA.keys()), key="paleta_s")
                color2_hex = PALETA[n2]
                swatch_with_label(color2_hex, "Secundario")
        # Nombres opcionales
        st.markdown("**Nombres (opcionales)**")
        c1, c2 = st.columns(2)
        with c1:
            color1_name = st.text_input("Nombre para principal", value="")
        if hay_secundario:
            with c2:
                color2_name = st.text_input("Nombre para secundario", value="")

    # --- Método Hex ---
    elif metodo == "Hex (picker)":
        col1, col2 = st.columns(2)
        with col1:
            color1_hex = st.color_picker("Color principal (hex)", "#cccccc", key="hex_p")
            swatch_with_label(color1_hex, "Principal")
        if hay_secundario:
            with col2:
                color2_hex = st.color_picker("Color secundario (hex)", "#bbbbbb", key="hex_s")
                swatch_with_label(color2_hex, "Secundario")
        # Nombres opcionales
        st.markdown("**Nombres (opcionales)**")
        c1, c2 = st.columns(2)
        with c1:
            color1_name = st.text_input("Nombre para principal", value="")
        if hay_secundario:
            with c2:
                color2_name = st.text_input("Nombre para secundario", value="")

    # --- Método Automático (muestra detección/ajustes pero NO guarda) ---
    else:
        if not foto_prenda:
            st.warning("El método 'Automático desde imagen' requiere subir la foto de la prenda.")
        else:
            img = Image.open(io.BytesIO(foto_prenda.getvalue())).convert("RGB")
            c1_auto, c2_auto, meta = auto_colors_from_image(img, AUTO_PARAMS)

            # Caso A: detecta ambos distintos
            if c1_auto and c2_auto and c1_auto.lower() != c2_auto.lower():
                color1_hex, color2_hex = c1_auto, c2_auto
                st.markdown("### 🎯 Detección automática")
                cA, cB = st.columns(2)
                with cA: swatch_with_label(color1_hex, "Principal (auto)")
                with cB: swatch_with_label(color2_hex, "Secundario (auto)")

                ok = st.radio("¿Se ha detectado correctamente?", ["Sí", "No"], index=0, horizontal=True, key="ok_auto")
                if ok == "Sí":
                    st.markdown("**Nombres (opcionales)**")
                    c1c, c2c = st.columns(2)
                    with c1c:
                        color1_name = st.text_input("Nombre para principal", value="", key="name_auto_p")
                    if hay_secundario:
                        with c2c:
                            color2_name = st.text_input("Nombre para secundario", value="", key="name_auto_s")
                else:
                    st.info("Corrige los colores con HEX:")
                    colx, coly = st.columns(2)
                    with colx:
                        color1_hex = st.color_picker("Principal (HEX)", color1_hex, key="auto_fix_p")
                        swatch_with_label(color1_hex, "Principal (ajustado)")
                    with coly:
                        if hay_secundario:
                            color2_hex = st.color_picker("Secundario (HEX)", color2_hex, key="auto_fix_s")
                            swatch_with_label(color2_hex, "Secundario (ajustado)")

            # Caso B: detecta solo principal -> mostrar directamente el picker manual del secundario
            elif c1_auto and (not c2_auto or c1_auto.lower() == (c2_auto or "").lower()):
                color1_hex = c1_auto
                st.markdown("### 🎯 Detección automática")
                swatch_with_label(color1_hex, "Principal (auto)")

                if hay_secundario:
                    st.warning("No se ha detectado un secundario distinto.")
                    use_swap = st.checkbox("El detectado como principal en realidad es el SECUNDARIO; definir PRINCIPAL en HEX", key="swap_sec")
                    if use_swap:
                        color2_hex = color1_hex
                        color1_hex = st.color_picker("Principal (HEX)", "#cccccc", key="auto_swap_prin")
                        cA, cB = st.columns(2)
                        with cA: swatch_with_label(color1_hex, "Principal (definido)")
                        with cB: swatch_with_label(color2_hex, "Secundario (auto)")
                    else:
                        color2_hex = st.color_picker("Secundario (manual)", "#bbbbbb", key="auto_add_sec")
                        swatch_with_label(color2_hex, "Secundario (manual)")

                # Nombres opcionales
                st.markdown("**Nombres (opcionales)**")
                c1c, c2c = st.columns(2)
                with c1c:
                    color1_name = st.text_input("Nombre para principal", value="", key="name_auto_p_only")
                if hay_secundario:
                    with c2c:
                        color2_name = st.text_input("Nombre para secundario", value="", key="name_auto_s_only")

            # Caso C: no detecta nada útil
            else:
                st.info("No se pudo detectar colores. Define por HEX:")
                colx, coly = st.columns(2)
                with colx:
                    color1_hex = st.color_picker("Principal (HEX)", "#cccccc", key="auto_fail_p")
                    swatch_with_label(color1_hex, "Principal")
                if hay_secundario:
                    with coly:
                        color2_hex = st.color_picker("Secundario (HEX)", "#bbbbbb", key="auto_fail_s")
                        swatch_with_label(color2_hex, "Secundario")

                st.markdown("**Nombres (opcionales)**")
                c1c, c2c = st.columns(2)
                with c1c:
                    color1_name = st.text_input("Nombre para principal", value="", key="name_auto_fail_p")
                if hay_secundario:
                    with c2c:
                        color2_name = st.text_input("Nombre para secundario", value="", key="name_auto_fail_s")

    # --- Regla global: si hay secundario, debe ser distinto del principal ---
    if hay_secundario and color1_hex and color2_hex and color1_hex.lower() == color2_hex.lower():
        st.error("El color secundario debe ser distinto del principal. Cambia uno de los dos.")

    # Validaciones básicas para ambos flujos (solo mensajes; el guardado depende del método)
    errores = []
    if metodo == "Automático desde imagen" and not foto_prenda:
        errores.append("Debes subir la foto de la prenda para usar el método Automático.")
    if not color1_hex:
        errores.append("Selecciona o define el color principal.")
    if hay_secundario and not color2_hex:
        errores.append("Has indicado color secundario: defínelo con el mismo método.")
    if hay_secundario and color1_hex and color2_hex and color1_hex.lower() == color2_hex.lower():
        errores.append("El color secundario debe ser distinto del principal.")

    # Etiqueta del botón según método
    submit_label = "🔍 Detectar colores" if metodo == "Automático desde imagen" else "➕ Añadir prenda"
    enviado = st.form_submit_button(submit_label)

    if enviado:
        if errores:
            for e in errores:
                st.error(e)
        else:
            if metodo == "Automático desde imagen":
                # SOLO trabajo con colores, no guardo en la tabla
                st.success("Colores preparados ✅ (no se ha añadido prenda a la tabla).")
                # Mostrar resumen compacto
                csum1, csum2 = st.columns(2)
                with csum1: swatch_with_label(color1_hex, "Principal listo")
                if hay_secundario:
                    with csum2: swatch_with_label(color2_hex, "Secundario listo")
            else:
                # Guardado normal (Paleta / Hex)
                nueva = pd.DataFrame([{
                    "Categoria": categoria,
                    "Tipo": tipo,
                    "Color1Hex": color1_hex or "",
                    "Color1Name": color1_name or "",
                    "Color2Hex": color2_hex if hay_secundario else "",
                    "Color2Name": color2_name if hay_secundario else "",
                    "FotoBase64": file_to_b64(foto_prenda)
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
        SubElement(item, "color1_name").text = str(row.get("Color1Name",""))
        SubElement(item, "color2_hex").text = str(row.get("Color2Hex",""))
        SubElement(item, "color2_name").text = str(row.get("Color2Name",""))
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
                "Color1Name": item.findtext("color1_name",""),
                "Color2Hex": item.findtext("color2_hex",""),
                "Color2Name": item.findtext("color2_name",""),
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

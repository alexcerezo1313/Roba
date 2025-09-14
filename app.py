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
SCHEMA_VERSION = "14.0"

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
# Estado para el flujo Automático (fase 2)
if "auto_state" not in st.session_state:
    st.session_state["auto_state"] = {
        "ready": False, "c1": "", "c2": "", "n1": "", "n2": "",
        "foto_b64": "", "categoria": "", "tipo": "", "hay_sec": False
    }

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
    selected = flat[mask.reshape(-1)]
    if selected.size == 0:
        return None, None, {}

    # Cuantización
    k_eff = int(np.clip(np.sqrt(selected.shape[0] / 300), 3, params["k_palette"]))
    pal = quantize_colors(selected, k=k_eff)
    total = sum(c for c, _ in pal) if pal else 1

    # Elegir (distintos)
    c1 = None; c2 = None
    if pal:
        c1 = pal[0][1]
        for cnt, rgb in pal[1:]:
            prop = cnt / total
            if prop >= params["min_prop_secondary"] and color_distance_hsv(c1, rgb) >= params["min_dist"]:
                c2 = rgb; break

    return (hex_from_rgb(c1) if c1 else None,
            hex_from_rgb(c2) if c2 else None,
            {})

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

# ---------------- UI principal ----------------
st.title("👕 Armario Digital — Paleta / Hex / Automático en 2 fases")
st.caption("En **Automático** primero **Detectar colores**; luego podrás **Añadir prenda**.")

# Datos comunes
c_top1, c_top2, c_top3 = st.columns([1,1,1])
with c_top1:
    categoria = st.selectbox("Categoría", CATEGORIAS, key="ui_categoria")
with c_top2:
    tipo = st.selectbox("Tipo", TIPOS, key="ui_tipo")
with c_top3:
    metodo = st.radio("Método de color", ["Paleta", "Hex (picker)", "Automático desde imagen"], key="ui_metodo")

hay_secundario = st.checkbox("¿La prenda tiene color secundario?", value=False, key="ui_hay_sec")

# ---------------- Método: Paleta ----------------
if metodo == "Paleta":
    with st.form("form_paleta", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            n1 = st.selectbox("Color principal (paleta)", list(PALETA.keys()), key="paleta_p")
            c1_hex = PALETA[n1]
            swatch_with_label(c1_hex, "Principal")
        c2_hex = ""
        if hay_secundario:
            with col2:
                n2 = st.selectbox("Color secundario (paleta)", list(PALETA.keys()), key="paleta_s")
                c2_hex = PALETA[n2]
                swatch_with_label(c2_hex, "Secundario")

        st.markdown("**Nombres (opcionales)**")
        ncol1, ncol2 = st.columns(2)
        with ncol1:
            c1_name = st.text_input("Nombre para principal", value="", key="paleta_name1")
        c2_name = ""
        if hay_secundario:
            with ncol2:
                c2_name = st.text_input("Nombre para secundario", value="", key="paleta_name2")

        foto = st.file_uploader("📦 Foto de la prenda (opcional)", type=["png","jpg","jpeg"], key="paleta_foto")

        errs = []
        if hay_secundario and c1_hex.lower() == c2_hex.lower():
            errs.append("El color secundario debe ser distinto del principal.")
        submit = st.form_submit_button("➕ Añadir prenda")
        if submit:
            if errs:
                for e in errs: st.error(e)
            else:
                nueva = pd.DataFrame([{
                    "Categoria": categoria, "Tipo": tipo,
                    "Color1Hex": c1_hex, "Color1Name": c1_name,
                    "Color2Hex": c2_hex if hay_secundario else "", "Color2Name": c2_name if hay_secundario else "",
                    "FotoBase64": file_to_b64(foto)
                }], columns=COLUMNS)
                st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
                st.success(f"{categoria} añadida ✅")

# ---------------- Método: Hex ----------------
elif metodo == "Hex (picker)":
    with st.form("form_hex", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            c1_hex = st.color_picker("Color principal (hex)", "#cccccc", key="hex_p")
            swatch_with_label(c1_hex, "Principal")
        c2_hex = ""
        if hay_secundario:
            with col2:
                c2_hex = st.color_picker("Color secundario (hex)", "#bbbbbb", key="hex_s")
                swatch_with_label(c2_hex, "Secundario")

        st.markdown("**Nombres (opcionales)**")
        ncol1, ncol2 = st.columns(2)
        with ncol1:
            c1_name = st.text_input("Nombre para principal", value="", key="hex_name1")
        c2_name = ""
        if hay_secundario:
            with ncol2:
                c2_name = st.text_input("Nombre para secundario", value="", key="hex_name2")

        foto = st.file_uploader("📦 Foto de la prenda (opcional)", type=["png","jpg","jpeg"], key="hex_foto")

        errs = []
        if not c1_hex: errs.append("Selecciona el color principal.")
        if hay_secundario and not c2_hex: errs.append("Has indicado color secundario: defínelo.")
        if hay_secundario and c1_hex.lower() == c2_hex.lower():
            errs.append("El color secundario debe ser distinto del principal.")
        submit = st.form_submit_button("➕ Añadir prenda")
        if submit:
            if errs:
                for e in errs: st.error(e)
            else:
                nueva = pd.DataFrame([{
                    "Categoria": categoria, "Tipo": tipo,
                    "Color1Hex": c1_hex, "Color1Name": c1_name,
                    "Color2Hex": c2_hex if hay_secundario else "", "Color2Name": c2_name if hay_secundario else "",
                    "FotoBase64": file_to_b64(foto)
                }], columns=COLUMNS)
                st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
                st.success(f"{categoria} añadida ✅")

# ---------------- Método: Automático (2 fases) ----------------
else:
    # Fase 1: inputs + botón Detectar colores (no guarda)
    foto = st.file_uploader("📦 Foto de la prenda (obligatoria)", type=["png","jpg","jpeg"], key="auto_foto")
    if foto:
        st.image(foto, caption="Vista previa (se usa para detectar y guardar)", use_container_width=True)

    detect = st.button("🔍 Detectar colores")
    if detect:
        errs = []
        if not foto:
            errs.append("Debes subir la foto para detectar colores.")
        if errs:
            for e in errs: st.error(e)
        else:
            img = Image.open(io.BytesIO(foto.getvalue())).convert("RGB")
            c1, c2, _ = auto_colors_from_image(img, AUTO_PARAMS)

            # Si solo hay principal y se quiere secundario, permitir manual/swap después
            st.session_state["auto_state"] = {
                "ready": True,
                "c1": c1 or "",
                "c2": c2 or "",
                "n1": "",
                "n2": "",
                "foto_b64": file_to_b64(foto),
                "categoria": categoria,
                "tipo": tipo,
                "hay_sec": bool(hay_secundario),
            }

    # Fase 2: si ya hay resultado (o si hubo uno previo)
    A = st.session_state["auto_state"]
    if A["ready"]:
        st.markdown("### 🎯 Resultado de la detección")
        c1_hex, c2_hex = A["c1"], A["c2"]
        c1_name, c2_name = A["n1"], A["n2"]
        hay_sec = A["hay_sec"]

        # Tres casos:
        # A) Ambos distintos
        if c1_hex and c2_hex and c1_hex.lower() != c2_hex.lower():
            colA, colB = st.columns(2)
            with colA: swatch_with_label(c1_hex, "Principal (auto)")
            with colB:
                if hay_sec:
                    swatch_with_label(c2_hex, "Secundario (auto)")
                else:
                    st.info("No marcaste color secundario para esta prenda.")

            if hay_sec:
                ok = st.radio("¿Se ha detectado correctamente?", ["Sí", "No"], index=0, horizontal=True, key="auto_ok_both")
            else:
                ok = "Sí"

            if ok == "No":
                colx, coly = st.columns(2)
                with colx:
                    c1_hex = st.color_picker("Principal (HEX)", c1_hex, key="auto_adj_p")
                    swatch_with_label(c1_hex, "Principal (ajustado)")
                if hay_sec:
                    with coly:
                        c2_hex = st.color_picker("Secundario (HEX)", c2_hex, key="auto_adj_s")
                        swatch_with_label(c2_hex, "Secundario (ajustado)")

            st.markdown("**Nombres (opcionales)**")
            n1c, n2c = st.columns(2)
            with n1c:
                c1_name = st.text_input("Nombre para principal", value=c1_name, key="auto_name_p")
            if hay_sec:
                with n2c:
                    c2_name = st.text_input("Nombre para secundario", value=c2_name, key="auto_name_s")

        # B) Solo principal detectado (o secundario igual)
        elif c1_hex and (not c2_hex or c1_hex.lower() == c2_hex.lower()):
            swatch_with_label(c1_hex, "Principal (auto)")

            if hay_sec:
                st.warning("No se detectó un color secundario distinto.")
                swap = st.checkbox("El detectado como principal realmente es el SECUNDARIO; definir PRINCIPAL en HEX", key="auto_swap")
                if swap:
                    c2_hex = c1_hex
                    c1_hex = st.color_picker("Principal (HEX)", "#cccccc", key="auto_swap_p")
                    cA, cB = st.columns(2)
                    with cA: swatch_with_label(c1_hex, "Principal (definido)")
                    with cB: swatch_with_label(c2_hex, "Secundario (auto)")
                else:
                    c2_hex = st.color_picker("Secundario (manual)", "#bbbbbb", key="auto_manual_sec")
                    swatch_with_label(c2_hex, "Secundario (manual)")

            st.markdown("**Nombres (opcionales)**")
            n1c, n2c = st.columns(2)
            with n1c:
                c1_name = st.text_input("Nombre para principal", value=c1_name, key="auto_only_name_p")
            if hay_sec:
                with n2c:
                    c2_name = st.text_input("Nombre para secundario", value=c2_name, key="auto_only_name_s")

        # C) No se detectó nada
        else:
            st.info("No se pudo detectar colores. Define por HEX:")
            colx, coly = st.columns(2)
            with colx:
                c1_hex = st.color_picker("Principal (HEX)", "#cccccc", key="auto_fail_p")
                swatch_with_label(c1_hex, "Principal")
            if hay_sec:
                with coly:
                    c2_hex = st.color_picker("Secundario (HEX)", "#bbbbbb", key="auto_fail_s")
                    swatch_with_label(c2_hex, "Secundario")

            st.markdown("**Nombres (opcionales)**")
            n1c, n2c = st.columns(2)
            with n1c:
                c1_name = st.text_input("Nombre para principal", value=c1_name, key="auto_fail_name_p")
            if hay_sec:
                with n2c:
                    c2_name = st.text_input("Nombre para secundario", value=c2_name, key="auto_fail_name_s")

        # Validaciones + botón Añadir prenda (fase final)
        errs = []
        if not c1_hex: errs.append("Selecciona o define el color principal.")
        if hay_sec and not c2_hex: errs.append("Has indicado color secundario: defínelo.")
        if hay_sec and c1_hex and c2_hex and c1_hex.lower() == c2_hex.lower():
            errs.append("El color secundario debe ser distinto del principal.")

        # Actualizar en estado antes de pulsar
        st.session_state["auto_state"].update({"c1": c1_hex, "c2": c2_hex, "n1": c1_name, "n2": c2_name})

        add = st.button("➕ Añadir prenda")
        if add:
            if errs:
                for e in errs: st.error(e)
            else:
                nueva = pd.DataFrame([{
                    "Categoria": A["categoria"] or categoria,
                    "Tipo": A["tipo"] or tipo,
                    "Color1Hex": c1_hex, "Color1Name": c1_name,
                    "Color2Hex": c2_hex if hay_sec else "", "Color2Name": c2_name if hay_sec else "",
                    "FotoBase64": A["foto_b64"]
                }], columns=COLUMNS)
                st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
                st.success("Prenda añadida ✅")
                # Reset de auto_state
                st.session_state["auto_state"] = {
                    "ready": False, "c1": "", "c2": "", "n1": "", "n2": "",
                    "foto_b64": "", "categoria": "", "tipo": "", "hay_sec": False
                }

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

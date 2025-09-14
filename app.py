import streamlit as st
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree, fromstring
from datetime import datetime
import base64, io, json, os, uuid
from PIL import Image
import numpy as np

st.set_page_config(page_title="👕 Armario Digital + Outfits", page_icon="🧥", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
CATEGORIAS = ["Camiseta", "Camisa", "Sudadera", "Pantalón", "Short", "Falda", "Zapatillas", "Botas", "Sandalias"]
TIPOS      = ["Corto", "Largo"]
ESTILOS    = ["Chándal", "Casual", "Formal"]  # solo estos

SEASONS = ["Invierno", "Primavera", "Verano", "Otoño"]

COLUMNS = [
    "Id", "Nombre", "Categoria", "Tipo",
    "Color1Hex", "Color1Name",
    "Color2Hex", "Color2Name",
    "Estilo",
    "FotoBase64"
]
SCHEMA_VERSION = "17.0"

AUTO_PARAMS = dict(
    center_keep=0.95, ignore_bg_mode="auto",
    exclude_skin=False, exclude_border=True,
    sat_min=0.12, val_min=0.12, val_max=0.98,
    border_sim_thresh=0.18, k_palette=7,
    min_dist=0.28, min_prop_secondary=0.10, user_bg_hex=None
)

DEFAULT_PALETTE = {
    "Negro": "#000000", "Blanco": "#FFFFFF", "Gris": "#808080", "Beige": "#F5F5DC",
    "Marrón": "#8B4513", "Azul marino": "#000080", "Azul claro": "#87CEEB",
    "Rojo": "#FF0000", "Verde": "#008000", "Amarillo": "#FFFF00", "Rosa": "#FFC0CB"
}

# ──────────────────────────────────────────────────────────────────────────────
# Estado
# ──────────────────────────────────────────────────────────────────────────────
if "armario" not in st.session_state:
    st.session_state["armario"] = pd.DataFrame(columns=COLUMNS)

# lista de dicts: {id, name, item_ids, temp_min, temp_max, seasons, elegance}
if "outfits" not in st.session_state:
    st.session_state["outfits"] = []

# Estado del flujo Automático (2 fases)
if "auto_state" not in st.session_state:
    st.session_state["auto_state"] = {
        "ready": False, "c1": "", "c2": "", "n1": "", "n2": "",
        "foto_b64": "", "categoria": "", "tipo": "", "hay_sec": False,
        "nombre_prenda": "", "estilo": ESTILOS[1]
    }

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de color / imagen
# ──────────────────────────────────────────────────────────────────────────────
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
    mx = np.max(rgb, axis=1); mn = np.min(rgb, axis=1); diff = mx - mn
    h = np.zeros_like(mx)
    mask_r = (mx == r) & (diff != 0)
    mask_g = (mx == g) & (diff != 0)
    mask_b = (mx == b) & (diff != 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
    h = (h / 6.0) % 1.0
    s = np.zeros_like(mx); s[mx != 0] = diff[mx != 0] / mx[mx != 0]
    v = mx
    return np.stack([h, s, v], axis=1)

def color_distance_hsv(c1_rgb, c2_rgb):
    def single(rgb):
        arr = np.array(rgb, dtype=np.uint8).reshape(1, 3)
        return rgb_to_hsv_vec(arr)[0]
    h1, s1, v1 = single(c1_rgb); h2, s2, v2 = single(c2_rgb)
    dh = min(abs(h1 - h2), 1 - abs(h1 - h2)) * 2.0
    ds = abs(s1 - s2); dv = abs(v1 - v2)
    return dh * 0.6 + ds * 0.8 + dv * 0.4

def quantize_colors(arr_rgb_uint8: np.ndarray, k=6):
    if arr_rgb_uint8.size == 0: return []
    n = arr_rgb_uint8.shape[0]
    w = int(np.ceil(np.sqrt(n))); h = int(np.ceil(n / w))
    pad = w*h - n
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
        # ordenado desc
    res.sort(key=lambda t: t[0], reverse=True)
    return res

def auto_colors_from_image(image: Image.Image, params: dict):
    w0, h0 = image.size
    scale = min(640 / max(w0, h0), 1.0)
    if scale < 1.0:
        image = image.resize((int(w0*scale), int(h0*scale)), Image.LANCZOS)
    w, h = image.size
    arr = np.array(image); flat = arr.reshape(-1, 3).astype(np.uint8)

    hsv = rgb_to_hsv_vec(flat).reshape(h, w, 3)
    mask = (hsv[:, :, 1] >= params["sat_min"]) & (hsv[:, :, 2] >= params["val_min"]) & (hsv[:, :, 2] <= params["val_max"])
    if params["ignore_bg_mode"] == "auto":
        mask &= ~((hsv[:, :, 2] > 0.92) & (hsv[:, :, 1] < 0.20))
        mask &= ~(hsv[:, :, 2] < 0.08)

    selected = arr.reshape(-1, 3)[mask.reshape(-1)]
    if selected.size == 0:
        return None, None, {}

    k_eff = int(np.clip(np.sqrt(selected.shape[0] / 300), 3, params["k_palette"]))
    pal = quantize_colors(selected, k=k_eff)
    total = sum(c for c, _ in pal) if pal else 1
    c1 = None; c2 = None
    if pal:
        c1 = pal[0][1]
        for cnt, rgb in pal[1:]:
            prop = cnt / total
            if prop >= params["min_prop_secondary"] and color_distance_hsv(c1, rgb) >= params["min_dist"]:
                c2 = rgb; break
    return hex_from_rgb(c1) if c1 else None, hex_from_rgb(c2) if c2 else None, {}

def swatch(hex_code, label=None):
    if not hex_code: return
    lab = f"&nbsp;{label}" if label else ""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<div style='width:22px;height:22px;border-radius:6px;border:1px solid #000;background:{hex_code};'></div>"
        f"<code>{hex_code}</code>{lab}"
        f"</div>", unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# Guía de colores (archivo externo opcional)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GUIDE = {
    "families": ["neutral","black","white","gray","beige","brown","red","orange","yellow","green","cyan","blue","purple","pink"],
    "compatibility": {
        "neutral": ["red","orange","yellow","green","cyan","blue","purple","pink","brown","beige","gray","black","white"],
        "black":    ["red","orange","yellow","green","cyan","blue","purple","pink","beige","gray","white"],
        "white":    ["red","orange","yellow","green","cyan","blue","purple","pink","brown","beige","gray","black"],
        "gray":     ["red","orange","yellow","green","cyan","blue","purple","pink","brown","beige","black","white"],
        "beige":    ["brown","white","black","green","blue","pink","red"],
        "brown":    ["beige","green","blue","white","pink"],
        "red":      ["neutral","white","black","gray","beige","blue","green","pink"],
        "orange":   ["neutral","white","black","gray","blue","green","brown"],
        "yellow":   ["neutral","white","black","gray","blue","green","brown"],
        "green":    ["neutral","white","black","gray","beige","brown","blue","red","yellow"],
        "cyan":     ["neutral","white","black","gray","blue","purple"],
        "blue":     ["neutral","white","black","gray","beige","brown","green","red","yellow","pink"],
        "purple":   ["neutral","white","black","gray","pink","yellow","green"],
        "pink":     ["neutral","white","black","gray","blue","red","beige"]
    }
}

def load_color_guide():
    path = "color_guide.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_GUIDE

GUIDE = load_color_guide()

def hex_to_family(hex_code: str) -> str:
    if not hex_code: return "neutral"
    h = hex_code.lstrip("#")
    if len(h) != 6: return "neutral"
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    # neutrales
    if max(r,g,b) - min(r,g,b) < 18:
        if max(r,g,b) > 230: return "white"
        if max(r,g,b) < 35:  return "black"
        return "gray"
    # beige / marrón simples
    if r>200 and g>200 and b<160: return "beige"
    if r>90 and g<110 and b<110:  return "brown"
    # HSV
    hsv = rgb_to_hsv_vec(np.array([[r,g,b]], dtype=np.uint8))[0]
    hue = hsv[0]*360
    if   hue < 15 or hue >= 345: return "red"
    elif hue < 45:  return "orange"
    elif hue < 70:  return "yellow"
    elif hue < 170: return "green"
    elif hue < 200: return "cyan"
    elif hue < 260: return "blue"
    elif hue < 310: return "purple"
    else:           return "pink"

def colors_compatible(hex_a: str, hex_b: str) -> bool:
    fa, fb = hex_to_family(hex_a), hex_to_family(hex_b)
    comp = GUIDE.get("compatibility", {})
    return fb in comp.get(fa, []) or fa in comp.get(fb, [])

def is_neon(hex_code: str, s_thr: float = 0.85, v_thr: float = 0.85) -> bool:
    """Detecta colores fosforitos/neón (saturación y valor muy altos)."""
    if not hex_code: return False
    h = hex_code.lstrip("#")
    if len(h) != 6: return False
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    hsv = rgb_to_hsv_vec(np.array([[r,g,b]], dtype=np.uint8))[0]
    s, v = float(hsv[1]), float(hsv[2])
    return (s >= s_thr) and (v >= v_thr)

# ──────────────────────────────────────────────────────────────────────────────
# Cabecera
# ──────────────────────────────────────────────────────────────────────────────
st.title("👕 Armario Digital + 🧩 Outfits")
st.caption("Estilos: Chándal, Casual y Formal. En **Formal** se evitan colores fosforitos/neón.")

# ──────────────────────────────────────────────────────────────────────────────
# Formulario de prenda con flujo de detección automática (2 fases)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## ➕ Añadir prenda")

# Campos básicos
c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
with c0:
    nombre_prenda = st.text_input("Nombre de la prenda", placeholder="Ej. Sudadera azul marino")
with c1:
    categoria = st.selectbox("Categoría", CATEGORIAS)
with c2:
    tipo = st.selectbox("Tipo", TIPOS)
with c3:
    estilo = st.selectbox("Estilo", ESTILOS)

c4, c5 = st.columns([1,1])
with c4:
    metodo = st.radio("Método de color", ["Paleta", "Hex (picker)", "Automático desde imagen"], horizontal=True)
with c5:
    hay_secundario = st.checkbox("¿La prenda tiene color secundario?")

# Imagen pequeña a un lado
ic1, ic2 = st.columns([1,2])
with ic1:
    foto_prenda = st.file_uploader("Foto (obligatoria en 'Automático')", type=["png","jpg","jpeg"])
    if foto_prenda:
        st.image(foto_prenda, caption="Vista previa", use_container_width=True)
with ic2:
    st.info("Consejo: fondos neutros mejoran la detección de color.")

# Paleta / Hex / Automático
color1_hex = ""; color2_hex = ""; color1_name = ""; color2_name = ""

if metodo == "Paleta":
    p1, p2 = st.columns(2)
    with p1:
        nombre_pal = st.selectbox("Principal (paleta)", list(DEFAULT_PALETTE.keys()))
        color1_hex = DEFAULT_PALETTE[nombre_pal]; color1_name = ""
        swatch(color1_hex, "principal")
    if hay_secundario:
        with p2:
            nombre_pal2 = st.selectbox("Secundario (paleta)", list(DEFAULT_PALETTE.keys()))
            color2_hex = DEFAULT_PALETTE[nombre_pal2]; color2_name = ""
            swatch(color2_hex, "secundario")

    n1, n2 = st.columns(2)
    with n1:
        color1_name = st.text_input("Nombre color principal (opcional)")
    if hay_secundario:
        with n2:
            color2_name = st.text_input("Nombre color secundario (opcional)")

elif metodo == "Hex (picker)":
    p1, p2 = st.columns(2)
    with p1:
        color1_hex = st.color_picker("Principal (hex)", "#3366cc")
        swatch(color1_hex, "principal")
    if hay_secundario:
        with p2:
            color2_hex = st.color_picker("Secundario (hex)", "#bbbbbb")
            swatch(color2_hex, "secundario")

    n1, n2 = st.columns(2)
    with n1:
        color1_name = st.text_input("Nombre color principal (opcional)")
    if hay_secundario:
        with n2:
            color2_name = st.text_input("Nombre color secundario (opcional)")

else:
    # Fase 1: Detectar
    det = st.button("🔍 Detectar colores")
    if det:
        errors = []
        if not foto_prenda:
            errors.append("Sube la foto para detectar los colores.")
        if errors:
            for e in errors: st.error(e)
        else:
            img = Image.open(io.BytesIO(foto_prenda.getvalue())).convert("RGB")
            c1, c2, _ = auto_colors_from_image(img, AUTO_PARAMS)
            st.session_state["auto_state"] = {
                "ready": True, "c1": c1 or "", "c2": c2 or "",
                "n1": "", "n2": "", "foto_b64": file_to_b64(foto_prenda),
                "categoria": categoria, "tipo": tipo, "hay_sec": bool(hay_secundario),
                "nombre_prenda": nombre_prenda, "estilo": estilo
            }

    A = st.session_state["auto_state"]
    if A["ready"]:
        st.markdown("#### 🎯 Resultado")
        color1_hex, color2_hex = A["c1"], A["c2"]
        color1_name, color2_name = A["n1"], A["n2"]

        # Caso ambos
        if color1_hex and color2_hex and color1_hex.lower()!=color2_hex.lower():
            cA, cB = st.columns(2)
            with cA: swatch(color1_hex, "principal (auto)")
            with cB: swatch(color2_hex, "secundario (auto)")
            ok = st.radio("¿Correcto?", ["Sí","No"], horizontal=True, key="auto_ok_both")
            if ok=="No":
                c1c, c2c = st.columns(2)
                with c1c:
                    color1_hex = st.color_picker("Principal (ajuste)", color1_hex, key="adj1")
                with c2c:
                    color2_hex = st.color_picker("Secundario (ajuste)", color2_hex, key="adj2")
        # Solo principal o nada
        elif color1_hex:
            swatch(color1_hex, "principal (auto)")
            if hay_secundario:
                st.warning("No se detectó un secundario distinto.")
                swap = st.checkbox("El detectado como principal es el SECUNDARIO; definir PRINCIPAL en hex")
                if swap:
                    color2_hex = color1_hex
                    color1_hex = st.color_picker("Principal (hex)", "#3366cc", key="swaphex")
                else:
                    color2_hex = st.color_picker("Secundario (manual hex)", "#bbbbbb", key="mansec")
                    swatch(color2_hex, "secundario (manual)")
        else:
            st.info("No se detectaron colores. Define por hex:")
            c1c, c2c = st.columns(2)
            with c1c:
                color1_hex = st.color_picker("Principal (hex)", "#3366cc", key="fail1")
            if hay_secundario:
                with c2c:
                    color2_hex = st.color_picker("Secundario (hex)", "#bbbbbb", key="fail2")

        n1, n2 = st.columns(2)
        with n1:
            color1_name = st.text_input("Nombre color principal (opcional)", key="auto_name1")
        if hay_secundario:
            with n2:
                color2_name = st.text_input("Nombre color secundario (opcional)", key="auto_name2")

        # Validaciones
        errs = []
        if not color1_hex: errs.append("Debe haber color principal.")
        if A["hay_sec"] and not color2_hex: errs.append("Has indicado secundario: defínelo.")
        if A["hay_sec"] and color1_hex and color2_hex and color1_hex.lower()==color2_hex.lower():
            errs.append("El secundario debe ser distinto al principal.")
        st.session_state["auto_state"].update({"c1": color1_hex, "c2": color2_hex, "n1": color1_name, "n2": color2_name})

        if st.button("➕ Añadir prenda"):
            if errs:
                for e in errs: st.error(e)
            else:
                nueva = pd.DataFrame([{
                    "Id": str(uuid.uuid4()),
                    "Nombre": A["nombre_prenda"] or nombre_prenda or f"Prenda {datetime.utcnow().strftime('%H%M%S')}",
                    "Categoria": A["categoria"] or categoria,
                    "Tipo": A["tipo"] or tipo,
                    "Color1Hex": color1_hex,
                    "Color1Name": color1_name,
                    "Color2Hex": color2_hex if A["hay_sec"] else "",
                    "Color2Name": color2_name if A["hay_sec"] else "",
                    "Estilo": A["estilo"],
                    "FotoBase64": A["foto_b64"]
                }], columns=COLUMNS)
                st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
                st.success("Prenda añadida ✅")
                st.session_state["auto_state"] = { "ready": False, "c1":"", "c2":"", "n1":"", "n2":"", "foto_b64":"", "categoria":"", "tipo":"", "hay_sec": False, "nombre_prenda":"", "estilo":ESTILOS[1] }

# Guardado directo (Paleta/Hex)
if metodo in ["Paleta","Hex (picker)"]:
    errs = []
    if not nombre_prenda: errs.append("Pon un nombre a la prenda.")
    if not color1_hex: errs.append("Selecciona color principal.")
    if hay_secundario and color2_hex and color1_hex.lower()==color2_hex.lower():
        errs.append("El secundario debe ser distinto al principal.")
    if st.button("💾 Guardar prenda", type="primary"):
        if errs:
            for e in errs: st.error(e)
        else:
            nueva = pd.DataFrame([{
                "Id": str(uuid.uuid4()),
                "Nombre": nombre_prenda,
                "Categoria": categoria,
                "Tipo": tipo,
                "Color1Hex": color1_hex, "Color1Name": color1_name,
                "Color2Hex": color2_hex if hay_secundario else "", "Color2Name": color2_name if hay_secundario else "",
                "Estilo": estilo,
                "FotoBase64": file_to_b64(foto_prenda)
            }], columns=COLUMNS)
            st.session_state["armario"] = pd.concat([st.session_state["armario"], nueva], ignore_index=True)
            st.success("Prenda guardada ✅")

# ──────────────────────────────────────────────────────────────────────────────
# Exportar / Importar armario + outfits (XML ÚNICO)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("💾 Guardar / Cargar armario + outfits (XML único)")
c1, c2 = st.columns(2)

def export_combined_xml(armario_df: pd.DataFrame, outfits: list) -> bytes:
    """
    Estructura:
    <wardrobe version="17.0">
      <items>
        <item>...</item>
      </items>
      <outfits>
        <outfit>
          <id>...</id>
          <name>...</name>
          <temp_min>...</temp_min>
          <temp_max>...</temp_max>
          <elegance>...</elegance>
          <seasons><season>Invierno</season>...</seasons>
          <item_ids><item_id>uuid</item_id>...</item_ids>
        </outfit>
      </outfits>
    </wardrobe>
    """
    root = Element("wardrobe", attrib={"version": SCHEMA_VERSION})

    # items
    items_el = SubElement(root, "items")
    for _, row in armario_df.iterrows():
        it = SubElement(items_el, "item")
        SubElement(it, "id").text          = str(row.get("Id",""))
        SubElement(it, "nombre").text      = str(row.get("Nombre",""))
        SubElement(it, "categoria").text   = str(row.get("Categoria",""))
        SubElement(it, "tipo").text        = str(row.get("Tipo",""))
        SubElement(it, "color1hex").text   = str(row.get("Color1Hex",""))
        SubElement(it, "color1name").text  = str(row.get("Color1Name",""))
        SubElement(it, "color2hex").text   = str(row.get("Color2Hex",""))
        SubElement(it, "color2name").text  = str(row.get("Color2Name",""))
        SubElement(it, "estilo").text      = str(row.get("Estilo",""))
        if row.get("FotoBase64"):
            SubElement(it, "photo_b64").text = row.get("FotoBase64")

    # outfits
    outs_el = SubElement(root, "outfits")
    for o in outfits:
        oe = SubElement(outs_el, "outfit")
        SubElement(oe, "id").text        = str(o.get("id",""))
        SubElement(oe, "name").text      = str(o.get("name",""))
        SubElement(oe, "temp_min").text  = str(o.get("temp_min",""))
        SubElement(oe, "temp_max").text  = str(o.get("temp_max",""))
        SubElement(oe, "elegance").text  = str(o.get("elegance",""))
        # seasons
        seas_el = SubElement(oe, "seasons")
        for s in o.get("seasons", []):
            SubElement(seas_el, "season").text = str(s)
        # item ids
        ids_el = SubElement(oe, "item_ids")
        for iid in o.get("item_ids", []):
            SubElement(ids_el, "item_id").text = str(iid)

    buf = io.BytesIO()
    ElementTree(root).write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()

def import_combined_xml(xml_bytes: bytes):
    """
    Devuelve (df_armario, outfits_list). Soporta XMLs antiguos sin <outfits>.
    """
    try:
        root = fromstring(xml_bytes)
    except Exception as e:
        st.error(f"XML no válido: {e}")
        return pd.DataFrame(columns=COLUMNS), []

    # items (soporta viejo formato sin <items>)
    rows = []
    items_parent = root.find("items")
    if items_parent is not None:
        items = items_parent.findall("item")
    else:
        # backward-compat: items directamente bajo root
        items = root.findall("item")
    for it in items:
        rows.append({
            "Id": it.findtext("id","") or str(uuid.uuid4()),
            "Nombre": it.findtext("nombre",""),
            "Categoria": it.findtext("categoria",""),
            "Tipo": it.findtext("tipo",""),
            "Color1Hex": it.findtext("color1hex",""),
            "Color1Name": it.findtext("color1name",""),
            "Color2Hex": it.findtext("color2hex",""),
            "Color2Name": it.findtext("color2name",""),
            "Estilo": it.findtext("estilo",""),
            "FotoBase64": it.findtext("photo_b64","") or ""
        })
    df_items = pd.DataFrame(rows, columns=COLUMNS)

    # outfits (puede no existir)
    outfits = []
    outs_parent = root.find("outfits")
    if outs_parent is not None:
        for oe in outs_parent.findall("outfit"):
            seasons = [se.text for se in (oe.find("seasons").findall("season") if oe.find("seasons") is not None else [])]
            item_ids = [ie.text for ie in (oe.find("item_ids").findall("item_id") if oe.find("item_ids") is not None else [])]
            outfits.append({
                "id": oe.findtext("id","") or str(uuid.uuid4()),
                "name": oe.findtext("name",""),
                "item_ids": item_ids,
                "temp_min": int(oe.findtext("temp_min","0") or 0),
                "temp_max": int(oe.findtext("temp_max","0") or 0),
                "seasons": seasons,
                "elegance": int(oe.findtext("elegance","3") or 3)
            })

    return df_items, outfits

with c1:
    if not st.session_state["armario"].empty or st.session_state["outfits"]:
        xml_bytes = export_combined_xml(st.session_state["armario"], st.session_state["outfits"])
        fname = f"wardrobe_outfits_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.xml"
        st.download_button("⬇️ Descargar XML combinado", data=xml_bytes, file_name=fname, mime="application/xml")
    else:
        st.info("Añade prendas o outfits para poder descargar el XML.")

with c2:
    up = st.file_uploader("⬆️ Cargar XML combinado", type=["xml"])
    modo = st.radio("Cómo cargar", ["Añadir", "Reemplazar"], horizontal=True)
    if up is not None:
        df_imp, outfits_imp = import_combined_xml(up.read())
        if not df_imp.empty or outfits_imp:
            if modo == "Reemplazar":
                st.session_state["armario"] = df_imp
                st.session_state["outfits"] = outfits_imp
            else:
                # añadir items (evitar duplicar Id)
                cur = st.session_state["armario"]
                new = pd.concat([cur, df_imp], ignore_index=True)
                if not new.empty:
                    new = new.drop_duplicates(subset=["Id"], keep="first")
                st.session_state["armario"] = new
                # añadir outfits (evitar duplicar id)
                cur_o = {o["id"]: o for o in st.session_state["outfits"]}
                for o in outfits_imp:
                    cur_o.setdefault(o["id"], o)
                st.session_state["outfits"] = list(cur_o.values())
            st.success("XML combinado cargado ✅")

# ──────────────────────────────────────────────────────────────────────────────
# OUTFITS (Sidebar)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("👗 Crear / Usar outfit")

def filter_prendas_for_weather(df, tmin, tmax, season, elegance):
    """Filtro por temp/estación/elegancia; en formal se excluye neón."""
    df = df.copy()

    # Largo vs corto por clima
    if tmax >= 23:  # caluroso
        df = df[~((df["Categoria"].isin(["Pantalón"])) & (df["Tipo"]=="Largo"))]
        df = df[~(df["Categoria"].isin(["Botas"]))]  # menos botas
    if tmax <= 12:  # frío
        df = df[~(df["Categoria"].isin(["Short","Falda","Sandalias"]))]

    # Elegancia aproximada por estilo (1 chándal, 2 casual, 4 formal)
    estilo_rank = {"Chándal":1, "Casual":2, "Formal":4}
    df["ElegScore"] = df["Estilo"].map(estilo_rank).fillna(2)

    # En formal (elegancia >=4) no permitimos fosforitos/neón
    if elegance >= 4:
        mask_neon = (df["Color1Hex"].apply(is_neon)) | (df["Color2Hex"].apply(is_neon))
        df = df[~mask_neon]

    # Tolerancia de ±1 en elegancia
    df = df[np.abs(df["ElegScore"] - elegance) <= 1]
    return df

def pick_by_color(top_hex, candidates_hex):
    compatible = [h for h in candidates_hex if colors_compatible(top_hex, h)]
    return compatible if compatible else candidates_hex

def auto_build_outfit(df, tmin, tmax, season, elegance):
    dfw = filter_prendas_for_weather(df, tmin, tmax, season, elegance)
    if dfw.empty: return None

    tops    = dfw[dfw["Categoria"].isin(["Camiseta","Camisa","Sudadera"])]
    bottoms = dfw[dfw["Categoria"].isin(["Pantalón","Short","Falda"])]
    shoes   = dfw[dfw["Categoria"].isin(["Zapatillas","Botas","Sandalias"])]

    if tops.empty or bottoms.empty or shoes.empty:
        return None

    # preferencia por largo si hace frío
    if tmax <= 16 and not tops[tops["Tipo"]=="Largo"].empty:
        top = tops[tops["Tipo"]=="Largo"].sample(1)
    else:
        top = tops.sample(1)
    top_hex = top.iloc[0]["Color1Hex"] or top.iloc[0]["Color2Hex"]

    bottom_pool = bottoms.copy()
    b_hexs = (bottom_pool["Color1Hex"].replace("", pd.NA).fillna(bottom_pool["Color2Hex"])).tolist()
    good_b = pick_by_color(top_hex, b_hexs)
    if not good_b:
        bottom = bottoms.sample(1)
    else:
        bottom = bottom_pool.iloc[[b_hexs.index(np.random.choice(good_b))]]

    # zapatos por clima
    if tmax >= 25 and not shoes[shoes["Categoria"]=="Sandalias"].empty:
        shoe_pool = shoes[shoes["Categoria"]=="Sandalias"]
    elif tmax <= 12 and not shoes[shoes["Categoria"]=="Botas"].empty:
        shoe_pool = shoes[shoes["Categoria"]=="Botas"]
    else:
        shoe_pool = shoes
    shoes_hexs = (shoe_pool["Color1Hex"].replace("", pd.NA).fillna(shoe_pool["Color2Hex"])).tolist()
    good_s = pick_by_color(top_hex, shoes_hexs)
    if not good_s:
        shoe = shoe_pool.sample(1)
    else:
        shoe = shoe_pool.iloc[[shoes_hexs.index(np.random.choice(good_s))]]

    return [top.iloc[0]["Id"], bottom.iloc[0]["Id"], shoe.iloc[0]["Id"]]

# Crear outfit manual
with st.sidebar.expander("✍️ Crear outfit manual", expanded=False):
    if st.session_state["armario"].empty:
        st.info("Añade prendas primero.")
    else:
        name = st.text_input("Nombre del outfit", key="out_name_manual")
        tmin = st.number_input("Temp. mínima", value=12, step=1)
        tmax = st.number_input("Temp. máxima", value=22, step=1)
        seas = st.multiselect("Estaciones", SEASONS, default=["Primavera","Otoño"])
        eleg = st.slider("Elegancia (1 poco · 5 mucho)", 1, 5, 3)
        df = st.session_state["armario"]

        ids = st.multiselect("Prendas (por nombre)", df["Nombre"].tolist())
        chosen = df[df["Nombre"].isin(ids)]

        # aviso si formal y hay neón
        if eleg >= 4 and not chosen.empty:
            neon_mask = (chosen["Color1Hex"].apply(is_neon)) | (chosen["Color2Hex"].apply(is_neon))
            if neon_mask.any():
                st.warning("Has seleccionado colores tipo neón. No se recomiendan para un outfit **Formal**.")

        if st.button("Guardar outfit manual"):
            chosen_ids = chosen["Id"].tolist()
            if not name or not chosen_ids:
                st.error("Pon nombre y elige al menos una prenda.")
            else:
                st.session_state["outfits"].append({
                    "id": str(uuid.uuid4()), "name": name,
                    "item_ids": chosen_ids, "temp_min": int(tmin), "temp_max": int(tmax),
                    "seasons": seas, "elegance": int(eleg)
                })
                st.success("Outfit guardado ✅")

# Crear outfit automático
with st.sidebar.expander("🤖 Crear outfit automático", expanded=True):
    if st.session_state["armario"].empty:
        st.info("Añade prendas primero.")
    else:
        tmin = st.number_input("Temp. mínima", value=12, step=1, key="at_min")
        tmax = st.number_input("Temp. máxima", value=22, step=1, key="at_max")
        season = st.selectbox("Estación", SEASONS, index=1, key="at_season")
        eleg = st.slider("Elegancia (1-5)", 1, 5, 3, key="at_eleg")
        if st.button("Sugerir outfit"):
            ids = auto_build_outfit(st.session_state["armario"], tmin, tmax, season, eleg)
            if not ids:
                st.warning("No he podido sugerir con tus filtros. Relaja condiciones o añade más prendas.")
            else:
                st.success("Sugerencia creada 👇")
                st.session_state["last_suggest"] = ids

        if "last_suggest" in st.session_state:
            ids = st.session_state["last_suggest"]
            df = st.session_state["armario"].set_index("Id")
            st.write("**Propuesta:**")
            for pid in ids:
                row = df.loc[pid]
                sw = row["Color1Hex"] or row["Color2Hex"]
                st.write(f"- {row['Nombre']} · {row['Categoria']} ({row['Tipo']}) · {row['Estilo']}")
                swatch(sw)
            name = st.text_input("Nombre para guardar", key="at_name_save")
            if st.button("Guardar esta propuesta"):
                if not name:
                    st.error("Pon un nombre para el outfit.")
                else:
                    st.session_state["outfits"].append({
                        "id": str(uuid.uuid4()), "name": name,
                        "item_ids": ids, "temp_min": int(st.session_state["at_min"]),
                        "temp_max": int(st.session_state["at_max"]), "seasons": [st.session_state["at_season"]],
                        "elegance": int(st.session_state["at_eleg"])
                    })
                    st.success("Outfit guardado ✅")

# Usar outfit guardado
with st.sidebar.expander("📚 Usar outfit guardado", expanded=False):
    if not st.session_state["outfits"]:
        st.info("Aún no hay outfits guardados.")
    else:
        tmin = st.number_input("Temp. mínima", value=12, step=1, key="use_min")
        tmax = st.number_input("Temp. máxima", value=22, step=1, key="use_max")
        season = st.selectbox("Estación", SEASONS, index=1, key="use_season")
        eleg = st.slider("Elegancia (1-5)", 1, 5, 3, key="use_eleg")
        candidates = [
            o for o in st.session_state["outfits"]
            if o["temp_min"] <= tmin and o["temp_max"] >= tmax
               and season in o["seasons"] and abs(o["elegance"]-eleg)<=1
        ]
        names = [o["name"] for o in candidates]
        pick = st.selectbox("Outfit", names) if names else None
        if pick:
            o = next(x for x in candidates if x["name"]==pick)
            st.write("**Outfit elegido:**")
            df = st.session_state["armario"].set_index("Id")
            for pid in o["item_ids"]:
                if pid in df.index:
                    row = df.loc[pid]
                    st.write(f"- {row['Nombre']} · {row['Categoria']} ({row['Tipo']}) · {row['Estilo']}")
                    swatch(row["Color1Hex"] or row["Color2Hex"])

# ──────────────────────────────────────────────────────────────────────────────
# Vista del armario
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 🗂 Tu Armario")
if st.session_state["armario"].empty:
    st.info("Aún no has añadido ninguna prenda.")
else:
    f1, f2, f3 = st.columns(3)
    with f1:
        f_cat = st.selectbox("Filtrar por categoría", ["Todos"] + sorted(st.session_state["armario"]["Categoria"].unique().tolist()))
    with f2:
        f_tipo = st.selectbox("Filtrar por tipo", ["Todos"] + sorted(st.session_state["armario"]["Tipo"].unique().tolist()))
    with f3:
        f_est = st.selectbox("Filtrar por estilo", ["Todos"] + ESTILOS)

    df = st.session_state["armario"].copy()
    if f_cat != "Todos": df = df[df["Categoria"] == f_cat]
    if f_tipo != "Todos": df = df[df["Tipo"] == f_tipo]
    if f_est != "Todos": df = df[df["Estilo"] == f_est]

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
                        caption=f"{row['Nombre']} ({row['Color1Hex']}" + (f", {row['Color2Hex']}" if row['Color2Hex'] else "") + ")",
                        use_container_width=True
                    )

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from supabase import create_client, Client
import io, uuid, time

st.set_page_config(page_title="Armario Digital (Cloud)", page_icon="🧥", layout="wide")

# =============================================================================
# Supabase: conexión + config
# =============================================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_BUCKET = st.secrets.get("SUPABASE_BUCKET", "wardrobe-photos")  # bucket privado
sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# (Opcional) Comprobación de conexión básica
try:
    _ = sb.table("items").select("id").limit(1).execute()
    st.caption("✅ Conexión con Supabase OK")
except Exception as e:
    st.error("❌ No se pudo conectar con Supabase. Revisa SUPABASE_URL / ANON KEY en Secrets.")
    st.stop()

# =============================================================================
# Catálogos y utilidades
# =============================================================================
CATEGORIAS = ["Camiseta", "Camisa", "Sudadera", "Pantalón", "Short", "Falda", "Zapatillas", "Botas", "Sandalias"]
TIPOS = ["Corto", "Largo"]
ESTILOS = ["Chándal", "Casual", "Formal"]
SEASONS = ["Invierno", "Primavera", "Verano", "Otoño"]
DEFAULT_PALETTE = {
    "Negro": "#000000", "Blanco": "#FFFFFF", "Gris": "#808080", "Beige": "#F5F5DC", "Marrón": "#8B4513",
    "Azul marino": "#000080", "Azul claro": "#87CEEB", "Rojo": "#FF0000", "Verde": "#008000",
    "Amarillo": "#FFFF00", "Rosa": "#FFC0CB"
}

def swatch(hex_code, label=None):
    """Muestra un cuadradito con el color + el hex."""
    if not hex_code:
        return
    lab = f"&nbsp;{label}" if label else ""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<div style='width:22px;height:22px;border-radius:6px;border:1px solid #000;background:{hex_code};'></div>"
        f"<code>{hex_code}</code>{lab}</div>",
        unsafe_allow_html=True
    )

# ---------------- HSV helpers, cuantización y reglas ----------------
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

def hex_from_rgb(rgb):
    r, g, b = [int(x) for x in rgb[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"

def quantize_colors(arr_rgb_uint8: np.ndarray, k=6):
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
    pal = q.getpalette()[:k * 3]
    counts = q.getcolors() or []
    res = []
    for count, idx in counts:
        r, g, b = pal[idx * 3: idx * 3 + 3]
        res.append((int(count), (int(r), int(g), int(b))))
    res.sort(key=lambda t: t[0], reverse=True)
    return res

def auto_colors_from_image(image: Image.Image):
    """Devuelve (hex_principal, hex_secundario | None) con filtros y cuantización."""
    params = dict(sat_min=0.12, val_min=0.12, val_max=0.98, k_palette=7, min_prop_secondary=0.10, min_dist=0.28)
    w0, h0 = image.size
    scale = min(640 / max(w0, h0), 1.0)
    if scale < 1.0:
        image = image.resize((int(w0 * scale), int(h0 * scale)), Image.LANCZOS)
    w, h = image.size
    arr = np.array(image)
    hsv = rgb_to_hsv_vec(arr.reshape(-1, 3)).reshape(h, w, 3)
    mask = (hsv[:, :, 1] >= params["sat_min"]) & (hsv[:, :, 2] >= params["val_min"]) & (hsv[:, :, 2] <= params["val_max"])
    mask &= ~((hsv[:, :, 2] > 0.92) & (hsv[:, :, 1] < 0.20))  # descarta blancos brillantes
    mask &= ~(hsv[:, :, 2] < 0.08)  # descarta sombras
    selected = arr.reshape(-1, 3)[mask.reshape(-1)]
    if selected.size == 0:
        return None, None
    k_eff = int(np.clip(np.sqrt(selected.shape[0] / 300), 3, params["k_palette"]))
    pal = quantize_colors(selected, k=k_eff)
    total = sum(c for c, _ in pal) if pal else 1
    c1 = pal[0][1] if pal else None
    c2 = None

    def dist(a, b):
        a = rgb_to_hsv_vec(np.array([a], dtype=np.uint8))[0]
        b = rgb_to_hsv_vec(np.array([b], dtype=np.uint8))[0]
        dh = min(abs(a[0] - b[0]), 1 - abs(a[0] - b[0])) * 2.0
        ds = abs(a[1] - b[1])
        dv = abs(a[2] - b[2])
        return dh * 0.6 + ds * 0.8 + dv * 0.4

    if pal:
        for cnt, rgb in pal[1:]:
            if (cnt / total) >= params["min_prop_secondary"] and dist(c1, rgb) >= params["min_dist"]:
                c2 = rgb
                break
    return (hex_from_rgb(c1) if c1 else None), (hex_from_rgb(c2) if c2 else None)

def is_neon(hex_code, s_thr=0.85, v_thr=0.85):
    if not hex_code:
        return False
    h = hex_code.lstrip("#")
    if len(h) != 6:
        return False
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    s, v = rgb_to_hsv_vec(np.array([[r, g, b]], dtype=np.uint8))[0][1:]
    return (float(s) >= s_thr) and (float(v) >= v_thr)

def hex_family(hex_code):
    if not hex_code:
        return "neutral"
    h = hex_code.lstrip("#")
    if len(h) != 6:
        return "neutral"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if max(r, g, b) - min(r, g, b) < 18:
        if max(r, g, b) > 230:
            return "white"
        if max(r, g, b) < 35:
            return "black"
        return "gray"
    hsv = rgb_to_hsv_vec(np.array([[r, g, b]], dtype=np.uint8))[0]
    hue = hsv[0] * 360
    if   hue < 15 or hue >= 345: return "red"
    elif hue < 45:  return "orange"
    elif hue < 70:  return "yellow"
    elif hue < 170: return "green"
    elif hue < 200: return "cyan"
    elif hue < 260: return "blue"
    elif hue < 310: return "purple"
    else:           return "pink"

COMPAT = {
    "neutral":["red","orange","yellow","green","cyan","blue","purple","pink","gray","black","white"],
    "black":["red","orange","yellow","green","cyan","blue","purple","pink","beige","gray","white"],
    "white":["red","orange","yellow","green","cyan","blue","purple","pink","brown","beige","gray","black"],
    "gray":["red","orange","yellow","green","cyan","blue","purple","pink","brown","beige","black","white"],
    "beige":["brown","white","black","green","blue","pink","red"],
    "brown":["beige","green","blue","white","pink"],
    "red":["neutral","white","black","gray","beige","blue","green","pink"],
    "orange":["neutral","white","black","gray","blue","green","brown"],
    "yellow":["neutral","white","black","gray","blue","green","brown"],
    "green":["neutral","white","black","gray","beige","brown","blue","red","yellow"],
    "cyan":["neutral","white","black","gray","blue","purple"],
    "blue":["neutral","white","black","gray","beige","brown","green","red","yellow","pink"],
    "purple":["neutral","white","black","gray","pink","yellow","green"],
    "pink":["neutral","white","black","gray","blue","red","beige"],
}
def colors_compatible(a, b):
    fa, fb = hex_family(a), hex_family(b)
    return (fb in COMPAT.get(fa, [])) or (fa in COMPAT.get(fb, []))

# =============================================================================
# Autenticación (sidebar) + sesión robusta
# =============================================================================
with st.sidebar:
    st.header("🔑 Cuenta")
    modo = st.radio("Acceso", ["Entrar", "Registrarme"], horizontal=True)
    email = st.text_input("Email")
    password = st.text_input("Contraseña", type="password")
    cA, cB = st.columns(2)

    if modo == "Entrar":
        if cA.button("Entrar"):
            try:
                sb.auth.sign_in_with_password({"email": email, "password": password})
                st.toast("Sesión iniciada ✅")
                time.sleep(0.4)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        if cA.button("Registrarme"):
            try:
                sb.auth.sign_up({"email": email, "password": password})
                st.success("Registro correcto. Revisa tu correo si pide verificación.")
            except Exception as e:
                st.error(f"Error: {e}")

    if cB.button("Salir"):
        sb.auth.sign_out()
        st.info("Sesión cerrada")
        time.sleep(0.3)
        st.rerun()

# Obtener usuario de forma segura (evita crash si no hay sesión todavía)
try:
    session = sb.auth.get_session()
    user = session.user if session and getattr(session, "user", None) else None
    if user is None:
        resp = sb.auth.get_user()  # fallback
        user = getattr(resp, "user", None)
except Exception:
    user = None

if not user:
    st.info("Inicia sesión para usar tu armario en la nube.")
    st.stop()

st.success(f"Conectado como: {user.email}")

# =============================================================================
# Storage (fotos): subir y firmar URLs
# =============================================================================
def upload_image(user_id: str, file) -> str:
    """Sube imagen al bucket privado y devuelve la ruta (key)."""
    if file is None:
        return ""
    ext = file.name.split(".")[-1].lower()
    key = f"{user_id}/{uuid.uuid4()}.{ext}"
    sb.storage.from_(SUPABASE_BUCKET).upload(key, file.getvalue(), {"content-type": file.type})
    return key

def signed_url(path: str, expires_sec: int = 3600) -> str:
    if not path:
        return ""
    res = sb.storage.from_(SUPABASE_BUCKET).create_signed_url(path, expires_sec)
    return res.get("signedURL") or res.get("signed_url") or ""

# =============================================================================
# DB helpers
# =============================================================================
def db_items_fetch():
    res = sb.table("items").select("*").eq("user_id", user.id).order("created_at").execute()
    return pd.DataFrame(res.data or [])

def db_item_insert(d):
    sb.table("items").insert({"user_id": user.id, **d}).execute()

def db_item_update(item_id, d):
    sb.table("items").update(d).eq("id", item_id).eq("user_id", user.id).execute()

def db_item_delete(item_id):
    sb.table("items").delete().eq("id", item_id).eq("user_id", user.id).execute()

def db_outfits_fetch():
    res = sb.table("outfits").select("*").eq("user_id", user.id).order("created_at").execute()
    return pd.DataFrame(res.data or [])

def db_outfit_insert(d):
    sb.table("outfits").insert({"user_id": user.id, **d}).execute()

def db_outfit_update(outfit_id, d):
    sb.table("outfits").update(d).eq("id", outfit_id).eq("user_id", user.id).execute()

def db_outfit_delete(outfit_id):
    sb.table("outfits").delete().eq("id", outfit_id).eq("user_id", user.id).execute()

# =============================================================================
# Añadir prenda (3 métodos color, auto con 2 fases)
# =============================================================================
st.title("👕 Armario Digital (Cloud)")

st.subheader("➕ Añadir prenda")
c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
with c0:
    nombre_prenda = st.text_input("Nombre de la prenda", placeholder="Ej. Sudadera azul marino")
with c1:
    categoria = st.selectbox("Categoría", CATEGORIAS)
with c2:
    tipo = st.selectbox("Tipo", TIPOS)
with c3:
    estilo = st.selectbox("Estilo", ESTILOS)

c4, c5 = st.columns([1, 1])
with c4:
    metodo = st.radio("Método de color", ["Paleta", "Hex (picker)", "Automático desde imagen"], horizontal=True)
with c5:
    hay_sec = st.checkbox("¿Tiene color secundario?")

ic1, ic2 = st.columns([1, 2])
with ic1:
    foto = st.file_uploader("Foto (obligatoria en 'Automático')", type=["png", "jpg", "jpeg"])
    if foto:
        st.image(foto, caption="Vista previa", use_container_width=True)
with ic2:
    st.caption("Consejo: fondo neutro mejora la detección automática.")

color1_hex = color2_hex = color1_name = color2_name = ""

if metodo == "Paleta":
    p1, p2 = st.columns(2)
    with p1:
        c1name = st.selectbox("Principal (paleta)", list(DEFAULT_PALETTE.keys()))
        color1_hex = DEFAULT_PALETTE[c1name]
        swatch(color1_hex, "principal")
    if hay_sec:
        with p2:
            c2name = st.selectbox("Secundario (paleta)", list(DEFAULT_PALETTE.keys()))
            color2_hex = DEFAULT_PALETTE[c2name]
            swatch(color2_hex, "secundario")
    n1, n2 = st.columns(2)
    with n1:
        color1_name = st.text_input("Nombre color principal (opcional)")
    if hay_sec:
        with n2:
            color2_name = st.text_input("Nombre color secundario (opcional)")

elif metodo == "Hex (picker)":
    p1, p2 = st.columns(2)
    with p1:
        color1_hex = st.color_picker("Principal (hex)", "#3366cc")
        swatch(color1_hex, "principal")
    if hay_sec:
        with p2:
            color2_hex = st.color_picker("Secundario (hex)", "#bbbbbb")
            swatch(color2_hex, "secundario")
    n1, n2 = st.columns(2)
    with n1:
        color1_name = st.text_input("Nombre color principal (opcional)")
    if hay_sec:
        with n2:
            color2_name = st.text_input("Nombre color secundario (opcional)")

else:
    # Flujo de dos fases: detectar y luego permitir corrección
    if "auto_state" not in st.session_state:
        st.session_state["auto_state"] = {"ready": False, "c1": "", "c2": ""}

    if st.button("🔍 Detectar colores"):
        if not foto:
            st.error("Sube la foto para detectar los colores.")
        else:
            img = Image.open(io.BytesIO(foto.getvalue())).convert("RGB")
            c1, c2 = auto_colors_from_image(img)
            st.session_state["auto_state"] = {"ready": True, "c1": c1 or "", "c2": c2 or ""}

    A = st.session_state["auto_state"]
    if A["ready"]:
        color1_hex, color2_hex = A["c1"], A["c2"]
        st.markdown("#### 🎯 Resultado")
        if color1_hex and color2_hex and color1_hex.lower() != color2_hex.lower():
            a, b = st.columns(2)
            with a:
                swatch(color1_hex, "principal (auto)")
            with b:
                swatch(color2_hex, "secundario (auto)")
            ok = st.radio("¿Correcto?", ["Sí", "No"], horizontal=True, key="okboth")
            if ok == "No":
                x, y = st.columns(2)
                with x:
                    color1_hex = st.color_picker("Principal (ajuste)", color1_hex)
                with y:
                    color2_hex = st.color_picker("Secundario (ajuste)", color2_hex)
        elif color1_hex:
            swatch(color1_hex, "principal (auto)")
            if hay_sec:
                st.warning("No se detectó un secundario distinto.")
                swap = st.checkbox("El detectado como principal es el SECUNDARIO; definir PRINCIPAL en hex")
                if swap:
                    color2_hex = color1_hex
                    color1_hex = st.color_picker("Principal (hex)", "#3366cc")
                else:
                    color2_hex = st.color_picker("Secundario (manual hex)", "#bbbbbb")
                    swatch(color2_hex, "secundario (manual)")
        else:
            st.info("No se detectaron colores. Define por hex:")
            x, y = st.columns(2)
            with x:
                color1_hex = st.color_picker("Principal (hex)", "#3366cc")
            if hay_sec:
                with y:
                    color2_hex = st.color_picker("Secundario (hex)", "#bbbbbb")

        a, b = st.columns(2)
        with a:
            color1_name = st.text_input("Nombre color principal (opcional)", key="n1auto")
        if hay_sec:
            with b:
                color2_name = st.text_input("Nombre color secundario (opcional)", key="n2auto")

# Validaciones + Guardado
errs = []
if not nombre_prenda:
    errs.append("Pon un nombre a la prenda.")
if hay_sec and color2_hex and color1_hex and color2_hex.lower() == color1_hex.lower():
    errs.append("El color secundario debe ser distinto al principal.")
if metodo == "Automático desde imagen" and ("auto_state" not in st.session_state or not st.session_state["auto_state"]["ready"]):
    errs.append("Pulsa 'Detectar colores' y ajusta si hace falta.")

if st.button("💾 Guardar prenda", type="primary"):
    if errs:
        for e in errs:
            st.error(e)
    else:
        photo_path = upload_image(user.id, foto) if foto else ""
        db_item_insert({
            "nombre": nombre_prenda,
            "categoria": categoria,
            "tipo": tipo,
            "estilo": estilo,
            "color1_hex": color1_hex or "",
            "color1_name": color1_name or "",
            "color2_hex": (color2_hex or "") if hay_sec else "",
            "color2_name": (color2_name or "") if hay_sec else "",
            "photo_path": photo_path
        })
        st.success("Prenda guardada en la nube ✅")
        if "auto_state" in st.session_state:
            st.session_state["auto_state"]["ready"] = False

st.divider()

# =============================================================================
# Armario: filtros, miniaturas, editar/borrar
# =============================================================================
st.subheader("🗂 Tus prendas")
items_df = db_items_fetch()
if items_df.empty:
    st.info("Aún no tienes prendas.")
else:
    f1, f2, f3 = st.columns(3)
    with f1:
        fcat = st.selectbox("Categoría", ["Todos"] + sorted(items_df["categoria"].dropna().unique().tolist()))
    with f2:
        ftip = st.selectbox("Tipo", ["Todos"] + sorted(items_df["tipo"].dropna().unique().tolist()))
    with f3:
        fest = st.selectbox("Estilo", ["Todos"] + ESTILOS)

    df = items_df.copy()
    if fcat != "Todos":
        df = df[df["categoria"] == fcat]
    if ftip != "Todos":
        df = df[df["tipo"] == ftip]
    if fest != "Todos":
        df = df[df["estilo"] == fest]

    st.dataframe(df.drop(columns=["user_id", "photo_path"]), use_container_width=True)

    sel = st.selectbox("Editar/borrar prenda", [""] + df["nombre"].tolist())
    if sel:
        row = df[df["nombre"] == sel].iloc[0]
        new_name = st.text_input("Nuevo nombre", row["nombre"])
        new_hex = st.color_picker("Nuevo color principal", row["color1_hex"] or "#000000")
        cc1, cc2 = st.columns(2)
        if cc1.button("Actualizar prenda"):
            db_item_update(row["id"], {"nombre": new_name, "color1_hex": new_hex})
            st.success("Actualizada ✅")
            st.rerun()
        if cc2.button("Eliminar prenda"):
            db_item_delete(row["id"])
            st.success("Eliminada ✅")
            st.rerun()

    thumbs = df[df["photo_path"].notna() & (df["photo_path"] != "")]
    if not thumbs.empty:
        st.write("### 📸 Miniaturas")
        cols = st.columns(6)
        for i, row in thumbs.iterrows():
            url = signed_url(row["photo_path"])
            if url:
                with cols[i % 6]:
                    st.image(url, caption=row["nombre"], use_container_width=True)

st.divider()

# =============================================================================
# Outfits: listado, crear manual/auto, editar/borrar
# =============================================================================
st.subheader("👗 Outfits")
outfits_df = db_outfits_fetch()
if outfits_df.empty:
    st.info("Aún no hay outfits guardados.")
else:
    st.dataframe(outfits_df.drop(columns=["user_id"]), use_container_width=True)
    sel_o = st.selectbox("Editar/borrar outfit", [""] + outfits_df["name"].tolist())
    if sel_o:
        row = outfits_df[outfits_df["name"] == sel_o].iloc[0]
        new_name = st.text_input("Nuevo nombre outfit", row["name"])
        oc1, oc2 = st.columns(2)
        if oc1.button("Actualizar outfit"):
            db_outfit_update(row["id"], {"name": new_name})
            st.success("Outfit actualizado ✅")
            st.rerun()
        if oc2.button("Eliminar outfit"):
            db_outfit_delete(row["id"])
            st.success("Outfit eliminado ✅")
            st.rerun()

# ---- Crear outfit manual ----
st.markdown("### ✍️ Crear outfit manual")
o_name = st.text_input("Nombre del outfit")
c7, c8 = st.columns(2)
with c7:
    tmin = st.number_input("Temp. mínima", value=12, step=1)
with c8:
    tmax = st.number_input("Temp. máxima", value=22, step=1)
seasons = st.multiselect("Estaciones", SEASONS, default=["Primavera", "Otoño"])
eleg = st.slider("Elegancia (1-5)", 1, 5, 3)

choices = st.multiselect("Prendas (por nombre)", items_df["nombre"].tolist())
item_ids = items_df[items_df["nombre"].isin(choices)]["id"].tolist()

# Aviso neón si Formal
if eleg >= 4 and choices:
    chosen = items_df[items_df["nombre"].isin(choices)]
    neon_mask = chosen["color1_hex"].apply(is_neon) | chosen["color2_hex"].apply(is_neon)
    if neon_mask.any():
        st.warning("Has elegido colores tipo neón; no recomendado para un outfit **Formal**.")

if st.button("💾 Guardar outfit manual"):
    if not o_name or not item_ids:
        st.error("Pon nombre y al menos una prenda.")
    else:
        db_outfit_insert({
            "name": o_name,
            "temp_min": int(tmin),
            "temp_max": int(tmax),
            "seasons": seasons,
            "elegance": int(eleg),
            "item_ids": item_ids
        })
        st.success("Outfit guardado en la nube ✅")

# ---- Crear outfit automático ----
st.markdown("### 🤖 Crear outfit automático")
aa1, aa2, aa3 = st.columns(3)
with aa1:
    atmin = st.number_input("Temp. mínima (auto)", value=12, step=1)
with aa2:
    atmax = st.number_input("Temp. máxima (auto)", value=22, step=1)
with aa3:
    aseason = st.selectbox("Estación (auto)", SEASONS, index=1)
aeleg = st.slider("Elegancia (auto)", 1, 5, 3)

def filter_for_weather(df, tmin, tmax, eleg):
    df = df.copy()
    # Frío / calor
    if tmax >= 23:
        df = df[~((df["categoria"].isin(["Pantalón"])) & (df["tipo"] == "Largo"))]
        df = df[~(df["categoria"].isin(["Botas"]))]  # menos botas con calor
    if tmax <= 12:
        df = df[~(df["categoria"].isin(["Short", "Falda", "Sandalias"]))]
    # Elegancia aproximada por estilo
    est_rank = {"Chándal": 1, "Casual": 2, "Formal": 4}
    df["ElegScore"] = df["estilo"].map(est_rank).fillna(2)
    # Formal => evita neón
    if eleg >= 4:
        mask_neon = (df["color1_hex"].apply(is_neon)) | (df["color2_hex"].apply(is_neon))
        df = df[~mask_neon]
    # Tolerancia ±1
    df = df[np.abs(df["ElegScore"] - eleg) <= 1]
    return df

def pick_by_color(top_hex, candidates_hex):
    comp = [h for h in candidates_hex if colors_compatible(top_hex, h)]
    return comp if comp else candidates_hex

def auto_build(df, tmin, tmax, eleg):
    dfw = filter_for_weather(df, tmin, tmax, eleg)
    if dfw.empty:
        return None
    tops    = dfw[dfw["categoria"].isin(["Camiseta", "Camisa", "Sudadera"])]
    bottoms = dfw[dfw["categoria"].isin(["Pantalón", "Short", "Falda"])]
    shoes   = dfw[dfw["categoria"].isin(["Zapatillas", "Botas", "Sandalias"])]
    if tops.empty or bottoms.empty or shoes.empty:
        return None

    # Top: si hace frío, prefiere largo si existe
    top = tops[tops["tipo"] == "Largo"].sample(1) if (tmax <= 16 and not tops[tops["tipo"] == "Largo"].empty) else tops.sample(1)
    top_hex = top.iloc[0]["color1_hex"] or top.iloc[0]["color2_hex"]

    # Bottom compatible
    bpool = bottoms.copy()
    bhex = (bpool["color1_hex"].replace("", pd.NA).fillna(bpool["color2_hex"])).tolist()
    wb = pick_by_color(top_hex, bhex)
    bottom = bpool.iloc[[bhex.index(np.random.choice(wb))]] if wb else bottoms.sample(1)

    # Zapatos por clima
    if tmax >= 25 and not shoes[shoes["categoria"] == "Sandalias"].empty:
        spool = shoes[shoes["categoria"] == "Sandalias"]
    elif tmax <= 12 and not shoes[shoes["categoria"] == "Botas"].empty:
        spool = shoes[shoes["categoria"] == "Botas"]
    else:
        spool = shoes
    shex = (spool["color1_hex"].replace("", pd.NA).fillna(spool["color2_hex"])).tolist()
    ws = pick_by_color(top_hex, shex)
    shoe = spool.iloc[[shex.index(np.random.choice(ws))]] if ws else spool.sample(1)

    return [top.iloc[0]["id"], bottom.iloc[0]["id"], shoe.iloc[0]["id"]]

if st.button("Sugerir outfit"):
    if items_df.empty:
        st.warning("Añade prendas primero.")
    else:
        ids = auto_build(items_df, atmin, atmax, aeleg)
        if not ids:
            st.warning("No pude sugerir con esos filtros. Añade más prendas o relaja condiciones.")
        else:
            st.session_state["last_suggest"] = ids
            st.success("Sugerencia creada 👇")

if "last_suggest" in st.session_state:
    ids = st.session_state["last_suggest"]
    df = items_df.set_index("id")
    st.write("**Propuesta:**")
    for pid in ids:
        if pid in df.index:
            row = df.loc[pid]
            st.write(f"- {row['nombre']} · {row['categoria']} ({row['tipo']}) · {row['estilo']}")
            swatch(row["color1_hex"] or row["color2_hex"])
    aname = st.text_input("Nombre para guardar esta propuesta")
    if st.button("Guardar esta propuesta"):
        if not aname:
            st.error("Pon un nombre al outfit.")
        else:
            db_outfit_insert({
                "name": aname,
                "temp_min": int(atmin),
                "temp_max": int(atmax),
                "seasons": [aseason],
                "elegance": int(aeleg),
                "item_ids": ids
            })
            st.success("Outfit guardado ✅")

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
import re
import pandas as pd
import streamlit as st

APP_TITLE = "Control de tramos torre"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_EXCEL = BASE_DIR / "CODIGOS definitivo.xlsx"
INVENTARIO_CSV = DATA_DIR / "inventario_fisico.csv"
MOVIMIENTOS_CSV = DATA_DIR / "movimientos.csv"
PLAN_CSV = DATA_DIR / "necesidades_montaje.csv"

ESTADOS = [
    "Disponible",
    "Reservado",
    "En tránsito",
    "En obra",
    "Pendiente revisión",
    "No disponible",
]

MOTIVOS = [
    "Alta inicial",
    "Montaje",
    "Desmontaje",
    "Traslado entre campas",
    "Reserva montaje",
    "Revisión OK",
    "Reparación",
    "Baja",
]

COLUMNAS_INVENTARIO = [
    "ID Tramo",
    "Codigo Comercial",
    "Código Estructural",
    "Codigo Plano",
    "Longitud",
    "Peso",
    "Tn",
    "Descripción",
    "Estado",
    "Ubicación actual",
    "Campa base",
    "Obra actual",
    "Fecha último movimiento",
    "Observaciones",
]

COLUMNAS_MOVIMIENTOS = [
    "Fecha",
    "ID Tramo",
    "Código Estructural",
    "Origen",
    "Destino",
    "Motivo",
    "Estado nuevo",
    "Responsable",
    "Obra",
    "Observaciones",
]

COLUMNAS_PLAN = [
    "Obra",
    "Fecha montaje",
    "Campa preferente",
    "Código Estructural",
    "Cantidad necesaria",
    "Observaciones",
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_text(c) for c in df.columns]

    rename = {
        "Codigo Comercial": "Codigo Comercial",
        "Código Comercial": "Codigo Comercial",
        "Codigo Plano": "Codigo Plano",
        "Código Plano": "Codigo Plano",
        "Descripció": "Descripción",
        "Descripcion": "Descripción",
        "Descripción": "Descripción",
    }
    df = df.rename(columns=rename)

    required = ["Codigo Comercial", "Código Estructural", "Codigo Plano", "Longitud", "Peso", "Tn", "Descripción"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en el Excel: {', '.join(missing)}")
        st.stop()

    df = df[required].dropna(how="all")
    for col in ["Codigo Comercial", "Código Estructural", "Codigo Plano", "Descripción"]:
        df[col] = df[col].map(clean_text)
    for col in ["Longitud", "Peso", "Tn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Código Estructural"] != ""].reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_catalog_from_path(path: str) -> pd.DataFrame:
    return normalize_catalog(pd.read_excel(path))


def load_catalog() -> pd.DataFrame:
    st.sidebar.subheader("Catálogo de códigos")
    uploaded = st.sidebar.file_uploader("Subir Excel de códigos", type=["xlsx"])

    if uploaded is not None:
        return normalize_catalog(pd.read_excel(uploaded))

    if DEFAULT_EXCEL.exists():
        return load_catalog_from_path(str(DEFAULT_EXCEL))

    st.warning("Sube el Excel de códigos para empezar.")
    st.stop()


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna("")
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]
    return pd.DataFrame(columns=columns)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def safe_code(text: str) -> str:
    text = text.replace(",", ".")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-")


def next_ids(inventario: pd.DataFrame, codigo_estructural: str, cantidad: int) -> list[str]:
    prefix = safe_code(codigo_estructural)
    existing = inventario["ID Tramo"].astype(str).tolist() if not inventario.empty else []
    nums = []
    for item in existing:
        if item.startswith(prefix + "-"):
            last = item.split("-")[-1]
            if last.isdigit():
                nums.append(int(last))
    start = max(nums, default=0) + 1
    return [f"{prefix}-{i:03d}" for i in range(start, start + cantidad)]


def get_catalog_row(catalogo: pd.DataFrame, codigo_estructural: str, codigo_comercial: str | None = None) -> pd.Series:
    if codigo_comercial:
        rows = catalogo[
            (catalogo["Código Estructural"] == codigo_estructural)
            & (catalogo["Codigo Comercial"].astype(str) == str(codigo_comercial))
        ]
        if not rows.empty:
            return rows.iloc[0]
    rows = catalogo[catalogo["Código Estructural"] == codigo_estructural]
    return rows.iloc[0]


def append_movements(rows: list[dict]) -> None:
    movimientos = read_csv(MOVIMIENTOS_CSV, COLUMNAS_MOVIMIENTOS)
    movimientos = pd.concat([movimientos, pd.DataFrame(rows)], ignore_index=True)
    save_csv(movimientos, MOVIMIENTOS_CSV)


def show_kpis(inventario: pd.DataFrame) -> None:
    total = len(inventario)
    disp = int((inventario["Estado"] == "Disponible").sum()) if total else 0
    obra = int((inventario["Estado"] == "En obra").sum()) if total else 0
    reserv = int((inventario["Estado"] == "Reservado").sum()) if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tramos registrados", total)
    c2.metric("Disponibles", disp)
    c3.metric("En obra", obra)
    c4.metric("Reservados", reserv)


def page_dashboard(catalogo: pd.DataFrame, inventario: pd.DataFrame, movimientos: pd.DataFrame) -> None:
    st.header("Dashboard")
    show_kpis(inventario)

    if inventario.empty:
        st.info("Todavía no hay inventario físico. Crea unidades en la página 'Inventario físico'.")
        return

    st.subheader("Stock por campa y estado")
    resumen = (
        inventario.groupby(["Ubicación actual", "Estado"])
        .size()
        .reset_index(name="Cantidad")
        .sort_values(["Ubicación actual", "Estado"])
    )
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.subheader("Disponibles por tipo y campa")
    disponibles = inventario[inventario["Estado"] == "Disponible"]
    if disponibles.empty:
        st.warning("No hay tramos disponibles.")
    else:
        pivot = pd.pivot_table(
            disponibles,
            index="Código Estructural",
            columns="Ubicación actual",
            values="ID Tramo",
            aggfunc="count",
            fill_value=0,
        ).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)

    if not movimientos.empty:
        st.subheader("Últimos movimientos")
        st.dataframe(movimientos.tail(20).iloc[::-1], use_container_width=True, hide_index=True)


def page_catalogo(catalogo: pd.DataFrame) -> None:
    st.header("Catálogo maestro de códigos")
    st.caption("Esta tabla viene del Excel. Define tipos de tramo, no unidades físicas.")

    busqueda = st.text_input("Buscar por código, plano o descripción")
    df = catalogo.copy()
    if busqueda:
        q = busqueda.lower()
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(q, regex=False).any(), axis=1)
        df = df[mask]

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Descargar catálogo filtrado CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "catalogo_codigos.csv",
        "text/csv",
    )


def page_inventario(catalogo: pd.DataFrame, inventario: pd.DataFrame) -> None:
    st.header("Inventario físico")
    st.caption("Aquí cada fila es un tramo real, con matrícula propia, ubicación y estado.")

    with st.expander("Crear unidades físicas", expanded=True):
        codigos = sorted(catalogo["Código Estructural"].dropna().unique().tolist())
        codigo = st.selectbox("Código estructural", codigos)
        opciones_comerciales = catalogo[catalogo["Código Estructural"] == codigo]["Codigo Comercial"].astype(str).unique().tolist()
        codigo_comercial = st.selectbox("Código comercial", opciones_comerciales)
        c1, c2, c3 = st.columns(3)
        cantidad = c1.number_input("Cantidad de unidades a crear", min_value=1, max_value=200, value=1, step=1)
        ubicacion = c2.text_input("Ubicación actual", value="Campa Lleida")
        campa_base = c3.text_input("Campa base", value=ubicacion)
        observaciones = st.text_input("Observaciones", value="")

        if st.button("Crear unidades", type="primary"):
            row = get_catalog_row(catalogo, codigo, codigo_comercial)
            ids = next_ids(inventario, codigo, int(cantidad))
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_rows = []
            mov_rows = []
            for tramo_id in ids:
                new_rows.append(
                    {
                        "ID Tramo": tramo_id,
                        "Codigo Comercial": row["Codigo Comercial"],
                        "Código Estructural": row["Código Estructural"],
                        "Codigo Plano": row["Codigo Plano"],
                        "Longitud": row["Longitud"],
                        "Peso": row["Peso"],
                        "Tn": row["Tn"],
                        "Descripción": row["Descripción"],
                        "Estado": "Disponible",
                        "Ubicación actual": ubicacion,
                        "Campa base": campa_base,
                        "Obra actual": "",
                        "Fecha último movimiento": now,
                        "Observaciones": observaciones,
                    }
                )
                mov_rows.append(
                    {
                        "Fecha": now,
                        "ID Tramo": tramo_id,
                        "Código Estructural": row["Código Estructural"],
                        "Origen": "",
                        "Destino": ubicacion,
                        "Motivo": "Alta inicial",
                        "Estado nuevo": "Disponible",
                        "Responsable": "",
                        "Obra": "",
                        "Observaciones": observaciones,
                    }
                )
            inventario = pd.concat([inventario, pd.DataFrame(new_rows)], ignore_index=True)
            save_csv(inventario, INVENTARIO_CSV)
            append_movements(mov_rows)
            st.success(f"Creadas {cantidad} unidades: {', '.join(ids)}")
            st.rerun()

    st.subheader("Inventario actual")
    if inventario.empty:
        st.info("No hay tramos registrados todavía.")
        return

    estados = st.multiselect("Filtrar por estado", ESTADOS, default=ESTADOS)
    ubicaciones = sorted([u for u in inventario["Ubicación actual"].unique().tolist() if u])
    ubic_filter = st.multiselect("Filtrar por ubicación", ubicaciones, default=ubicaciones)

    df = inventario[inventario["Estado"].isin(estados)]
    if ubic_filter:
        df = df[df["Ubicación actual"].isin(ubic_filter)]

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Descargar inventario CSV", df.to_csv(index=False).encode("utf-8-sig"), "inventario_fisico.csv", "text/csv")


def page_movimientos(inventario: pd.DataFrame, movimientos: pd.DataFrame) -> None:
    st.header("Registrar movimientos")

    if inventario.empty:
        st.info("Primero crea unidades en inventario físico.")
        return

    ids = sorted(inventario["ID Tramo"].tolist())
    seleccion = st.multiselect("Tramos a mover", ids)
    c1, c2, c3 = st.columns(3)
    destino = c1.text_input("Destino", value="Obra ")
    estado_nuevo = c2.selectbox("Estado nuevo", ESTADOS, index=ESTADOS.index("En obra"))
    motivo = c3.selectbox("Motivo", MOTIVOS, index=MOTIVOS.index("Montaje"))
    obra = st.text_input("Obra asociada", value="")
    responsable = st.text_input("Responsable", value="")
    observaciones = st.text_area("Observaciones", value="")

    if seleccion:
        st.subheader("Tramos seleccionados")
        st.dataframe(inventario[inventario["ID Tramo"].isin(seleccion)], use_container_width=True, hide_index=True)

    if st.button("Guardar movimiento", type="primary"):
        if not seleccion:
            st.error("Selecciona al menos un tramo.")
            return
        if not destino.strip():
            st.error("Indica un destino.")
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        mov_rows = []
        for tramo_id in seleccion:
            idx = inventario.index[inventario["ID Tramo"] == tramo_id][0]
            origen = inventario.loc[idx, "Ubicación actual"]
            codigo = inventario.loc[idx, "Código Estructural"]
            inventario.loc[idx, "Ubicación actual"] = destino.strip()
            inventario.loc[idx, "Estado"] = estado_nuevo
            inventario.loc[idx, "Obra actual"] = obra if estado_nuevo in ["Reservado", "En obra"] else ""
            inventario.loc[idx, "Fecha último movimiento"] = now
            inventario.loc[idx, "Observaciones"] = observaciones
            mov_rows.append(
                {
                    "Fecha": now,
                    "ID Tramo": tramo_id,
                    "Código Estructural": codigo,
                    "Origen": origen,
                    "Destino": destino.strip(),
                    "Motivo": motivo,
                    "Estado nuevo": estado_nuevo,
                    "Responsable": responsable,
                    "Obra": obra,
                    "Observaciones": observaciones,
                }
            )

        save_csv(inventario, INVENTARIO_CSV)
        append_movements(mov_rows)
        st.success("Movimiento guardado.")
        st.rerun()

    st.subheader("Historial")
    st.dataframe(movimientos.iloc[::-1], use_container_width=True, hide_index=True)


def calcular_disponibilidad(inventario: pd.DataFrame, plan: pd.DataFrame, obra: str) -> pd.DataFrame:
    plan_obra = plan[plan["Obra"] == obra].copy()
    if plan_obra.empty:
        return pd.DataFrame()

    rows = []
    for _, need in plan_obra.iterrows():
        codigo = need["Código Estructural"]
        campa = need["Campa preferente"]
        necesarios = int(float(need["Cantidad necesaria"]))
        disponibles = inventario[
            (inventario["Código Estructural"] == codigo)
            & (inventario["Estado"] == "Disponible")
        ]
        disp_campa = disponibles[disponibles["Ubicación actual"] == campa]
        disp_otras = disponibles[disponibles["Ubicación actual"] != campa]
        faltan = max(0, necesarios - len(disp_campa))
        otras_txt = (
            disp_otras.groupby("Ubicación actual").size().reset_index(name="Cantidad").to_dict("records")
            if not disp_otras.empty
            else []
        )
        rows.append(
            {
                "Código Estructural": codigo,
                "Necesarios": necesarios,
                "Campa preferente": campa,
                "Disponibles campa": len(disp_campa),
                "Faltan en campa": faltan,
                "Disponibles otras campas": len(disp_otras),
                "Detalle otras campas": "; ".join([f"{x['Ubicación actual']}: {x['Cantidad']}" for x in otras_txt]),
                "Estado planificación": "OK" if faltan == 0 else "FALTAN",
            }
        )
    return pd.DataFrame(rows)


def page_planificacion(catalogo: pd.DataFrame, inventario: pd.DataFrame, plan: pd.DataFrame) -> None:
    st.header("Planificar montaje")

    with st.expander("Añadir necesidad de montaje", expanded=True):
        c1, c2, c3 = st.columns(3)
        obra = c1.text_input("Obra", value="Obra ")
        fecha_montaje = c2.date_input("Fecha montaje", value=date.today())
        campa_pref = c3.text_input("Campa preferente", value="Campa Lleida")
        codigo = st.selectbox("Código estructural necesario", sorted(catalogo["Código Estructural"].unique().tolist()))
        cantidad = st.number_input("Cantidad necesaria", min_value=1, max_value=200, value=1, step=1)
        obs = st.text_input("Observaciones", value="")

        if st.button("Añadir línea de necesidad", type="primary"):
            new = pd.DataFrame([
                {
                    "Obra": obra.strip(),
                    "Fecha montaje": str(fecha_montaje),
                    "Campa preferente": campa_pref.strip(),
                    "Código Estructural": codigo,
                    "Cantidad necesaria": int(cantidad),
                    "Observaciones": obs,
                }
            ])
            plan = pd.concat([plan, new], ignore_index=True)
            save_csv(plan, PLAN_CSV)
            st.success("Necesidad añadida.")
            st.rerun()

    if plan.empty:
        st.info("Todavía no hay necesidades de montaje.")
        return

    st.subheader("Necesidades registradas")
    edited = st.data_editor(plan, use_container_width=True, hide_index=True, num_rows="dynamic")
    if st.button("Guardar cambios en necesidades"):
        save_csv(edited, PLAN_CSV)
        st.success("Cambios guardados.")
        st.rerun()

    obras = sorted([x for x in edited["Obra"].unique().tolist() if x])
    obra_sel = st.selectbox("Obra a comprobar", obras)
    disponibilidad = calcular_disponibilidad(inventario, edited, obra_sel)

    st.subheader("Comprobación de disponibilidad")
    st.dataframe(disponibilidad, use_container_width=True, hide_index=True)

    if not disponibilidad.empty:
        faltan_total = int(disponibilidad["Faltan en campa"].sum())
        if faltan_total == 0:
            st.success("La campa preferente tiene todos los tramos necesarios disponibles.")
        else:
            st.warning(f"Faltan {faltan_total} tramos en la campa preferente. Revisa otras campas.")

    st.subheader("Reserva automática")
    st.caption("Reserva primero desde la campa preferente. Si faltan, completa desde otras campas.")
    if st.button("Reservar tramos para esta obra"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        mov_rows = []
        reservados = []
        no_reservados = []

        for _, need in edited[edited["Obra"] == obra_sel].iterrows():
            codigo = need["Código Estructural"]
            cantidad = int(float(need["Cantidad necesaria"]))
            campa = need["Campa preferente"]
            disponibles = inventario[
                (inventario["Código Estructural"] == codigo)
                & (inventario["Estado"] == "Disponible")
            ].copy()
            disponibles["prioridad"] = (disponibles["Ubicación actual"] != campa).astype(int)
            disponibles = disponibles.sort_values(["prioridad", "Ubicación actual", "ID Tramo"])
            elegidos = disponibles.head(cantidad)

            if len(elegidos) < cantidad:
                no_reservados.append(f"{codigo}: faltan {cantidad - len(elegidos)}")

            for idx, tramo in elegidos.iterrows():
                inventario.loc[idx, "Estado"] = "Reservado"
                inventario.loc[idx, "Obra actual"] = obra_sel
                inventario.loc[idx, "Fecha último movimiento"] = now
                reservados.append(tramo["ID Tramo"])
                mov_rows.append(
                    {
                        "Fecha": now,
                        "ID Tramo": tramo["ID Tramo"],
                        "Código Estructural": codigo,
                        "Origen": tramo["Ubicación actual"],
                        "Destino": tramo["Ubicación actual"],
                        "Motivo": "Reserva montaje",
                        "Estado nuevo": "Reservado",
                        "Responsable": "",
                        "Obra": obra_sel,
                        "Observaciones": "Reserva automática desde planificación",
                    }
                )

        save_csv(inventario, INVENTARIO_CSV)
        if mov_rows:
            append_movements(mov_rows)
        if reservados:
            st.success(f"Reservados {len(reservados)} tramos para {obra_sel}.")
        if no_reservados:
            st.error("No se ha podido reservar todo: " + "; ".join(no_reservados))
        st.rerun()

    st.subheader("Parte de carga provisional")
    parte = inventario[(inventario["Estado"] == "Reservado") & (inventario["Obra actual"] == obra_sel)]
    if parte.empty:
        st.info("No hay tramos reservados para esta obra.")
    else:
        cols = ["ID Tramo", "Código Estructural", "Codigo Comercial", "Ubicación actual", "Peso", "Tn", "Descripción"]
        st.dataframe(parte[cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar parte de carga CSV",
            parte[cols].to_csv(index=False).encode("utf-8-sig"),
            f"parte_carga_{safe_code(obra_sel)}.csv",
            "text/csv",
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    catalogo = load_catalog()
    inventario = read_csv(INVENTARIO_CSV, COLUMNAS_INVENTARIO)
    movimientos = read_csv(MOVIMIENTOS_CSV, COLUMNAS_MOVIMIENTOS)
    plan = read_csv(PLAN_CSV, COLUMNAS_PLAN)

    pagina = st.sidebar.radio(
        "Menú",
        ["Dashboard", "Catálogo", "Inventario físico", "Movimientos", "Planificación"],
    )

    if pagina == "Dashboard":
        page_dashboard(catalogo, inventario, movimientos)
    elif pagina == "Catálogo":
        page_catalogo(catalogo)
    elif pagina == "Inventario físico":
        page_inventario(catalogo, inventario)
    elif pagina == "Movimientos":
        page_movimientos(inventario, movimientos)
    elif pagina == "Planificación":
        page_planificacion(catalogo, inventario, plan)

    st.sidebar.divider()
    st.sidebar.caption("Archivos de datos: data/inventario_fisico.csv, data/movimientos.csv, data/necesidades_montaje.csv")


if __name__ == "__main__":
    main()

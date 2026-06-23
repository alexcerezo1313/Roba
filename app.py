from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
from io import StringIO
import base64
import gzip
import re

import pandas as pd
import streamlit as st


APP_TITLE = "Control de tramos torre"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

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


# Catálogo integrado desde tu Excel CODIGOS definitivo.xlsx
CATALOGO_B64_GZIP = """
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
"""


def cargar_catalogo_integrado():
    texto_limpio = "".join(CATALOGO_B64_GZIP.split())
    csv_bytes = gzip.decompress(base64.b64decode(texto_limpio))
    csv_text = csv_bytes.decode("utf-8")
    df = pd.read_csv(StringIO(csv_text))

    columnas = [
        "Codigo Comercial",
        "Código Estructural",
        "Codigo Plano",
        "Longitud",
        "Peso",
        "Tn",
        "Descripción",
    ]

    df = df[columnas]

    for col in ["Codigo Comercial", "Código Estructural", "Codigo Plano", "Descripción"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["Longitud", "Peso", "Tn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def read_csv(path, columns):
    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna("")

        for col in columns:
            if col not in df.columns:
                df[col] = ""

        return df[columns]

    return pd.DataFrame(columns=columns)


def save_csv(df, path):
    df.to_csv(path, index=False)


def safe_code(text):
    text = str(text).replace(",", ".")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-")


def next_ids(inventario, codigo_estructural, cantidad):
    prefix = safe_code(codigo_estructural)

    if inventario.empty:
        existing = []
    else:
        existing = inventario["ID Tramo"].astype(str).tolist()

    nums = []

    for item in existing:
        if item.startswith(prefix + "-"):
            last = item.split("-")[-1]
            if last.isdigit():
                nums.append(int(last))

    start = max(nums, default=0) + 1

    return [f"{prefix}-{i:03d}" for i in range(start, start + cantidad)]


def get_catalog_row(catalogo, codigo_estructural, codigo_comercial=None):
    if codigo_comercial:
        rows = catalogo[
            (catalogo["Código Estructural"] == codigo_estructural)
            & (catalogo["Codigo Comercial"].astype(str) == str(codigo_comercial))
        ]

        if not rows.empty:
            return rows.iloc[0]

    rows = catalogo[catalogo["Código Estructural"] == codigo_estructural]
    return rows.iloc[0]


def append_movements(rows):
    movimientos = read_csv(MOVIMIENTOS_CSV, COLUMNAS_MOVIMIENTOS)
    movimientos = pd.concat([movimientos, pd.DataFrame(rows)], ignore_index=True)
    save_csv(movimientos, MOVIMIENTOS_CSV)


def show_kpis(inventario):
    total = len(inventario)

    if total:
        disp = int((inventario["Estado"] == "Disponible").sum())
        obra = int((inventario["Estado"] == "En obra").sum())
        reserv = int((inventario["Estado"] == "Reservado").sum())
    else:
        disp = 0
        obra = 0
        reserv = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tramos registrados", total)
    c2.metric("Disponibles", disp)
    c3.metric("En obra", obra)
    c4.metric("Reservados", reserv)


def page_dashboard(inventario, movimientos):
    st.header("Dashboard")

    show_kpis(inventario)

    if inventario.empty:
        st.info("Todavía no hay inventario físico. Crea unidades en la página 'Inventario físico'.")
        return

    st.subheader("Stock por ubicación y estado")

    resumen = (
        inventario.groupby(["Ubicación actual", "Estado"])
        .size()
        .reset_index(name="Cantidad")
        .sort_values(["Ubicación actual", "Estado"])
    )

    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.subheader("Disponibles por tipo y ubicación")

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


def page_catalogo(catalogo):
    st.header("Catálogo maestro integrado")
    st.caption("Estos datos ya están integrados dentro del código. No necesitas subir el Excel.")

    busqueda = st.text_input("Buscar por código, plano o descripción")

    df = catalogo.copy()

    if busqueda:
        q = busqueda.lower()

        mask = df.astype(str).apply(
            lambda row: row.str.lower().str.contains(q, regex=False).any(),
            axis=1,
        )

        df = df[mask]

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Descargar catálogo CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "catalogo_codigos.csv",
        "text/csv",
    )


def page_inventario(catalogo, inventario):
    st.header("Inventario físico")
    st.caption("Aquí cada fila es un tramo real, con matrícula propia, ubicación y estado.")

    with st.expander("Crear unidades físicas", expanded=True):
        codigos = sorted(catalogo["Código Estructural"].dropna().unique().tolist())

        codigo = st.selectbox("Código estructural", codigos)

        opciones_comerciales = (
            catalogo[catalogo["Código Estructural"] == codigo]["Codigo Comercial"]
            .astype(str)
            .unique()
            .tolist()
        )

        codigo_comercial = st.selectbox("Código comercial", opciones_comerciales)

        info = catalogo[
            (catalogo["Código Estructural"] == codigo)
            & (catalogo["Codigo Comercial"].astype(str) == str(codigo_comercial))
        ]

        st.dataframe(info, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)

        cantidad = c1.number_input(
            "Cantidad de unidades a crear",
            min_value=1,
            max_value=200,
            value=1,
            step=1,
        )

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

    st.download_button(
        "Descargar inventario CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "inventario_fisico.csv",
        "text/csv",
    )


def page_movimientos(inventario, movimientos):
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
        st.dataframe(
            inventario[inventario["ID Tramo"].isin(seleccion)],
            use_container_width=True,
            hide_index=True,
        )

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

            if estado_nuevo in ["Reservado", "En obra"]:
                inventario.loc[idx, "Obra actual"] = obra
            else:
                inventario.loc[idx, "Obra actual"] = ""

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


def calcular_disponibilidad(inventario, plan, obra):
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

        if not disp_otras.empty:
            otras = disp_otras.groupby("Ubicación actual").size().reset_index(name="Cantidad")
            detalle_otras = "; ".join(
                [f"{row['Ubicación actual']}: {row['Cantidad']}" for _, row in otras.iterrows()]
            )
        else:
            detalle_otras = ""

        rows.append(
            {
                "Código Estructural": codigo,
                "Necesarios": necesarios,
                "Campa preferente": campa,
                "Disponibles campa": len(disp_campa),
                "Faltan en campa": faltan,
                "Disponibles otras campas": len(disp_otras),
                "Detalle otras campas": detalle_otras,
                "Estado planificación": "OK" if faltan == 0 else "FALTAN",
            }
        )

    return pd.DataFrame(rows)


def page_planificacion(catalogo, inventario, plan):
    st.header("Planificar montaje")

    with st.expander("Añadir necesidad de montaje", expanded=True):
        c1, c2, c3 = st.columns(3)

        obra = c1.text_input("Obra", value="Obra ")
        fecha_montaje = c2.date_input("Fecha montaje", value=date.today())
        campa_pref = c3.text_input("Campa preferente", value="Campa Lleida")

        codigo = st.selectbox(
            "Código estructural necesario",
            sorted(catalogo["Código Estructural"].unique().tolist()),
        )

        cantidad = st.number_input(
            "Cantidad necesaria",
            min_value=1,
            max_value=200,
            value=1,
            step=1,
        )

        obs = st.text_input("Observaciones", value="")

        if st.button("Añadir línea de necesidad", type="primary"):
            new = pd.DataFrame(
                [
                    {
                        "Obra": obra.strip(),
                        "Fecha montaje": str(fecha_montaje),
                        "Campa preferente": campa_pref.strip(),
                        "Código Estructural": codigo,
                        "Cantidad necesaria": int(cantidad),
                        "Observaciones": obs,
                    }
                ]
            )

            plan = pd.concat([plan, new], ignore_index=True)

            save_csv(plan, PLAN_CSV)

            st.success("Necesidad añadida.")
            st.rerun()

    if plan.empty:
        st.info("Todavía no hay necesidades de montaje.")
        return

    st.subheader("Necesidades registradas")

    edited = st.data_editor(
        plan,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
    )

    if st.button("Guardar cambios en necesidades"):
        save_csv(edited, PLAN_CSV)
        st.success("Cambios guardados.")
        st.rerun()

    obras = sorted([x for x in edited["Obra"].unique().tolist() if x])

    if not obras:
        st.info("No hay obras válidas.")
        return

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

            disponibles = disponibles.sort_values(
                ["prioridad", "Ubicación actual", "ID Tramo"]
            )

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

    parte = inventario[
        (inventario["Estado"] == "Reservado")
        & (inventario["Obra actual"] == obra_sel)
    ]

    if parte.empty:
        st.info("No hay tramos reservados para esta obra.")
    else:
        cols = [
            "ID Tramo",
            "Código Estructural",
            "Codigo Comercial",
            "Ubicación actual",
            "Peso",
            "Tn",
            "Descripción",
        ]

        st.dataframe(parte[cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar parte de carga CSV",
            parte[cols].to_csv(index=False).encode("utf-8-sig"),
            f"parte_carga_{safe_code(obra_sel)}.csv",
            "text/csv",
        )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)

    catalogo = cargar_catalogo_integrado()
    inventario = read_csv(INVENTARIO_CSV, COLUMNAS_INVENTARIO)
    movimientos = read_csv(MOVIMIENTOS_CSV, COLUMNAS_MOVIMIENTOS)
    plan = read_csv(PLAN_CSV, COLUMNAS_PLAN)

    pagina = st.sidebar.radio(
        "Menú",
        [
            "Dashboard",
            "Catálogo",
            "Inventario físico",
            "Movimientos",
            "Planificación",
        ],
    )

    if pagina == "Dashboard":
        page_dashboard(inventario, movimientos)

    elif pagina == "Catálogo":
        page_catalogo(catalogo)

    elif pagina == "Inventario físico":
        page_inventario(catalogo, inventario)

    elif pagina == "Movimientos":
        page_movimientos(inventario, movimientos)

    elif pagina == "Planificación":
        page_planificacion(catalogo, inventario, plan)

    st.sidebar.divider()
    st.sidebar.caption("El catálogo de códigos está integrado dentro de app.py.")


if __name__ == "__main__":
    main()

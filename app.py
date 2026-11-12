"""Predicción de diabetes — aplicación de despliegue del modelo.

Proyecto de aula de Data Mining, Universidad de la Costa (CUC).
Tercer corte: integración y despliegue del modelo con Streamlit.

La aplicación no entrena nada. Carga el pipeline ya ajustado que el cuaderno guardó con
joblib y se limita a pedirle predicciones. Las métricas que muestra las recalcula aplicando
ese mismo pipeline sobre la partición de prueba, de modo que no pueden separarse de las del
cuaderno.
"""

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

RAIZ = Path(__file__).parent
RUTA_MODELO = RAIZ / "modelo" / "pipeline_diabetes.joblib"
RUTA_DATOS = RAIZ / "datos" / "diabetes_preparado.csv"

ENLACE_DATOS = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
INTEGRANTES = ["Santiago Andrés Lozano"]

# Misma escala de grises que las figuras del informe, para que la aplicación y el documento
# se lean como una sola entrega.
GRIS, GRIS_OSCURO, NEGRO = "0.70", "0.40", "0.15"

st.set_page_config(page_title="Predicción de diabetes — CUC",
                   page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- carga
@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_paquete():
    return joblib.load(RUTA_MODELO)


@st.cache_data(show_spinner="Cargando los datos…")
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


def estilo_grafico():
    matplotlib.rcParams.update({
        "figure.dpi": 130, "axes.grid": True, "grid.color": "0.90", "grid.linewidth": 0.7,
        "axes.edgecolor": "0.30", "axes.axisbelow": True, "font.size": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


if not RUTA_MODELO.exists() or not RUTA_DATOS.exists():
    st.error(
        "No se encontraron los archivos del modelo.\n\n"
        f"Se esperaban `{RUTA_MODELO.relative_to(RAIZ)}` y `{RUTA_DATOS.relative_to(RAIZ)}`. "
        "Los genera el cuaderno de la carpeta `notebook/` en su sección 10."
    )
    st.stop()

PAQUETE = cargar_paquete()
PIPELINE = PAQUETE["pipeline"]
UMBRAL = PAQUETE["umbral"]
COLUMNAS = PAQUETE["columnas"]
VARIABLES = PAQUETE["variables"]
METRICAS = PAQUETE["metricas_prueba"]
estilo_grafico()
datos = cargar_datos()


# ----------------------------------------------------------------------- funciones
def validar_entrada(entrada, variables=VARIABLES):
    """Devuelve la lista de errores de una entrada del formulario. Vacía si todo está bien.

    Es la misma función que el cuaderno ejecuta en su sección 11, para que lo que la
    aplicación rechaza quede documentado allí.
    """
    errores = []
    for campo, meta in variables.items():
        valor = entrada.get(campo, None)
        if valor is None:
            if not meta.get("opcional", False):
                errores.append(f"**{meta['etiqueta']}**: es un campo obligatorio.")
            continue
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            errores.append(f"**{meta['etiqueta']}**: se esperaba un número.")
            continue
        if np.isnan(numero):
            errores.append(f"**{meta['etiqueta']}**: se esperaba un número.")
            continue
        if meta["tipo"] == "entero" and numero != int(numero):
            errores.append(f"**{meta['etiqueta']}**: debe ser un número entero.")
        if not (meta["min"] <= numero <= meta["max"]):
            unidad = f" {meta['unidad']}" if meta["unidad"] else ""
            errores.append(
                f"**{meta['etiqueta']}**: {numero:g}{unidad} está fuera del rango admitido "
                f"({meta['min']:g} a {meta['max']:g}{unidad})."
            )
    return errores


def construir_fila(entrada, columnas=COLUMNAS):
    """Convierte los valores del formulario en la fila que espera el pipeline.

    Un campo en None significa que la medición no se tomó: se deja como NaN y el marcador
    de ausencia correspondiente se pone en 1, igual que durante el entrenamiento.
    """
    fila = {c: np.nan for c in columnas}
    for campo, valor in entrada.items():
        fila[campo] = np.nan if valor is None else float(valor)
    fila["insulin_faltante"] = int(pd.isna(fila["insulin"]))
    fila["skin_thickness_faltante"] = int(pd.isna(fila["skin_thickness"]))
    return pd.DataFrame([[fila[c] for c in columnas]], columns=columnas)


def predecir(entrada):
    """Probabilidad de la clase positiva y clase asignada con el umbral del modelo."""
    fila = construir_fila(entrada)
    probabilidad = float(PIPELINE.predict_proba(fila)[0, 1])
    return probabilidad, int(probabilidad >= UMBRAL), fila


# ------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.subheader("Modelo en producción")
    st.write(f"**{PAQUETE['nombre_modelo']}**")
    st.caption(f"Imputación: {PAQUETE['imputacion']}")
    st.markdown(
        f"""
| | |
|---|---|
| Umbral de decisión | **{UMBRAL}** |
| Exhaustividad (clase 1) | **{METRICAS['exhaustividad_1']}** |
| Exactitud | **{METRICAS['exactitud']}** |
| AUC | **{METRICAS['auc']}** |
"""
    )
    st.caption(
        f"Entrenado el {PAQUETE['fecha_entrenamiento']} con semilla {PAQUETE['semilla']}. "
        f"Hiperparámetros: C={PAQUETE['hiperparametros']['C']}, "
        f"regularización {PAQUETE['hiperparametros']['regularizacion']}, "
        f"pesos de clase {PAQUETE['hiperparametros']['class_weight']}."
    )
    version_entrenamiento = PAQUETE["versiones"]["scikit-learn"]
    if version_entrenamiento != sklearn.__version__:
        st.warning(
            f"El modelo se creó con scikit-learn {version_entrenamiento} y aquí está instalada "
            f"la {sklearn.__version__}. Revise `requirements.txt`: con versiones distintas las "
            "predicciones pueden no coincidir con las del cuaderno."
        )
    else:
        st.caption(f"scikit-learn {sklearn.__version__} · coincide con el entrenamiento.")


# ------------------------------------------------------------------------------ cabecera
st.title("Predicción de diabetes a partir de indicadores clínicos")
st.caption(
    "Proyecto de aula · Data Mining · Universidad de la Costa (CUC) · "
    "Departamento de Ciencias de la Computación y Electrónica"
)
st.error(
    "**Prototipo académico. No sustituye una decisión profesional.** Este modelo se entrenó "
    "con 768 registros de una sola población y deja sin detectar a 7 de cada 67 pacientes "
    f"diabéticas del conjunto de prueba. No debe usarse para decidir sobre ninguna persona: "
    "su propósito es demostrar el ciclo completo de un ejercicio de minería de datos."
)

pestanas = st.tabs(["El proyecto", "Los datos", "Predicción", "Rendimiento del modelo"])


# ========================================================================== 1. proyecto
with pestanas[0]:
    izquierda, derecha = st.columns([3, 2], gap="large")

    with izquierda:
        st.subheader("Qué se busca responder")
        st.markdown(
            "> ¿Qué indicadores clínicos y demográficos permiten predecir el diagnóstico de "
            "diabetes en esta población, y qué modelo de clasificación logra la mejor "
            "detección de los casos positivos?"
        )
        st.markdown(
            "La segunda mitad de la pregunta es deliberada. En un problema de tamizaje no "
            "basta con acertar mucho en total: lo que importa es no dejar pasar a las "
            "pacientes que sí tienen la enfermedad. Los dos errores no cuestan lo mismo."
        )
        st.markdown(
            """
| Error | Qué significa | Qué cuesta |
|---|---|---|
| **Falso negativo** | Una paciente con diabetes clasificada como sana | Se va sin confirmación ni tratamiento y la enfermedad avanza |
| **Falso positivo** | Una paciente sana marcada como caso de riesgo | Una consulta de confirmación que resultará negativa |
"""
        )
        st.markdown(
            "Por esa asimetría el modelo **no** se eligió por exactitud, sino por la "
            "exhaustividad de la clase positiva leída junto al F1 de esa misma clase. Es la "
            "razón por la que la aplicación prefiere equivocarse marcando riesgo que dejando "
            "pasar un caso."
        )

        st.subheader("Cómo funciona esta aplicación")
        st.markdown(
            """
1. El cuaderno entrena el pipeline completo —imputación, escalamiento y clasificador— y lo
   guarda con `joblib`.
2. La aplicación carga ese archivo. **No vuelve a entrenar nada** en ningún momento.
3. El formulario valida los rangos y los tipos de dato antes de predecir.
4. Las métricas de la última pestaña se recalculan aplicando el mismo pipeline cargado sobre
   la partición de prueba, así que son exactamente las del cuaderno.
"""
        )

    with derecha:
        st.subheader("Ficha del proyecto")
        st.markdown(
            f"""
| | |
|---|---|
| **Tipo de problema** | Clasificación supervisada binaria |
| **Variable objetivo** | `outcome` — 1 diagnosticada, 0 no |
| **Registros** | 768 (576 entrenamiento / 192 prueba) |
| **Variables de entrada** | 8 originales + 2 marcadores de ausencia |
| **Asignatura** | Data Mining |
| **Docente** | José Escorcia-Gutiérrez, Ph.D. |
"""
        )
        st.markdown("**Integrantes**")
        for persona in INTEGRANTES:
            st.markdown(f"- {persona}")

        st.markdown("**Fuente de los datos**")
        st.markdown(
            f"Pima Indians Diabetes Database, del National Institute of Diabetes and Digestive "
            f"and Kidney Diseases (NIDDK), distribuida por el UCI Machine Learning Repository "
            f"y [Kaggle]({ENLACE_DATOS}).\n\n"
            "Reúne mediciones de mujeres de la comunidad indígena Pima mayores de 21 años, con "
            "un diagnóstico binario de diabetes tipo 2 (Smith et al., 1988)."
        )
        st.info(
            "Los datos corresponden a una sola población. Los resultados **no** son "
            "extrapolables a otros grupos, y en particular no lo son al contexto colombiano.",
            icon="⚠️",
        )


# ============================================================================= 2. datos
with pestanas[1]:
    st.subheader("Tres lecturas del análisis exploratorio")
    st.caption(
        "Las gráficas se calculan sobre los 768 registros con los faltantes declarados como "
        "nulos. El análisis completo está en las secciones 1 a 7 del cuaderno."
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("##### 1. La variable objetivo está desbalanceada")
        conteo = datos["outcome"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        barras = ax.bar(["Sin diabetes (0)", "Con diabetes (1)"], conteo.values,
                        color=[GRIS, GRIS_OSCURO], edgecolor=NEGRO, width=0.5)
        for b, v in zip(barras, conteo.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 8,
                    f"{v}\n({v / len(datos) * 100:.1f} %)", ha="center", va="bottom",
                    fontsize=8.5, color=NEGRO)
        ax.set_ylabel("Número de pacientes")
        ax.set_ylim(0, 620)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            "De 768 pacientes, 500 no tienen diabetes y 268 sí. Un clasificador que "
            "respondiera siempre «sin diabetes» acertaría el **65.1 %** sin haber aprendido "
            "nada: ese es el piso contra el que hay que comparar cualquier resultado, y la "
            "razón por la que la exactitud sola no sirve para elegir un modelo."
        )

    with col_b:
        st.markdown("##### 2. La glucosa es lo que mejor separa a los dos grupos")
        fig, ax = plt.subplots(figsize=(5, 3))
        bordes = np.histogram_bin_edges(datos["glucose"].dropna(), bins=22)
        ax.hist(datos.loc[datos["outcome"] == 0, "glucose"].dropna(), bins=bordes,
                color="0.82", edgecolor=NEGRO, linewidth=0.5, label="Sin diabetes")
        ax.hist(datos.loc[datos["outcome"] == 1, "glucose"].dropna(), bins=bordes,
                color="0.42", edgecolor=NEGRO, linewidth=0.5, alpha=0.85,
                label="Con diabetes")
        ax.set_xlabel("Glucosa en plasma (mg/dL)")
        ax.set_ylabel("Frecuencia")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        medianas = datos.groupby("outcome")["glucose"].median()
        st.markdown(
            f"La mediana pasa de **{medianas[0]:.0f} mg/dL** en las pacientes sin diabetes a "
            f"**{medianas[1]:.0f}** en las diagnosticadas. Los grupos se solapan en la zona "
            "media pero las colas son muy distintas, y es justo en ese solapamiento donde el "
            "modelo comete sus errores."
        )

    st.divider()
    st.markdown("##### 3. La proporción de diagnósticos crece con la glucosa y con la edad")
    rangos = datos.copy()
    rangos["rango_glucosa"] = pd.cut(rangos["glucose"], bins=[0, 100, 125, 150, 200],
                                     labels=["≤ 100", "101–125", "126–150", "> 150"])
    rangos["rango_edad"] = pd.cut(rangos["age"], bins=[20, 30, 40, 50, 100],
                                  labels=["21–30", "31–40", "41–50", "> 50"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))
    for ax, columna, titulo in [(axes[0], "rango_glucosa", "Nivel de glucosa (mg/dL)"),
                                (axes[1], "rango_edad", "Grupo de edad (años)")]:
        tasa = rangos.groupby(columna, observed=True)["outcome"].agg(["mean", "count"])
        barras = ax.bar(tasa.index.astype(str), tasa["mean"] * 100, color=GRIS,
                        edgecolor=NEGRO, width=0.55)
        for b, (m, n) in zip(barras, tasa.values):
            ax.text(b.get_x() + b.get_width() / 2, m * 100 + 1.5,
                    f"{m * 100:.1f} %\n(n={int(n)})", ha="center", va="bottom",
                    fontsize=8, color=NEGRO)
        ax.set_xlabel(titulo)
        ax.set_ylabel("% con diagnóstico positivo")
        ax.set_ylim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        "El gradiente de la glucosa sigue los umbrales clínicos de referencia: del **8.6 %** "
        "de diagnósticos positivos hasta 100 mg/dL al **75.0 %** por encima de 150. El de la "
        "edad es más suave y no monótono —baja en las mayores de 50—, lo que probablemente se "
        "deba a que ese tramo solo reúne 81 pacientes. Por eso cada barra reporta su `n`."
    )

    with st.expander("Ver el conjunto de datos preparado"):
        st.caption(
            "Los valores en blanco son los faltantes declarados: los ceros clínicamente "
            "imposibles de cinco variables se convirtieron en nulos, y la imputación por "
            "mediana vive dentro del pipeline, no en los datos."
        )
        st.dataframe(datos, width="stretch", height=320)


# ======================================================================== 3. predicción
with pestanas[2]:
    DEFECTOS = {c: VARIABLES[c]["defecto"] for c in VARIABLES}
    OPCIONALES = [c for c in VARIABLES if VARIABLES[c].get("opcional")]

    for campo, defecto in DEFECTOS.items():
        st.session_state.setdefault(f"v_{campo}", defecto)
    for campo in OPCIONALES:
        st.session_state.setdefault(f"m_{campo}", True)
    st.session_state.setdefault("nota_caso", None)

    def aplicar_caso(valores, nota=None):
        """Carga un conjunto de valores en el formulario. Se ejecuta antes del rerun."""
        for campo, valor in valores.items():
            if campo in OPCIONALES:
                st.session_state[f"m_{campo}"] = valor is not None
                if valor is not None:
                    st.session_state[f"v_{campo}"] = valor
            else:
                st.session_state[f"v_{campo}"] = valor
        st.session_state["nota_caso"] = nota

    st.subheader("Casos de prueba")
    st.caption(
        "La guía de la actividad pide comprobar tres situaciones. Cada botón carga una de "
        "ellas en el formulario."
    )
    b1, b2, b3 = st.columns(3)
    caso_prueba = PAQUETE["casos_verificacion"][-1]

    b1.button(
        "① Caso normal", width="stretch",
        on_click=aplicar_caso, args=(DEFECTOS, "Valores por defecto, dentro de rango."),
        help="Valores centrales y plausibles para todas las variables.",
    )
    b2.button(
        "② Entrada inválida", width="stretch",
        on_click=aplicar_caso,
        args=(dict(DEFECTOS, glucose=12.0, age=8, bmi=5.0),
              "Entrada deliberadamente inválida: la validación debe rechazarla."),
        help="Glucosa, edad e IMC fuera de los rangos admitidos.",
    )
    b3.button(
        "③ Registro conocido del conjunto de prueba", width="stretch",
        on_click=aplicar_caso,
        args=(caso_prueba["entrada"],
              f"Registro {caso_prueba['indice_original']} del conjunto de prueba. "
              f"El cuaderno le asigna una probabilidad de "
              f"{caso_prueba['probabilidad']:.6f} y un diagnóstico real de "
              f"{caso_prueba['diagnostico_real']}."),
        help="Un registro real, para comprobar que la aplicación coincide con el cuaderno.",
    )

    if st.session_state["nota_caso"]:
        st.info(st.session_state["nota_caso"], icon="📋")

    st.divider()
    st.subheader("Datos de la paciente")

    columnas_form = st.columns(2, gap="large")
    entrada = {}
    campos_obligatorios = [c for c in VARIABLES if c not in OPCIONALES]

    with columnas_form[0]:
        st.markdown("**Mediciones obligatorias**")
        for campo in campos_obligatorios:
            meta = VARIABLES[campo]
            etiqueta = meta["etiqueta"] + (f" ({meta['unidad']})" if meta["unidad"] else "")
            if meta["tipo"] == "entero":
                valor = st.number_input(etiqueta, step=1, key=f"v_{campo}",
                                        help=meta["ayuda"])
            else:
                valor = st.number_input(etiqueta, step=float(meta["paso"]),
                                        format="%.2f", key=f"v_{campo}",
                                        help=meta["ayuda"])
            st.caption(f"Rango admitido: {meta['min']:g} a {meta['max']:g}"
                       + (f" {meta['unidad']}" if meta["unidad"] else ""))
            entrada[campo] = valor

    with columnas_form[1]:
        st.markdown("**Exámenes de laboratorio**")
        st.caption(
            "El modelo fue entrenado contemplando que estos dos exámenes falten, porque en el "
            "conjunto original faltan en el 48.70 % y el 29.56 % de los registros. Si no se "
            "tomaron, desmarque la casilla: la aplicación activa el marcador de ausencia "
            "correspondiente, igual que en el entrenamiento."
        )
        for campo in OPCIONALES:
            meta = VARIABLES[campo]
            etiqueta = meta["etiqueta"] + (f" ({meta['unidad']})" if meta["unidad"] else "")
            tomada = st.checkbox(f"Se tomó: {meta['etiqueta'].lower()}", key=f"m_{campo}")
            valor = st.number_input(etiqueta, step=float(meta["paso"]), format="%.2f",
                                    key=f"v_{campo}", disabled=not tomada,
                                    help=meta["ayuda"])
            if tomada:
                st.caption(f"Rango admitido: {meta['min']:g} a {meta['max']:g} {meta['unidad']}")
                entrada[campo] = valor
            else:
                st.caption("Se enviará como dato ausente.")
                entrada[campo] = None

    st.divider()
    errores = validar_entrada(entrada)

    if errores:
        st.error("**La entrada no es válida. No se calcula ninguna predicción.**")
        for error in errores:
            st.markdown(f"- {error}")
        st.caption(
            "Los rangos admitidos combinan el recorrido observado en el conjunto de datos con "
            "lo clínicamente plausible. Fuera de ellos el modelo extrapolaría, y una "
            "extrapolación no es una predicción."
        )
    else:
        probabilidad, clase, fila = predecir(entrada)

        resultado, detalle = st.columns([2, 3], gap="large")
        with resultado:
            if clase == 1:
                st.markdown("### Riesgo alto")
                st.markdown(
                    f"La probabilidad estimada, **{probabilidad * 100:.1f} %**, alcanza el "
                    f"umbral de decisión del modelo ({UMBRAL * 100:.0f} %). En un tamizaje "
                    "este resultado significa **enviar a confirmación diagnóstica**, no "
                    "diagnosticar."
                )
            else:
                st.markdown("### Riesgo bajo")
                st.markdown(
                    f"La probabilidad estimada, **{probabilidad * 100:.1f} %**, queda por "
                    f"debajo del umbral de decisión ({UMBRAL * 100:.0f} %). Conviene recordar "
                    "que el modelo deja sin detectar a 7 de cada 67 pacientes diabéticas, así "
                    "que un riesgo bajo no descarta la enfermedad."
                )
            st.metric("Probabilidad de diagnóstico positivo", f"{probabilidad * 100:.1f} %")

        with detalle:
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh([0], [probabilidad * 100], color=GRIS_OSCURO, edgecolor=NEGRO, height=0.45)
            ax.axvline(UMBRAL * 100, color=NEGRO, linestyle="-.", linewidth=1.4)
            ax.annotate(f"umbral {UMBRAL * 100:.0f} %", xy=(UMBRAL * 100, 0.42),
                        xytext=(UMBRAL * 100 + 1.5, 0.38), fontsize=8, color=NEGRO)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.4, 0.6)
            ax.set_yticks([])
            ax.set_xlabel("Probabilidad estimada de diagnóstico positivo (%)")
            ax.grid(axis="y", visible=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"El umbral no es el 50 % por defecto, sino **{UMBRAL}**. Se eligió por "
                "validación cruzada sobre el conjunto de entrenamiento, maximizando el F2 de "
                "la clase positiva con la exactitud por encima de 0.70. El conjunto de prueba "
                "no participó en esa decisión."
            )

        with st.expander("Ver la fila exacta que recibió el modelo"):
            st.caption(
                "Es la fila que entra al pipeline, en el orden en que fue entrenado. Los dos "
                "marcadores de ausencia se derivan de las casillas de los exámenes."
            )
            st.dataframe(fila.T.rename(columns={0: "valor"}), width="stretch")
            st.caption(
                f"Pasos del pipeline: "
                f"{' → '.join(nombre for nombre, _ in PIPELINE.steps)}. "
                "La imputación por mediana y el escalamiento se aplican con los parámetros "
                "aprendidos únicamente de los 576 registros de entrenamiento."
            )


# ======================================================================== 4. rendimiento
with pestanas[3]:
    prueba = datos[datos["particion"] == "prueba"]
    X_prueba = prueba[COLUMNAS]
    y_prueba = prueba["outcome"].astype(int)
    probabilidades = PIPELINE.predict_proba(X_prueba)[:, 1]
    predicciones = (probabilidades >= UMBRAL).astype(int)

    from sklearn.metrics import (accuracy_score, confusion_matrix, fbeta_score, f1_score,
                                 precision_score, recall_score, roc_auc_score, roc_curve)

    st.subheader(f"Desempeño sobre el conjunto de prueba ({len(y_prueba)} registros)")
    st.caption(
        f"De las {len(y_prueba)} pacientes de prueba, {int(y_prueba.sum())} tienen diagnóstico "
        "positivo. Las métricas se calculan aplicando el pipeline cargado sobre esos registros, "
        "que el modelo nunca vio durante el entrenamiento ni durante la optimización."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exhaustividad (clase 1)", f"{recall_score(y_prueba, predicciones):.3f}",
              delta=f"{recall_score(y_prueba, predicciones) - PAQUETE['metricas_segundo_corte']['exhaustividad_1']:+.3f} vs. segundo corte")
    m2.metric("Precisión (clase 1)", f"{precision_score(y_prueba, predicciones):.3f}",
              delta=f"{precision_score(y_prueba, predicciones) - PAQUETE['metricas_segundo_corte']['precision_1']:+.3f}")
    m3.metric("F1 (clase 1)", f"{f1_score(y_prueba, predicciones):.3f}",
              delta=f"{f1_score(y_prueba, predicciones) - PAQUETE['metricas_segundo_corte']['f1_1']:+.3f}")
    m4.metric("Exactitud", f"{accuracy_score(y_prueba, predicciones):.3f}",
              delta=f"{accuracy_score(y_prueba, predicciones) - PAQUETE['metricas_segundo_corte']['exactitud']:+.3f}")
    st.caption(
        "La variación se mide frente al modelo del segundo corte. La exhaustividad sube y la "
        "precisión baja: es el intercambio que el proyecto eligió a propósito."
    )

    izq, der = st.columns([3, 4], gap="large")

    with izq:
        st.markdown("##### Matriz de confusión")
        cm = confusion_matrix(y_prueba, predicciones)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max() * 1.25)
        nombres = [["Verdaderos\nnegativos", "Falsos\npositivos"],
                   ["Falsos\nnegativos", "Verdaderos\npositivos"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{nombres[i][j]}", ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > cm.max() * 0.7 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predice sana", "Predice diabetes"], fontsize=8.5)
        ax.set_yticklabels(["Sin diabetes", "Con diabetes"], fontsize=8.5)
        ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        vn, fp, fn, vp = cm.ravel()
        st.markdown(
            f"El modelo detecta **{vp} de las {int(y_prueba.sum())}** pacientes diabéticas y "
            f"deja pasar **{fn}**. A cambio envía a confirmación a **{fp}** de las {vn + fp} "
            "pacientes sanas. La celda de los falsos negativos es la que importa: es la que "
            "representa a una persona que se va sin tratamiento."
        )

    with der:
        st.markdown("##### Curva ROC")
        fpr, tpr, _ = roc_curve(y_prueba, probabilidades)
        auc = roc_auc_score(y_prueba, probabilidades)
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.plot(fpr, tpr, "-", color=NEGRO, linewidth=1.8, label=f"Modelo (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ":", color="0.55", linewidth=1.2,
                label="Clasificador aleatorio")
        punto = (1 - recall_score(y_prueba, predicciones, pos_label=0),
                 recall_score(y_prueba, predicciones))
        ax.plot(*punto, "o", color=NEGRO, markersize=6, markerfacecolor="white")
        ax.annotate(f"umbral {UMBRAL}", xy=punto, xytext=(punto[0] + 0.05, punto[1] - 0.12),
                    fontsize=8, color=NEGRO)
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            f"El AUC de **{auc:.3f}** resume la capacidad de separación del modelo en todos "
            "los umbrales posibles. El punto marcado es dónde opera la aplicación: el umbral "
            "no cambia la curva, solo elige un lugar sobre ella."
        )

    st.divider()
    st.markdown("##### Antes y después de la optimización del tercer corte")
    segundo = PAQUETE["metricas_segundo_corte"]
    comparacion = pd.DataFrame({
        "Segundo corte": [segundo["exactitud"], segundo["precision_1"],
                          segundo["exhaustividad_1"], segundo["f1_1"], segundo["auc"],
                          segundo["matriz_confusion"][1][1], segundo["matriz_confusion"][1][0],
                          segundo["matriz_confusion"][0][1]],
        "Tercer corte": [round(accuracy_score(y_prueba, predicciones), 3),
                         round(precision_score(y_prueba, predicciones), 3),
                         round(recall_score(y_prueba, predicciones), 3),
                         round(f1_score(y_prueba, predicciones), 3),
                         round(auc, 3), int(vp), int(fn), int(fp)],
    }, index=["Exactitud", "Precisión (clase 1)", "Exhaustividad (clase 1)", "F1 (clase 1)",
              "AUC", "Pacientes detectadas", "Falsos negativos", "Falsos positivos"])
    comparacion["Diferencia"] = (comparacion["Tercer corte"]
                                 - comparacion["Segundo corte"]).round(3)
    st.dataframe(comparacion, width="stretch")
    st.markdown(
        "La optimización del tercer corte —búsqueda de hiperparámetros y revisión del umbral— "
        "cambió la configuración del modelo, no su familia. Sigue siendo una regresión "
        "logística, ahora con regularización fuerte y una ponderación de clases de 3 a 1 a "
        "favor de la clase positiva. El resultado son **14 pacientes diabéticas más "
        "detectadas** a cambio de **16 consultas de confirmación adicionales**."
    )

    st.divider()
    st.markdown("##### Limitaciones del modelo")
    st.markdown(
        f"""
- **Deja pasar casos.** {fn} de las {int(y_prueba.sum())} pacientes diabéticas del conjunto de
  prueba quedan sin detectar. Los errores se concentran en la zona de glucosa intermedia, donde
  las dos poblaciones se solapan.
- **Marca pacientes sanas.** {fp} de {vn + fp} pacientes sin diabetes salen como riesgo alto. Es
  el precio deliberado de priorizar la detección, pero significa que una proporción alta de las
  confirmaciones resultará negativa.
- **El conjunto es pequeño.** 768 registros, de los cuales 192 son de prueba. El movimiento de
  dos o tres pacientes desplaza las métricas en varios puntos.
- **La población es específica.** Mujeres mayores de 21 años de una sola comunidad indígena. Los
  resultados no generalizan a otros grupos ni al contexto colombiano.
- **Dos variables están muy incompletas.** Falta casi la mitad de los valores de insulina y cerca
  de un tercio de los del pliegue cutáneo, así que lo que el modelo aprende de ellas descansa en
  buena parte sobre valores imputados.
- **Son asociaciones, no causas.** El modelo encuentra patrones en este conjunto de datos; no
  demuestra que ninguna variable cause la enfermedad.
- **La optimización se hizo sobre 576 registros.** Es un conjunto pequeño para ajustar
  hiperparámetros, de modo que parte de la mejora podría no sostenerse sobre datos nuevos.
"""
    )
    st.error(
        "**Por todo lo anterior, esta aplicación es un prototipo académico y no sustituye una "
        "decisión profesional.** Ningún resultado que muestre debe usarse para diagnosticar, "
        "descartar o priorizar la atención de una persona real."
    )

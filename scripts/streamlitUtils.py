import streamlit as st
import plotly_express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def visualise_classification(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "True Label", "y": "Predicted Label"},
        title="Predicted vs True Labels"
    )
    st.plotly_chart(fig, use_container_width=True)

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=[f"Pred {i}" for i in range(cm.shape[1])],
        y=[f"True {i}" for i in range(cm.shape[0])],
        colorscale="Blues"
    )
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

def visualise_regression(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title="Predicted vs Actual"
    )
    fig.add_shape(
        type="line",
        x0=y_test.min(),
        y0=y_test.min(),
        x1=y_test.max(),
        y1=y_test.max(),
        line=dict(dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)

def visualise_clustering(pipeline, data):
    X_transformed = pipeline[:-1].transform(data.X)
    labels = pipeline[-1].labels_

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_transformed)

    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=labels.astype(str),
        title="Clustering Visualization (PCA)",
        labels={"x": "PC1", "y": "PC2"}
    )
    st.plotly_chart(fig, use_container_width=True)

def render_model_param(spec, name):
    default = spec["default"]

    if "options" in spec:
        return st.selectbox(
            name,
            spec["options"],
            index=spec["options"].index(default) if default in spec["options"] else 0,
        )

    if isinstance(default, bool):
        return st.checkbox(name, value=default)

    # Integer
    if isinstance(default, int):
        min_v = spec.get("min")
        max_v = spec.get("max")

        min_v = int(min_v) if min_v is not None else None
        max_v = int(max_v) if max_v is not None else None

        if (min_v is not None and default < min_v) or \
           (max_v is not None and default > max_v):
            min_v, max_v = None, None

        return st.number_input(
            name,
            value=default,
            min_value=min_v,
            max_value=max_v,
            step=1,
        )

    # Float
    if isinstance(default, float):
        min_v = spec.get("min")
        max_v = spec.get("max")

        min_v = float(min_v) if min_v is not None else None
        max_v = float(max_v) if max_v is not None else None

        # 🚨 Guard: only apply bounds if default is valid
        if (min_v is not None and default < min_v) or \
           (max_v is not None and default > max_v):
            min_v, max_v = None, None

        return st.number_input(
            name,
            value=default,
            min_value=min_v,
            max_value=max_v,
        )

    if isinstance(default, str):
        return st.text_input(name, value=default)

    if default is None:
        return st.text_input(name, value="")

    st.text(f"{name}: {default} ({spec['type']})")
    return default

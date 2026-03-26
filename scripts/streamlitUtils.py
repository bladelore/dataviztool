import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def visualise_classification(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    labels = sorted(set(y_test))

    # --- Scoring metrics ---
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")

    st.subheader("Per-Class Metrics")
    st.dataframe(report_df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{report['accuracy']:.4f}")
    with col2:
        st.metric("Macro F1", f"{report['macro avg']['f1-score']:.4f}")
    with col3:
        st.metric("Weighted F1", f"{report['weighted avg']['f1-score']:.4f}")

    # --- Confusion matrix ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- Predicted vs True scatter ---
    st.subheader("Predicted vs True Labels")
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "True Label", "y": "Predicted Label"},
        color=(np.array(y_test) == np.array(y_pred)).astype(str),
        color_discrete_map={"True": "green", "False": "red"},
        title="Predicted vs True (green = correct)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Per-class accuracy bar chart ---
    st.subheader("Per-Class Accuracy")
    class_acc = {
        str(l): np.mean(np.array(y_pred)[np.array(y_test) == l] == l)
        for l in labels
    }
    fig_bar = px.bar(
        x=list(class_acc.keys()),
        y=list(class_acc.values()),
        labels={"x": "Class", "y": "Accuracy"},
        title="Accuracy by Class",
        text_auto=".2f"
    )
    fig_bar.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)


def visualise_regression(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_test = np.array(y_test).ravel()
    y_pred = np.array(y_pred).ravel()
    residuals = y_test - y_pred

    # --- Scoring metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
    with col2:
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
    with col3:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    with col4:
        st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.4f}")

    # --- Predicted vs Actual ---
    st.subheader("Predicted vs Actual")
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title="Predicted vs Actual",
        opacity=0.6
    )
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(dash="dash", color="red")
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Residuals vs Predicted ---
    st.subheader("Residuals vs Predicted")
    fig_res = px.scatter(
        x=y_pred,
        y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        title="Residuals vs Predicted",
        opacity=0.6
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_res, use_container_width=True)

    # --- Residual distribution ---
    st.subheader("Residual Distribution")
    fig_hist = px.histogram(
        x=residuals,
        nbins=40,
        labels={"x": "Residual", "y": "Count"},
        title="Distribution of Residuals"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Actual vs index (sorted) ---
    st.subheader("Actual vs Predicted (sorted by actual)")
    sorted_idx = np.argsort(y_test)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(y=y_test[sorted_idx], mode="lines", name="Actual"))
    fig_line.add_trace(go.Scatter(y=y_pred[sorted_idx], mode="lines", name="Predicted", line=dict(dash="dash")))
    fig_line.update_layout(
        xaxis_title="Sample (sorted)",
        yaxis_title="Value",
        title="Actual vs Predicted Values"
    )
    st.plotly_chart(fig_line, use_container_width=True)


def visualise_clustering(pipeline, data):
    X_transformed = pipeline[:-1].transform(data.X)
    labels = pipeline[-1].labels_

    # --- Cluster size distribution ---
    unique, counts = np.unique(labels, return_counts=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Clusters", len(unique[unique != -1]))
    with col2:
        noise = int(np.sum(labels == -1))
        st.metric("Noise Points (label -1)", noise)

    st.subheader("Cluster Sizes")
    fig_bar = px.bar(
        x=[str(u) for u in unique],
        y=counts,
        labels={"x": "Cluster", "y": "Count"},
        title="Samples per Cluster",
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- PCA 2D scatter ---
    st.subheader("Cluster Visualisation (PCA 2D)")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_transformed)
    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=labels.astype(str),
        title=f"PCA 2D — explained variance: {pca.explained_variance_ratio_.sum():.1%}",
        labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"},
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- PCA 3D scatter ---
    if X_transformed.shape[1] >= 3:
        st.subheader("Cluster Visualisation (PCA 3D)")
        pca3 = PCA(n_components=3)
        X_3d = pca3.fit_transform(X_transformed)
        fig3 = px.scatter_3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            color=labels.astype(str),
            title=f"PCA 3D — explained variance: {pca3.explained_variance_ratio_.sum():.1%}",
            labels={
                "x": f"PC1 ({pca3.explained_variance_ratio_[0]:.1%})",
                "y": f"PC2 ({pca3.explained_variance_ratio_[1]:.1%})",
                "z": f"PC3 ({pca3.explained_variance_ratio_[2]:.1%})",
            },
            opacity=0.7
        )
        st.plotly_chart(fig3, use_container_width=True)


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
    if isinstance(default, int):
        min_v = spec.get("min")
        max_v = spec.get("max")
        min_v = int(min_v) if min_v is not None else None
        max_v = int(max_v) if max_v is not None else None
        if (min_v is not None and default < min_v) or \
           (max_v is not None and default > max_v):
            min_v, max_v = None, None
        return st.number_input(name, value=default, min_value=min_v, max_value=max_v, step=1)
    if isinstance(default, float):
        min_v = spec.get("min")
        max_v = spec.get("max")
        min_v = float(min_v) if min_v is not None else None
        max_v = float(max_v) if max_v is not None else None
        if (min_v is not None and default < min_v) or \
           (max_v is not None and default > max_v):
            min_v, max_v = None, None
        return st.number_input(name, value=default, min_value=min_v, max_value=max_v)
    if isinstance(default, str):
        return st.text_input(name, value=default)
    if default is None:
        return st.text_input(name, value="")
    st.text(f"{name}: {default} ({spec['type']})")
    return default
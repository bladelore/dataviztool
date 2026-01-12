import streamlit as st
from scripts.simpleModelLoader import ModelSelector
from scripts.dataLoader import UciDataLoader
import plotly_express as px
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA

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

        # 🚨 Guard: only apply bounds if default is valid
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

VISUALISERS = {
    "classification": visualise_classification,
    "regression": visualise_regression,
    "clustering": visualise_clustering,
}

loader = UciDataLoader()
datasetOptions = loader.datasets

# Page config
st.set_page_config(page_title="UCI ML Model Visualizer", layout="wide")

# Initialize loader
if 'loader' not in st.session_state:
    st.session_state.loader = UciDataLoader()

loader = st.session_state.loader
datasetOptions = loader.datasets

st.title("🤖 UCI ML Model Visualizer")

# Dataset selection
option = st.selectbox(
    "Select UCI Dataset",
    options=list(datasetOptions.keys()),
    index=None,
    placeholder="Choose a dataset...",
)

# Load and display data
if option:

    with st.spinner(f"Loading {option}..."):
        data = loader.load(option)
    
    # Display metadata
    st.subheader(f"📊 Dataset: {option}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", data.features.shape[0])
    with col2:
        st.metric("Features", data.features.shape[1])
    with col3:
        st.metric("Tasks", ", ".join(data.tasks))
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Features", "Targets", "Raw Data", "Model Training"])
    
    with tab1:
        st.subheader("Features")
        st.dataframe(data.features, width='stretch')
        
        # Feature statistics
        with st.expander("Feature Statistics"):
            st.write(data.features.describe())
    
    with tab2:
        st.subheader("Targets")
        fig = px.histogram(x=data.y, title="Target Distribution")
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader("Complete Dataset")
        st.dataframe(data.data, width='stretch')
    
    with tab4:
        st.subheader("Model Training")

        # Initialize ModelSelector
        selector = ModelSelector(data)

        # Determine available models based on task type
        task_types = ['regression', 'classification', 'clustering']
        available_models = []

        for task in task_types:
            if task in data.tasks:
                available_models.extend(selector.list_models_by_type(task))

        # Model selection dropdown
        model_name = st.selectbox(
            "Select Model",
            options=available_models,
            index=0
        )

        # Show default parameters
        st.subheader("Model Parameters")
        st.write("Edit parameters if needed:")

        param_specs = selector.get_model_param_specs(model_name)

        user_params = {}
        for name, spec in param_specs.items():
            user_params[name] = render_model_param(spec, name)

        # Train/Test split ratio slider
        test_size = st.slider(
            "Test Set Ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Fraction of data to use as test set"
        )

        # Train button
        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training {model_name}..."):
                try:
                    # Train model
                    pipeline = selector.fit(model_name)

                    X_train, X_test, y_train, y_test = train_test_split(
                        data.X, data.y, test_size=test_size, random_state=42
                    )

                    st.success("✅ Model trained successfully!")

                    model_type = selector.get_model_type(model_name)

                    # Score
                    if model_type != "clustering":
                        score = pipeline.score(X_test, y_test)
                        st.metric("Model Score (R² / Accuracy)", f"{score:.4f}")
                    else:
                        labels = pipeline[-1].labels_
                        st.metric("Number of Clusters", len(set(labels)))

                    # Visualisation
                    st.subheader("Model Visualisation")

                    visualiser = VISUALISERS.get(model_type)

                    if model_type == "clustering":
                        visualiser(pipeline, data)
                    else:
                        visualiser(pipeline, X_test, y_test)

                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")


else:
    st.info("👈 Select a dataset to get started")
    
import streamlit as st
import polars as pl
from scripts.simpleModelLoader import ModelSelector
from scripts.dataLoader import UciDataLoader
from scripts.streamlitUtils import render_model_param, visualise_classification, visualise_clustering, visualise_regression
import plotly.express as px
from sklearn.model_selection import train_test_split

VISUALISERS = {
    "classification": visualise_classification,
    "regression": visualise_regression,
    "clustering": visualise_clustering,
}

def doTraining(model_name):
    try:
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

st.set_page_config(page_title="UCI ML Model Explorer", layout="wide")

# Initialize loader
if 'loader' not in st.session_state:
    st.session_state.loader = UciDataLoader()

loader = st.session_state.loader
datasetOptions = loader.datasets

st.title("UCI ML Model Visualizer")

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
    
    st.subheader(f"Dataset: {option}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", data.features.shape[0])
    with col2:
        st.metric("Features", data.features.shape[1])
    with col3:
        st.metric("Tasks", ", ".join(data.tasks))
    
    has_targets = data.targets is not None and len(data.targets.columns) > 0

    tab1, tab2 = st.tabs(["Features/Targets", "Model Training"])
    
    with tab1:
        st.subheader("Data Summary")
        st.write(data.info)

        st.subheader("Variable info")
        st.dataframe(data.variables, width='content')

        st.subheader("Data")
        st.dataframe(data.data, width='content')

        st.subheader("Feature Statistics")
        st.write(data.features.describe())
                
        st.subheader("Targets")

        all_cols_df = pl.concat([data.features, data.targets], how="horizontal").to_pandas()

        numeric_cols = all_cols_df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            cols = st.columns(min(len(numeric_cols), 3))
            for i, col_name in enumerate(numeric_cols):
                with cols[i % 3]:
                    fig = px.box(all_cols_df, y=col_name, title=col_name)
                    st.plotly_chart(fig, width='content')
        else:
            st.info("No numeric columns to display.")

    with tab2:
        st.subheader("Model Training")

        selector = ModelSelector(data)

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

        if has_targets:
            if data.targets.width > 1:
                selected_target = st.selectbox(
                    "Select Target Column",
                    options=data.targets.columns
                )
            else:
                selected_target = data.targets.columns[0]
        else:
            selected_target = None

        st.subheader("Model Parameters")
        st.write("Edit parameters if needed:")

        param_specs = selector.get_model_param_specs(model_name)

        user_params = {}
        for name, spec in param_specs.items():
            user_params[name] = render_model_param(spec, name)

        test_size = st.slider(
            "Test Set Ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Fraction of data to use as test set"
        )

        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training {model_name}..."):
                doTraining(model_name)

else:
    st.info("Select a dataset to get started")
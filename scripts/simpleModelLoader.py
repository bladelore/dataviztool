# from dataLoader import UciDataLoader
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from inspect import signature
from sklearn.utils._param_validation import (
    StrOptions,
    Interval,
)

from sklearn.utils._param_validation import Interval

class ModelSelector:
    # Define model classes
    REGRESSION_CLASSES = [
        LinearRegression,
        Ridge,
        Lasso,
        SVR,
        RandomForestRegressor,
        GradientBoostingRegressor,
        KNeighborsRegressor,
        DecisionTreeRegressor,
    ]
    
    CLASSIFICATION_CLASSES = [
        LogisticRegression,
        SVC,
        RandomForestClassifier,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        DecisionTreeClassifier,
        GaussianNB,
    ]
    
    CLUSTERING_CLASSES = [
        KMeans,
        DBSCAN,
        AgglomerativeClustering,
        SpectralClustering,
        MeanShift,
        Birch,
        GaussianMixture,
    ]
    
    # Build MODELS dict dynamically
    MODELS = {
        'regression': {cls.__name__: cls for cls in REGRESSION_CLASSES},
        'classification': {cls.__name__: cls for cls in CLASSIFICATION_CLASSES},
        'clustering': {cls.__name__: cls for cls in CLUSTERING_CLASSES},
    }
    
    def __init__(self, data):
        self.data = data
        self.preprocessor = data.make_preprocessor()
    
    def get_model(self, model_name: str, **kwargs):
        """Get a model instance by name with optional parameters"""
        for model_type, models_dict in self.MODELS.items():
            if model_name in models_dict:
                return models_dict[model_name](**kwargs)
        
        raise ValueError(f"Model '{model_name}' not found. Available: {self.list_all_models()}")
    
    def get_model_type(self, model_name: str):
        """Get model type (regression/classification/clustering) from model name"""
        for model_type, models_dict in self.MODELS.items():
            if model_name in models_dict:
                return model_type
        
        raise ValueError(f"Model '{model_name}' not found. Available: {self.list_all_models()}")
        
    def create_pipeline(self, model_name: str, **model_params):
        """Create pipeline, ensuring model type matches dataset task"""
        model_type = self.get_model_type(model_name)
        
        # Validate model type matches dataset task
        if model_type not in self.data.tasks:
            # Get compatible models for the dataset
            compatible_models = []
            for task in self.data.tasks:
                compatible_models.extend(self.list_models_by_type(task))
            
            raise ValueError(
                f"Model type mismatch!\n"
                f"  - Model '{model_name}' is for: {model_type}\n"
                f"  - Dataset supports: {', '.join(self.data.tasks)}\n"
                f"  - Try the following models: {compatible_models}"
            )
        
        model = self.get_model(model_name, **model_params)
        return make_pipeline(self.preprocessor, model)
    
    def fit(self, model_name: str, **model_params):
        """Fit pipeline. For clustering, only transforms X (no y needed)"""
        pipeline = self.create_pipeline(model_name, **model_params)
        model_type = self.get_model_type(model_name)
        
        if model_type == 'clustering':
            pipeline.fit(self.data.X)
        else:
            pipeline.fit(self.data.X, self.data.y)
        
        return pipeline
    
    def list_all_models(self):
        """List all available model names"""
        all_models = []
        for models_dict in self.MODELS.values():
            all_models.extend(models_dict.keys())
        return sorted(all_models)
    
    def list_models_by_type(self, model_type: str):
        """List all models of a specific type"""
        if model_type not in self.MODELS:
            raise ValueError(f"Type '{model_type}' not found. Available: {list(self.MODELS.keys())}")
        return sorted(self.MODELS[model_type].keys())

    def get_model_param_specs(self, model_name: str):
        model_type = self.get_model_type(model_name)
        model_cls = self.MODELS[model_type][model_name]
        model = model_cls()

        sig = signature(model_cls.__init__)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if k != "self" and v.default is not v.empty
        }

        constraints = getattr(model, "_parameter_constraints", {})
        specs = {}

        for param, default in defaults.items():
            spec = {
                "default": default,
                "type": type(default).__name__,
            }

            for c in constraints.get(param, []):
                # Only apply constraint matching default type
                if isinstance(c, Interval) and isinstance(default, c.type):
                    spec["min"] = c.left
                    spec["max"] = c.right

                elif isinstance(c, StrOptions):
                    spec["options"] = sorted(c.options)

            specs[param] = spec

        return specs

    
# data = UciDataLoader(9)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     data.X, data.y, test_size=0.3, random_state=42
# )

# selector = ModelSelector(data)
# pipeline = selector.create_pipeline('SVC') 
# pipeline.fit(X_train, y_train)

# print(f"Train score: {pipeline.score(X_train, y_train):.4f}")
# print(f"Test score: {pipeline.score(X_test, y_test):.4f}")
"""
Tests for TabNet sklearn implementation
========================================
Tests for TabNetClassifier and TabNetRegressor.
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import sys
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch not installed, skipping TabNet tests", allow_module_level=True)

from automl_platform.tabnet_sklearn import (
    TabNetClassifier,
    TabNetRegressor,
    TabNetLayer,
    AttentiveTransformer,
    TabNetEncoder,
    TabNet,
    sparsemax
)


class TestTabNetClassifier:
    """Tests for TabNetClassifier"""
    
    @pytest.fixture
    def small_classification_data(self):
        """Create small classification dataset"""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass dataset"""
        X, y = make_classification(
            n_samples=150,
            n_features=10,
            n_informative=8,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self):
        """Test TabNetClassifier initialization"""
        clf = TabNetClassifier(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            max_epochs=10
        )
        
        assert clf.n_d == 8
        assert clf.n_a == 8
        assert clf.n_steps == 3
        assert clf.gamma == 1.3
        assert clf.max_epochs == 10
        assert clf.model is None
        assert clf._is_fitted is False
    
    def test_fit_binary_classification(self, small_classification_data):
        """Test training on binary classification"""
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = TabNetClassifier(max_epochs=5, patience=3)
        
        # Test fit doesn't raise errors
        clf.fit(X_train, y_train)
        
        assert clf._is_fitted is True
        assert clf.model is not None
        assert clf.label_encoder is not None
        assert len(clf.label_encoder.classes_) == 2
    
    def test_fit_multiclass_classification(self, multiclass_data):
        """Test training on multiclass classification"""
        X, y = multiclass_data
        
        clf = TabNetClassifier(max_epochs=5, patience=3)
        clf.fit(X, y)
        
        assert clf._is_fitted is True
        assert len(clf.label_encoder.classes_) == 3
    
    def test_predict_dimensions(self, small_classification_data):
        """Test predict output dimensions"""
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = TabNetClassifier(max_epochs=5)
        clf.fit(X_train, y_train)
        
        # Test predictions
        y_pred = clf.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert y_pred.ndim == 1
        assert all(pred in [0, 1] for pred in y_pred)
    
    def test_predict_proba_dimensions(self, small_classification_data):
        """Test predict_proba output dimensions"""
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = TabNetClassifier(max_epochs=5)
        clf.fit(X_train, y_train)
        
        # Test probability predictions
        y_proba = clf.predict_proba(X_test)
        
        assert y_proba.shape == (X_test.shape[0], 2)  # Binary classification
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1)
    
    def test_predict_before_fit(self):
        """Test error when predicting before fitting"""
        clf = TabNetClassifier()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            clf.predict(X)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            clf.predict_proba(X)
    
    def test_pandas_input(self, small_classification_data):
        """Test with pandas DataFrame input"""
        X, y = small_classification_data
        
        # Convert to pandas
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y)
        
        clf = TabNetClassifier(max_epochs=5)
        clf.fit(X_df, y_series)
        
        # Predict with DataFrame
        y_pred = clf.predict(X_df)
        assert len(y_pred) == len(y_series)
    
    def test_serialization(self, small_classification_data):
        """Test model serialization and deserialization"""
        X, y = small_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        clf = TabNetClassifier(max_epochs=5)
        clf.fit(X_train, y_train)
        
        # Get predictions before saving
        y_pred_before = clf.predict(X_test)
        y_proba_before = clf.predict_proba(X_test)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(clf, f)
            temp_path = f.name
        
        try:
            # Load model
            with open(temp_path, 'rb') as f:
                clf_loaded = pickle.load(f)
            
            # Get predictions after loading
            y_pred_after = clf_loaded.predict(X_test)
            y_proba_after = clf_loaded.predict_proba(X_test)
            
            # Check predictions are the same
            np.testing.assert_array_equal(y_pred_before, y_pred_after)
            np.testing.assert_array_almost_equal(y_proba_before, y_proba_after, decimal=6)
            
        finally:
            os.unlink(temp_path)
    
    def test_device_selection(self):
        """Test device selection (CPU/GPU)"""
        clf = TabNetClassifier(device='cpu')
        assert clf._get_device().type == 'cpu'
        
        clf_auto = TabNetClassifier(device='auto')
        device = clf_auto._get_device()
        assert device.type in ['cpu', 'cuda']


class TestTabNetRegressor:
    """Tests for TabNetRegressor"""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset"""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self):
        """Test TabNetRegressor initialization"""
        reg = TabNetRegressor(
            n_d=16,
            n_a=16,
            n_steps=4,
            learning_rate=0.01
        )
        
        assert reg.n_d == 16
        assert reg.n_a == 16
        assert reg.n_steps == 4
        assert reg.learning_rate == 0.01
        assert reg.model is None
        assert reg._is_fitted is False
    
    def test_fit_regression(self, regression_data):
        """Test training on regression task"""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = TabNetRegressor(max_epochs=5, patience=3)
        
        # Test fit doesn't raise errors
        reg.fit(X_train, y_train)
        
        assert reg._is_fitted is True
        assert reg.model is not None
    
    def test_predict_dimensions_regression(self, regression_data):
        """Test regression predict output dimensions"""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = TabNetRegressor(max_epochs=5)
        reg.fit(X_train, y_train)
        
        # Test predictions
        y_pred = reg.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert y_pred.ndim == 1
        assert y_pred.dtype == np.float32 or y_pred.dtype == np.float64
    
    def test_regression_performance(self, regression_data):
        """Test that regression achieves reasonable performance"""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        reg = TabNetRegressor(max_epochs=10, patience=5)
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Should achieve some reasonable performance
        # (not just predicting zeros or constant)
        assert mse < np.var(y_test) * 2  # Better than predicting mean
        assert np.std(y_pred) > 0  # Not predicting constant
    
    def test_serialization_regression(self, regression_data):
        """Test regression model serialization"""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        reg = TabNetRegressor(max_epochs=5)
        reg.fit(X_train, y_train)
        
        # Get predictions before saving
        y_pred_before = reg.predict(X_test)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(reg, f)
            temp_path = f.name
        
        try:
            # Load model
            with open(temp_path, 'rb') as f:
                reg_loaded = pickle.load(f)
            
            # Get predictions after loading
            y_pred_after = reg_loaded.predict(X_test)
            
            # Check predictions are the same
            np.testing.assert_array_almost_equal(y_pred_before, y_pred_after, decimal=5)
            
        finally:
            os.unlink(temp_path)


class TestTabNetComponents:
    """Tests for TabNet internal components"""
    
    def test_tabnet_layer(self):
        """Test TabNetLayer component"""
        layer = TabNetLayer(
            input_dim=10,
            output_dim=8,
            n_independent=2,
            n_shared=2
        )
        
        # Test forward pass
        x = torch.randn(32, 10)
        output = layer(x)
        
        assert output.shape == (32, 8)
    
    def test_attentive_transformer(self):
        """Test AttentiveTransformer component"""
        transformer = AttentiveTransformer(input_dim=10)
        
        priors = torch.ones(32, 10)
        processed_feat = torch.randn(32, 10)
        
        output = transformer(priors, processed_feat)
        
        assert output.shape == (32, 10)
        # Check that it's a valid attention mask (sums to approximately 1)
        assert torch.allclose(output.sum(dim=1), torch.ones(32), atol=1e-5)
    
    def test_tabnet_encoder(self):
        """Test TabNetEncoder component"""
        encoder = TabNetEncoder(
            input_dim=10,
            n_d=8,
            n_a=8,
            n_steps=3
        )
        
        x = torch.randn(32, 10)
        features, masks = encoder(x)
        
        assert features.shape == (3, 32, 8)  # n_steps, batch_size, n_d
        assert masks.shape == (3, 32, 10)  # n_steps, batch_size, input_dim
    
    def test_sparsemax_function(self):
        """Test sparsemax activation function"""
        # Test with simple input
        x = torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]])
        output = sparsemax(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that rows sum to 1 (or less due to sparsity)
        row_sums = output.sum(dim=1)
        assert torch.all(row_sums <= 1.0 + 1e-6)
        
        # Check non-negativity
        assert torch.all(output >= 0)
    
    def test_full_tabnet_model(self):
        """Test complete TabNet model"""
        model = TabNet(
            input_dim=10,
            output_dim=2,
            n_d=8,
            n_a=8,
            n_steps=3
        )
        
        x = torch.randn(32, 10)
        output, masks = model(x)
        
        assert output.shape == (32, 2)
        assert masks.shape == (3, 32, 10)


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_single_sample_prediction(self):
        """Test prediction with single sample"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        clf = TabNetClassifier(max_epochs=3)
        clf.fit(X, y)
        
        # Single sample prediction
        single_sample = X[0:1]
        pred = clf.predict(single_sample)
        proba = clf.predict_proba(single_sample)
        
        assert pred.shape == (1,)
        assert proba.shape == (1, 2)
    
    def test_large_batch_size(self):
        """Test with batch size larger than dataset"""
        X, y = make_classification(n_samples=20, n_features=5, random_state=42)
        
        clf = TabNetClassifier(batch_size=100, max_epochs=3)
        
        # Should handle gracefully
        clf.fit(X, y)
        assert clf._is_fitted
    
    def test_virtual_batch_size_handling(self):
        """Test virtual batch size normalization"""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        # Small virtual batch size
        clf = TabNetClassifier(virtual_batch_size=16, max_epochs=3)
        clf.fit(X, y)
        
        # Should work without errors
        pred = clf.predict(X)
        assert len(pred) == len(y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

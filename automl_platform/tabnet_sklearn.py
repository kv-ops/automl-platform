"""
TabNet Implementation compatible with sklearn
No external pytorch-tabnet dependency required
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings


class TabNetLayer(nn.Module):
    """Single TabNet layer implementation."""
    
    def __init__(self, input_dim, output_dim, n_independent=2, n_shared=2, virtual_batch_size=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.virtual_batch_size = virtual_batch_size
        
        # Shared layers
        self.shared_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else output_dim, output_dim)
            for i in range(n_shared)
        ])
        
        # Independent layers  
        self.independent_layers = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(n_independent)
        ])
        
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # Shared processing
        shared_out = x
        for layer in self.shared_layers:
            shared_out = F.relu(layer(shared_out))
        
        # Independent processing
        independent_out = shared_out
        for layer in self.independent_layers:
            independent_out = F.relu(layer(independent_out))
        
        # Virtual batch normalization
        if self.training and x.size(0) > self.virtual_batch_size:
            chunks = torch.split(independent_out, self.virtual_batch_size, dim=0)
            normalized_chunks = []
            for chunk in chunks:
                if chunk.size(0) > 1:
                    normalized_chunks.append(self.bn(chunk))
                else:
                    normalized_chunks.append(chunk)
            independent_out = torch.cat(normalized_chunks, dim=0)
        elif independent_out.size(0) > 1:
            independent_out = self.bn(independent_out)
            
        return independent_out


class AttentiveTransformer(nn.Module):
    """Attentive Transformer for feature selection."""
    
    def __init__(self, input_dim, virtual_batch_size=128):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, input_dim)
        self.virtual_batch_size = virtual_batch_size
        
    def forward(self, priors, processed_feat):
        x = torch.mul(priors, processed_feat)
        
        # Virtual batch normalization
        if self.training and x.size(0) > self.virtual_batch_size:
            chunks = torch.split(x, self.virtual_batch_size, dim=0)
            normalized_chunks = []
            for chunk in chunks:
                if chunk.size(0) > 1:
                    normalized_chunks.append(self.bn(chunk))
                else:
                    normalized_chunks.append(chunk)
            x = torch.cat(normalized_chunks, dim=0)
        elif x.size(0) > 1:
            x = self.bn(x)
            
        x = self.fc(x)
        return F.sparsemax(x, dim=-1)


class TabNetEncoder(nn.Module):
    """TabNet Encoder implementation."""
    
    def __init__(self, input_dim, n_d=64, n_a=64, n_steps=5, gamma=1.3,
                 n_independent=2, n_shared=2, virtual_batch_size=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a  
        self.n_steps = n_steps
        self.gamma = gamma
        self.virtual_batch_size = virtual_batch_size
        
        # Initial feature transformer
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # Attentive transformers for each step
        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(input_dim, virtual_batch_size)
            for _ in range(n_steps)
        ])
        
        # Feature transformers for each step
        self.feature_transformers = nn.ModuleList([
            TabNetLayer(input_dim, n_d + n_a, n_independent, n_shared, virtual_batch_size)
            for _ in range(n_steps)
        ])
        
    def forward(self, x):
        # Initial batch normalization
        if x.size(0) > 1:
            x = self.initial_bn(x)
        
        prior = torch.ones((x.size(0), self.input_dim), device=x.device)
        masks = []
        features = []
        
        for step in range(self.n_steps):
            # Attentive transformer
            mask = self.attentive_transformers[step](prior, x)
            masks.append(mask)
            
            # Apply mask to input
            masked_x = torch.mul(mask, x)
            
            # Feature transformer
            out = self.feature_transformers[step](masked_x)
            
            # Split decision and attention features
            decision_out = out[:, :self.n_d]
            attention_out = out[:, self.n_d:]
            
            features.append(decision_out)
            
            # Update prior for next step
            prior = torch.mul(self.gamma - mask, prior)
            
        return torch.stack(features, dim=0), torch.stack(masks, dim=0)


class TabNet(nn.Module):
    """Complete TabNet model."""
    
    def __init__(self, input_dim, output_dim, n_d=64, n_a=64, n_steps=5,
                 gamma=1.3, n_independent=2, n_shared=2, virtual_batch_size=128):
        super().__init__()
        
        self.encoder = TabNetEncoder(
            input_dim, n_d, n_a, n_steps, gamma,
            n_independent, n_shared, virtual_batch_size
        )
        
        # Final classifier/regressor
        self.final_layer = nn.Linear(n_d, output_dim)
        
    def forward(self, x):
        features, masks = self.encoder(x)
        
        # Aggregate features from all steps  
        aggregated = torch.sum(features, dim=0)
        
        # Final prediction
        out = self.final_layer(aggregated)
        
        return out, masks


# Sparsemax implementation
def sparsemax(tensor, dim=-1):
    """Sparsemax activation function."""
    original_size = tensor.size()
    tensor = tensor.view(-1, tensor.size(dim))
    
    # Sort in descending order
    sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)
    
    # Calculate cumulative sum
    cumsum = torch.cumsum(sorted_tensor, dim=1)
    
    # Calculate range
    range_tensor = torch.arange(1, tensor.size(1) + 1, device=tensor.device, dtype=tensor.dtype)
    range_tensor = range_tensor.expand_as(sorted_tensor)
    
    # Calculate condition
    condition = sorted_tensor - (cumsum - 1) / range_tensor > 0
    
    # Find k
    k = torch.sum(condition, dim=1, keepdim=True)
    
    # Calculate tau
    tau = (torch.gather(cumsum, 1, k - 1) - 1) / k.float()
    
    # Apply sparsemax
    output = torch.clamp(tensor - tau, min=0)
    
    return output.view(original_size)


# Add sparsemax to F for convenience  
F.sparsemax = sparsemax


class TabNetClassifier(BaseEstimator, ClassifierMixin):
    """TabNet Classifier compatible with sklearn."""
    
    def __init__(self, n_d=64, n_a=64, n_steps=5, gamma=1.3, n_independent=2,
                 n_shared=2, virtual_batch_size=128, max_epochs=100, patience=10,
                 learning_rate=0.02, batch_size=256, device='auto'):
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        
        self.model = None
        self.label_encoder = None
        self._is_fitted = False
        
    def _get_device(self):
        """Get appropriate device."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)
        
    def fit(self, X, y):
        """Fit TabNet classifier."""
        device = self._get_device()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Initialize model
        input_dim = X.shape[1]
        output_dim = len(self.label_encoder.classes_)
        
        self.model = TabNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            virtual_batch_size=self.virtual_batch_size
        ).to(device)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        y_val = torch.LongTensor(y_val).to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            # Training phase
            optimizer.zero_grad()
            
            if X_train.size(0) > self.batch_size:
                # Mini-batch training
                indices = torch.randperm(X_train.size(0))[:self.batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]
            else:
                batch_X = X_train
                batch_y = y_train
            
            outputs, _ = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Validation phase
            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _ = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        break
                        
                self.model.train()
        
        self._is_fitted = True
        return self
        
    def predict(self, X):
        """Make predictions."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        device = self._get_device()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X, dtype=np.float32)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            
        # Decode labels
        predictions = predictions.cpu().numpy()
        return self.label_encoder.inverse_transform(predictions)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        device = self._get_device()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X, dtype=np.float32)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()


class TabNetRegressor(BaseEstimator, RegressorMixin):
    """TabNet Regressor compatible with sklearn."""
    
    def __init__(self, n_d=64, n_a=64, n_steps=5, gamma=1.3, n_independent=2,
                 n_shared=2, virtual_batch_size=128, max_epochs=100, patience=10,
                 learning_rate=0.02, batch_size=256, device='auto'):
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        
        self.model = None
        self._is_fitted = False
        
    def _get_device(self):
        """Get appropriate device."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)
        
    def fit(self, X, y):
        """Fit TabNet regressor."""
        device = self._get_device()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Initialize model
        input_dim = X.shape[1]
        output_dim = 1
        
        self.model = TabNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            virtual_batch_size=self.virtual_batch_size
        ).to(device)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        y_val = torch.FloatTensor(y_val).to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            # Training phase
            optimizer.zero_grad()
            
            if X_train.size(0) > self.batch_size:
                # Mini-batch training
                indices = torch.randperm(X_train.size(0))[:self.batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]
            else:
                batch_X = X_train
                batch_y = y_train
            
            outputs, _ = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Validation phase
            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _ = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        break
                        
                self.model.train()
        
        self._is_fitted = True
        return self
        
    def predict(self, X):
        """Make predictions."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        device = self._get_device()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X, dtype=np.float32)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            
        return outputs.cpu().numpy().flatten()


# Usage example
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    # Classification example
    X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    
    clf = TabNetClassifier(max_epochs=50)
    clf.fit(X_cls, y_cls)
    
    y_pred = clf.predict(X_cls)
    print(f"Classification Accuracy: {accuracy_score(y_cls, y_pred):.4f}")
    
    # Regression example
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    
    reg = TabNetRegressor(max_epochs=50)
    reg.fit(X_reg, y_reg)
    
    y_pred_reg = reg.predict(X_reg)
    print(f"Regression MSE: {mean_squared_error(y_reg, y_pred_reg):.4f}")

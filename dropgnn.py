import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score

class DROPGNN(nn.Module):
    """
    DROPGNN: Dropout-based Robust Graph Neural Network
    Implements the model from the paper with adaptive dropout and robust training
    """
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout_rate=0.5, num_layers=2, heads=1):
        super(DROPGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Initialize convolutional layers
        if heads > 1:
            self.convs = nn.ModuleList([GATConv(num_features, hidden_dim, heads=heads)])
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            self.final_conv = GATConv(hidden_dim * heads, num_classes, heads=1)
        else:
            self.convs = nn.ModuleList([GCNConv(num_features, hidden_dim)])
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.final_conv = GCNConv(hidden_dim, num_classes)
        
        # Adaptive dropout parameters
        self.adaptive_dropout = nn.ModuleList([AdaptiveDropout(dropout_rate) for _ in range(num_layers)])
        
    def forward(self, x, edge_index):
        # Apply adaptive dropout and GNN layers
        for i in range(self.num_layers):
            x = self.adaptive_dropout[i](x)
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Final layer without activation
        x = self.final_conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def robust_loss(self, output, target, edge_index, lambda_=0.1):
        # Standard cross-entropy loss
        ce_loss = F.nll_loss(output, target)
        
        # Graph regularization loss (smoothness)
        diff = output[edge_index[0]] - output[edge_index[1]]
        reg_loss = torch.norm(diff, p=2) / edge_index.size(1)
        
        return ce_loss + lambda_ * reg_loss

class AdaptiveDropout(nn.Module):
    """
    Adaptive Dropout layer that adjusts dropout rates based on node importance
    """
    def __init__(self, base_rate=0.5):
        super(AdaptiveDropout, self).__init__()
        self.base_rate = base_rate
        self.importance = None
    
    def forward(self, x):
        if not self.training:
            return x
        
        if self.importance is None:
            # Initialize importance scores uniformly
            self.importance = torch.ones(x.size(0), device=x.device)
        
        # Compute adaptive dropout rates
        rates = self.base_rate * (1 - self.importance / (self.importance.max() + 1e-7))
        rates = rates.unsqueeze(1).expand_as(x)
        
        # Apply dropout
        mask = torch.bernoulli(1 - rates).float()
        return x * mask
    
    def update_importance(self, gradients):
        # Update importance scores based on gradient magnitudes
        grad_norms = torch.norm(gradients, p=2, dim=1)
        self.importance = 0.9 * self.importance + 0.1 * grad_norms

def train_model(model, data, optimizer, epochs=200, verbose=True):
    model.train()
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = model.robust_loss(output, data.y, data.edge_index)
        loss.backward()
        
        # Update adaptive dropout importance
        for layer in model.adaptive_dropout:
            if hasattr(layer, 'update_importance'):
                layer.update_importance(data.x.grad)
        
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.y.size(0)
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')
    
    return losses, accuracies

def visualize_graph(data, node_colors=None):
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    
    if node_colors is None:
        node_colors = data.y.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors, cmap='viridis', 
            with_labels=False, node_size=50, alpha=0.8)
    plt.title("Graph Visualization")
    return plt

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings.cpu().detach().numpy())
    
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels.cpu().numpy(), 
                cmap='viridis', alpha=0.6)
    plt.title("t-SNE Visualization of Node Embeddings")
    plt.colorbar()
    return plt

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred = output.argmax(dim=1)
        
        accuracy = accuracy_score(data.y.cpu(), pred.cpu())
        f1 = f1_score(data.y.cpu(), pred.cpu(), average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': pred,
            'output': output
        }

def streamlit_app():
    st.title("DROPGNN: Interactive Graph Neural Network")
    st.write("""
    This interactive demo showcases the DROPGNN model with adaptive dropout and robust training.
    Upload your own graph data or use the synthetic example.
    """)
    
    # Sidebar controls
    st.sidebar.header("Model Configuration")
    hidden_dim = st.sidebar.slider("Hidden dimension", 16, 256, 64)
    dropout_rate = st.sidebar.slider("Base dropout rate", 0.0, 0.9, 0.5)
    num_layers = st.sidebar.slider("Number of layers", 1, 5, 2)
    use_gat = st.sidebar.checkbox("Use GAT instead of GCN", False)
    heads = st.sidebar.slider("GAT heads", 1, 8, 1) if use_gat else 1
    epochs = st.sidebar.slider("Training epochs", 10, 500, 200)
    lr = st.sidebar.slider("Learning rate", 1e-4, 1e-1, 1e-2, step=1e-4, format="%.4f")
    
    # Load synthetic data or allow upload
    st.sidebar.header("Data Options")
    use_example = st.sidebar.checkbox("Use example data", True)
    
    if use_example:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
        st.write(f"Using Cora dataset with {data.x.size(0)} nodes and {data.edge_index.size(1)} edges.")
    else:
        st.warning("Custom data upload not implemented in this demo. Using example data.")
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
    
    # Initialize model
    model = DROPGNN(
        num_features=data.x.size(1),
        num_classes=dataset.num_classes,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        num_layers=num_layers,
        heads=heads
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    st.header("Model Training")
    if st.button("Train Model"):
        st.write("Training started...")
        losses, accuracies = train_model(model, data, optimizer, epochs=epochs, verbose=False)
        
        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        
        ax2.plot(accuracies)
        ax2.set_title("Training Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        
        st.pyplot(fig)
        
        # Evaluation
        st.header("Model Evaluation")
        results = evaluate_model(model, data)
        st.write(f"Test Accuracy: {results['accuracy']:.4f}")
        st.write(f"Macro F1 Score: {results['f1_score']:.4f}")
        
        # Visualizations
        st.header("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Graph Structure")
            fig = visualize_graph(data)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Node Embeddings")
            with torch.no_grad():
                embeddings = model.convs[0](data.x, data.edge_index)
            fig = visualize_embeddings(embeddings, data.y)
            st.pyplot(fig)
        
        # Show some predictions
        st.subheader("Sample Predictions")
        sample_nodes = np.random.choice(len(data.y), size=10, replace=False)
        sample_data = {
            "Node ID": sample_nodes,
            "True Label": data.y[sample_nodes].cpu().numpy(),
            "Predicted Label": results['predictions'][sample_nodes].cpu().numpy()
        }
        st.table(sample_data)

if __name__ == "__main__":
    streamlit_app()
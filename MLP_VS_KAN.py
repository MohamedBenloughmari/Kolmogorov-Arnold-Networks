import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KAN import KAN



def create_datasets():
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class KANModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KANModel, self).__init__()
        self.kan = nn.Sequential(
            KAN(n=input_dim, m=hidden_dim),
            KAN(n=hidden_dim, m=hidden_dim),
            KAN(n=hidden_dim, m=output_dim),
        )
    
    def forward(self, x):
        return self.kan(x)


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Testing
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            predicted = (torch.sigmoid(test_outputs) > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
        
        train_losses.append(loss.item())
        test_accuracies.append(accuracy.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')
    
    return train_losses, test_accuracies

if __name__ == "__main__":
    # Load and prepare data
    X_train, y_train, X_test, y_test = create_datasets()
    
    # Create models
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 1
    
    mlp_model = MLP(input_dim, hidden_dim, output_dim)
    kan_model = KANModel(input_dim, hidden_dim, output_dim)
    
    # Train MLP
    print("\nTraining MLP...")
    mlp_train_losses, mlp_test_accuracies = train_model(
        mlp_model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001
    )
    
    # Train KAN
    print("\nTraining KAN...")
    kan_train_losses, kan_test_accuracies = train_model(
        kan_model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001
    )
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp_train_losses, label='MLP')
    plt.plot(kan_train_losses, label='KAN')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_test_accuracies, label='MLP')
    plt.plot(kan_test_accuracies, label='KAN')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

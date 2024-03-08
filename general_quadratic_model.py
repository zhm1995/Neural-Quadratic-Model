import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(29)

# Initialize the dimensions of the data
d = 200  # Dimension of the feature space
n = 100  # Number of samples

# Generate a normally distributed dataset and normalize it
X = np.random.normal(0, 1, (n, d))
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Generate labels randomly from {1, -1}
y = np.random.choice([1, -1], n).reshape(n, 1)

# Regularization parameter
gamma = 1e-4

# Generate a random diagonal matrix for each sample, which is the Hessian.
random_diagonals = np.random.choice([1, -1], size=(n, d))
identity_matrix = np.eye(d)
Sigma = random_diagonals[:, :, np.newaxis] * identity_matrix[np.newaxis, :, :]

# Learning rate and epochs for gradient descent
lr = 38  # Learning rate
epoch = 300  # Number of epochs

# Initialize weights vector
w = np.zeros((d, 1))

# List to store loss values
losses = []
top_eigs = []
# Gradient descent loop
for t in range(epoch):
    # Compute gradient of the loss with respect to weights
    dfdw = X + gamma * np.einsum('nij,jk->nik', Sigma, w).squeeze(2)
    # Compute the function value
    f = X @ w + 0.5 * gamma * np.einsum('nij,jk,ik->nk', Sigma, w, w)
    # Update weights using gradient descent step
    w = w - 2 / n * lr * dfdw.T @ (f - y)
    # Compute and record the loss
    loss = 1 / n * np.linalg.norm(f - y) ** 2

    K = dfdw@dfdw.T
    top_eigs.append(np.linalg.norm(K,2))
    losses.append(loss)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Creating 1 row, 2 columns subplot

# Plotting the loss
axs[0].plot(losses)
axs[0].set_title('Loss over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].tick_params(axis='both', which='major', labelsize=10)

# Plotting the top eigenvalues
axs[1].plot(top_eigs)
axs[1].set_title('Top Eigenvalue over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Top Eigenvalue')
axs[1].tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()


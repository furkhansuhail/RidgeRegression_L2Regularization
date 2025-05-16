# --- Ridge Regression from scratch (only for 2x2 feature matrices) ---

# Function to transpose a matrix (rows become columns and vice versa)
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


# Function to multiply two matrices
def matmul(A, B):
    # Initialize a result matrix filled with zeros
    result = [[0] * len(B[0]) for _ in range(len(A))]
    # Multiply each row of A by each column of B
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


# Function to add two matrices element-wise
def add_matrices(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


# Function to create an identity matrix of given size
def identity_matrix(size):
    I = [[0] * size for _ in range(size)]
    for i in range(size):
        I[i][i] = 1  # Set diagonal elements to 1
    return I


# Function to invert a 2x2 matrix
def inverse_matrix(matrix):
    # Check if matrix is 2x2
    if len(matrix) != 2 or len(matrix[0]) != 2:
        raise ValueError("Only 2x2 matrix inversion implemented")

    # Unpack matrix elements
    a, b = matrix[0]
    c, d = matrix[1]

    # Calculate the determinant
    det = a * d - b * c
    if det == 0:
        raise ValueError("Matrix not invertible")  # No inverse if determinant is zero

    # Inverse formula for 2x2 matrix
    inv_det = 1 / det
    return [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det]
    ]


# Main Ridge Regression function
def ridge_regression(X, y, lambda_):
    # Step 1: Calculate X^T (transpose of X)
    X_T = transpose(X)

    # Step 2: Calculate X^T * X
    XTX = matmul(X_T, X)

    # Step 3: Create lambda * I (identity matrix scaled by lambda)
    n_features = len(XTX)  # Number of features (columns)
    I = identity_matrix(n_features)
    lambda_I = [[lambda_ * I[i][j] for j in range(n_features)] for i in range(n_features)]

    # Step 4: Add lambda * I to X^T * X
    XTX_plus_lambdaI = add_matrices(XTX, lambda_I)

    # Step 5: Invert the (X^T * X + lambda * I) matrix
    XTX_inv = inverse_matrix(XTX_plus_lambdaI)

    # Step 6: Calculate X^T * y
    # Reshape y as a column vector for matrix multiplication
    XTy = matmul(X_T, [[v] for v in y])

    # Step 7: Calculate final weights
    # weights = (X^T X + λI)^-1 X^T y
    weights = matmul(XTX_inv, XTy)

    # Flatten the result (remove extra brackets)
    return [w[0] for w in weights]


# --- Example Usage ---

# Example feature matrix X (4 samples, 2 features)
X = [
    [1, 1],
    [1, 2],
    [2, 2],
    [2, 3]
]

# Example target vector y
y = [6, 8, 9, 11]

# Regularization strength
lambda_ = 0.5

# Train the Ridge Regression model
weights = ridge_regression(X, y, lambda_)

# Output the learned weights
print("Learned Weights:", weights)

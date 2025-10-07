import functools
from typing import Callable, List, Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


max_len = 28


def transformer(
    X: ad.Node,
    nodes: List[ad.Node],
    model_dim: int,
    seq_length: int,
    eps,
    batch_size,
    num_classes,
) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """
    # Unpack the parameter nodes
    W_input, W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # Input projection to map from input_dim to model_dim
    X_projected = ad.matmul(X, W_input)  # (batch_size, seq_length, model_dim)

    # Multi-head Self-Attention
    # Compute Q, K, V matrices
    Q = ad.matmul(X_projected, W_Q)  # (batch_size, seq_length, model_dim)
    K = ad.matmul(X_projected, W_K)  # (batch_size, seq_length, model_dim)
    V = ad.matmul(X_projected, W_V)  # (batch_size, seq_length, model_dim)

    # Compute attention scores: Q @ K^T
    K_T = ad.transpose(K, -1, -2)  # (batch_size, model_dim, seq_length)
    attention_scores = ad.matmul(Q, K_T)  # (batch_size, seq_length, seq_length)

    # Scale by sqrt(model_dim)
    scale = 1.0 / np.sqrt(model_dim)
    scaled_scores = ad.mul_by_const(attention_scores, scale)

    # Apply softmax to get attention weights
    attention_weights = ad.softmax(scaled_scores, dim=-1)

    # Apply attention to values
    attention_output = ad.matmul(
        attention_weights, V
    )  # (batch_size, seq_length, model_dim)

    # Apply output projection
    attention_output = ad.matmul(attention_output, W_O)

    # Add & Norm (Layer Normalization after residual connection)
    residual_1 = ad.add(X_projected, attention_output)
    norm_1 = ad.layernorm(residual_1, normalized_shape=[model_dim], eps=eps)

    # Feed Forward Network
    # First linear layer with ReLU activation
    ff_1 = ad.matmul(norm_1, W_1)
    ff_1 = ad.add(ff_1, b_1)  # Add bias
    ff_1 = ad.relu(ff_1)

    # Second linear layer
    ff_2 = ad.matmul(ff_1, W_2)
    ff_2 = ad.add(ff_2, b_2)  # Add bias

    # Add & Norm (Second residual connection with layer norm)
    # Note: For the final layer, we might not need residual connection
    # since dimensions might not match. Let's just use the ff_2 output.

    # Global average pooling over sequence length for classification
    # Average over the sequence dimension (dim=1)
    pooled_output = ad.mean(ff_2, dim=(1,), keepdim=False)  # (batch_size, num_classes)

    return pooled_output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    # Compute softmax probabilities
    softmax_probs = ad.softmax(Z, dim=-1)

    # Compute cross-entropy loss: -sum(y_true * log(y_pred))
    log_probs = ad.log(softmax_probs)
    cross_entropy = ad.mul(y_one_hot, log_probs)

    # Sum over classes (dim=1) and then over batch (dim=0)
    loss_per_sample = ad.sum_op(cross_entropy, dim=(1,), keepdim=False)
    total_loss = ad.sum_op(loss_per_sample, dim=(0,), keepdim=False)

    # Negate and average over batch
    neg_loss = ad.mul_by_const(total_loss, -1.0)
    avg_loss = ad.div_by_const(neg_loss, batch_size)

    return avg_loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (
        num_examples + batch_size - 1
    ) // batch_size  # Compute the number of batches
    total_loss = 0.0

    # Unpack model weights at the beginning
    W_input, W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = model_weights

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Compute forward and backward passes
        (
            logits,
            loss_val,
            grad_W_input,
            grad_W_Q,
            grad_W_K,
            grad_W_V,
            grad_W_O,
            grad_W_1,
            grad_W_2,
            grad_b_1,
            grad_b_2,
        ) = f_run_model(
            [W_input, W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2, X_batch, y_batch]
        )

        # Update weights and biases
        W_input = W_input - lr * grad_W_input
        W_Q = W_Q - lr * grad_W_Q
        W_K = W_K - lr * grad_W_K
        W_V = W_V - lr * grad_W_V
        W_O = W_O - lr * grad_W_O
        W_1 = W_1 - lr * grad_W_1
        W_2 = W_2 - lr * grad_W_2
        b_1 = b_1 - lr * grad_b_1
        b_2 = b_2 - lr * grad_b_2

        # Accumulate the loss
        total_loss += loss_val.item() * (end_idx - start_idx)

    # Update model_weights list
    model_weights = [W_input, W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    # Compute the average loss
    average_loss = total_loss / num_examples
    print("Avg_loss:", average_loss)

    return model_weights, average_loss


def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10  #
    model_dim = 128  #
    eps = 1e-5

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # TODO: Define the forward graph.
    X = ad.Variable(name="X")
    W_input = ad.Variable(name="W_input")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")

    nodes = [W_input, W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(
        X, nodes, model_dim, seq_length, eps, batch_size, num_classes
    )
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)

    # TODO: Construct the backward graph.
    grads: List[ad.Node] = ad.gradients(
        loss, [W_input, W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]
    )

    # TODO: Create the evaluator.
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # Convert the train dataset to NumPy arrays
    X_train = (
        train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = (
        test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    )  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=False
    )  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_input_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_Q_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(inputs):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        (
            W_input_val,
            W_Q_val,
            W_K_val,
            W_V_val,
            W_O_val,
            W_1_val,
            W_2_val,
            b_1_val,
            b_2_val,
            X_val,
            y_val,
        ) = inputs
        result = evaluator.run(
            input_values={
                X: X_val.float(),
                W_input: W_input_val,
                W_Q: W_Q_val,
                W_K: W_K_val,
                W_V: W_V_val,
                W_O: W_O_val,
                W_1: W_1_val,
                W_2: W_2_val,
                b_1: b_1_val,
                b_2: b_2_val,
                y_groundtruth: y_val.float(),
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (
            num_examples + batch_size - 1
        ) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run(
                {
                    X: torch.tensor(X_batch, dtype=torch.float32),
                    W_input: model_weights[0],
                    W_Q: model_weights[1],
                    W_K: model_weights[2],
                    W_V: model_weights[3],
                    W_O: model_weights[4],
                    W_1: model_weights[5],
                    W_2: model_weights[6],
                    b_1: model_weights[7],
                    b_2: model_weights[8],
                }
            )
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test = (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.DoubleTensor(y_train),
        torch.DoubleTensor(y_test),
    )
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_input_val, dtype=torch.float32),
        torch.tensor(W_Q_val, dtype=torch.float32),
        torch.tensor(W_K_val, dtype=torch.float32),
        torch.tensor(W_V_val, dtype=torch.float32),
        torch.tensor(W_O_val, dtype=torch.float32),
        torch.tensor(W_1_val, dtype=torch.float32),
        torch.tensor(W_2_val, dtype=torch.float32),
        torch.tensor(b_1_val, dtype=torch.float32),
        torch.tensor(b_2_val, dtype=torch.float32),
    ]
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")

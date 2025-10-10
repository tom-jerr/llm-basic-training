import numpy as np


def split_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mp_size: int,
    dp_size: int,
    rank: int,
):
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        - Data is split uniformly across data parallel (DP) groups.
        - All model parallel (MP) ranks within the same DP group share the same data.
        - The data length is guaranteed to be divisible by dp_size.
        - Do not shuffle the data indices as shuffling will be done later.
    """

    # TODO: Your code here
    split_x_train = None
    split_y_train = None
    chunk_x_sz = x_train.shape[0] // dp_size
    chunk_y_sz = y_train.shape[0] // dp_size
    for i in range(dp_size):
        if (rank // mp_size) % dp_size == i:
            split_x_train = x_train[i * chunk_x_sz : (i + 1) * chunk_x_sz]
            split_y_train = y_train[i * chunk_y_sz : (i + 1) * chunk_y_sz]

    return split_x_train, split_y_train

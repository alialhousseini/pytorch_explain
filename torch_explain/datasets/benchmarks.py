import numpy as np
import torch
import logging
import typing

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

logger = logging.getLogger(__name__)


def is_bin_even(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset for the 21-multiplexer problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples uniformly between 0 and 1
    x = np.random.uniform(0, 1, (size, 4))
    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = np.stack([
        x[:, 0] > 0.5,  # c0
        x[:, 1] > 0.5,  # c1
        x[:, 2] > 0.5,  # c2
        x[:, 3] > 0.5,  # c3
    ]).T
    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    y = np.logical_not(c[:, 3])

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def mux412(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a balanced dataset for the 21-multiplexer problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    if size % 16 != 0:
        raise ValueError(
            "Size must be a multiple of 16 for balancing purposes.")

    np.random.seed(random_state)

    # Generate a balanced set of combinations for the concepts
    base_combinations = np.array([[i, j, k, l, m, n]
                                  for i in [0, 1]
                                  for j in [0, 1]
                                  for k in [0, 1]
                                  for l in [0, 1]
                                  for m in [0, 1]
                                  for n in [0, 1]])

    # Ensure enough repetitions to match the desired size
    repetitions = size // 16
    balanced_combinations = np.tile(base_combinations, (repetitions, 1))

    # Shuffle the dataset to ensure randomness
    np.random.shuffle(balanced_combinations)

    # Convert to float and normalize between 0 and 1
    x = balanced_combinations.astype(np.float32)
    # Add small random noise to avoid exact 0.5 threshold
    x = x + np.random.uniform(0, 0.5, x.shape)
    x = np.clip(x, 0, 1)  # Ensure values are between 0 and 1

    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = balanced_combinations

    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    # Define labels based on conditions
    y1 = np.logical_or(
        np.logical_and(
            np.logical_and(c[:, 0], np.logical_not(c[:, 4])),
            np.logical_not(c[:, 5])),

        np.logical_and(
            np.logical_and(c[:, 1], np.logical_not(c[:, 4])),
            c[:, 5])
    )

    y2 = np.logical_or(
        np.logical_and(
            np.logical_and(c[:, 2], c[:, 4]),
            np.logical_not(c[:, 5])),

        np.logical_and(
            np.logical_and(c[:, 3], c[:, 4]),
            c[:, 5])
    )

    y = np.logical_or(y1, y2)

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def mux41(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset for the 21-multiplexer problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples uniformly between 0 and 1
    x = np.random.uniform(0, 1, (size, 6))
    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = np.stack([
        x[:, 0] > 0.5,  # c0
        x[:, 1] > 0.5,  # c1
        x[:, 2] > 0.5,  # c2
        x[:, 3] > 0.5,  # c3
        x[:, 4] > 0.5,  # x1
        x[:, 5] > 0.5,  # x2
    ]).T
    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    # Define labels as XOR of the two concepts
    y1 = np.logical_or(
        np.logical_and(
            np.logical_and(c[:, 0], np.logical_not(c[:, 4])),
            np.logical_not(c[:, 5])),

        np.logical_and(
            np.logical_and(c[:, 1], np.logical_not(c[:, 4])),
            c[:, 5])
    )

    y2 = np.logical_or(
        np.logical_and(
            np.logical_and(c[:, 2], c[:, 4]),
            np.logical_not(c[:, 5])),

        np.logical_and(
            np.logical_and(c[:, 3], c[:, 4]),
            c[:, 5])
    )

    y = np.logical_or(y1, y2)

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def mux41Mod(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset for the 21-multiplexer problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples uniformly between 0 and 1
    x = np.random.uniform(0, 1, (size, 6))
    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = np.stack([
        x[:, 0] > 0.5,  # c0
        x[:, 1] > 0.5,  # c1
        x[:, 2] > 0.5,  # c2
        x[:, 3] > 0.5,  # c3
        x[:, 4] > 0.5,  # x1
        x[:, 5] > 0.5,  # x2
    ]).T
    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    y = (x[:, 4] > 0.5).astype(int) * 2 + (x[:, 5] > 0.5).astype(int)

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def mux41_two_inputs(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset for the 41-multiplexer modified problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples uniformly between 0 and 1
    x = np.random.uniform(0, 1, (size, 3))
    ones = np.ones((size, 1))
    zeros = np.zeros((size, 1))
    x = np.hstack([x, ones, zeros])

    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = np.stack([
        x[:, 0] > 0.5,  # c
        x[:, 1] > 0.5,  # A
        x[:, 2] > 0.5,  # B
        x[:, 3] > 0.5,  # 1
        x[:, 4] > 0.5,  # 0
    ]).T
    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    # Define labels as XOR of the two concepts
    y1 = np.logical_and(
        np.logical_not(c[:, 1]),
        c[:, 0])

    y2 = np.logical_and(
        c[:, 1],
        np.logical_not(c[:, 2]))

    y = np.logical_or(y1, y2)

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def two_muxes(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset for the 41-multiplexer modified problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples uniformly between 0 and 1
    x = np.random.uniform(0, 1, (size, 3))

    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = np.stack([
        x[:, 0] > 0.5,  # E
        x[:, 1] > 0.5,  # X
        x[:, 2] > 0.5,  # Y
    ]).T

    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    # Define labels as XOR of the two concepts
    y1 = np.logical_and(
        np.logical_not(c[:, 2]),
        c[:, 1])

    y2 = np.logical_and(
        np.logical_and(
            c[:, 2],
            np.logical_not(c[:, 1])),
        c[:, 0])

    y = np.logical_or(y1, y2)

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def xor_mod(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a balanced dataset for the XOR problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Ensure size is divisible by 4 for equal distribution
    assert size % 4 == 0, "Size must be divisible by 4 to ensure equal distribution of samples."

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Calculate the number of samples per combination
    samples_per_combination = size // 4

    # Create each combination of inputs
    combinations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Generate the balanced dataset
    x_list = []
    c_list = []
    y_list = []

    for comb in combinations:
        x1 = np.random.uniform(0, 0.5, samples_per_combination) if comb[0] == 0 else np.random.uniform(
            0.5, 1, samples_per_combination)
        x2 = np.random.uniform(0, 0.5, samples_per_combination) if comb[1] == 0 else np.random.uniform(
            0.5, 1, samples_per_combination)

        x_comb = np.column_stack((x1, x2))
        c_comb = np.array([comb] * samples_per_combination)
        y_comb = np.logical_xor(c_comb[:, 0], c_comb[:, 1])

        x_list.append(x_comb)
        c_list.append(c_comb)
        y_list.append(y_comb)

    # Concatenate the lists to form the final arrays
    x = np.vstack(x_list)
    c = np.vstack(c_list)
    y = np.hstack(y_list)

    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def xor(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset for the XOR problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples uniformly between 0 and 1
    x = np.random.uniform(0, 1, (size, 2))
    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = np.stack([
        x[:, 0] > 0.5,
        x[:, 1] > 0.5,
    ]).T

    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    # Define labels as XOR of the two concepts
    y = np.logical_xor(c[:, 0], c[:, 1])

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def xnor(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset for the XOR problem.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples uniformly between 0 and 1
    x = np.random.uniform(0, 1, (size, 2))
    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")

    # Define concepts: whether each value is greater than 0.5
    c = np.stack([
        x[:, 0] > 0.5,
        x[:, 1] > 0.5,
    ]).T

    if log:
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    # Define labels as XOR of the two concepts
    y = np.logical_not(np.logical_xor(c[:, 0], c[:, 1]))

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    return x, c, y.unsqueeze(-1)


def trigonometry(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset with trigonometric features and a downstream task.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (input_features), concepts (concepts), and labels (downstream_task).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random samples from a normal distribution
    h = np.random.normal(0, 2, (size, 3))
    if log:
        logger.info(f"h: {h.shape}")
        logger.info(f"h: {h[:5]}")

    x, y, z = h[:, 0], h[:, 1], h[:, 2]

    # Compute raw features using trigonometric and polynomial functions
    input_features = np.stack([
        np.sin(x) + x,
        np.cos(x) + x,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        x ** 2 + y ** 2 + z ** 2,
    ]).T

    if log:
        logger.info(f"input_features: {input_features.shape}")
        logger.info(f"input_features: {input_features[:5]}")

    # Define concepts: whether each value is greater than 0
    concepts = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    if log:
        logger.info(f"concepts: {concepts.shape}")
        logger.info(f"concepts: {concepts[:5]}")

    # Define downstream task: whether the sum of values exceeds 1
    downstream_task = (x + y + z) > 1

    if log:
        logger.info(f"downstream_task: {downstream_task.shape}")
        logger.info(f"downstream_task: {downstream_task[:5]}")

    # Convert numpy arrays to PyTorch tensors
    input_features = torch.FloatTensor(input_features)
    concepts = torch.FloatTensor(concepts)
    downstream_task = torch.FloatTensor(downstream_task)

    return input_features, concepts, downstream_task.unsqueeze(-1)


def dot(size: int, random_state: int = 42, log: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset based on dot products of vectors.

    Args:
    size (int): Number of samples to generate.
    random_state (int): Seed for random number generation.
    log (bool): Whether to log the dataset details.

    Returns:
    tuple: A tuple containing input features (x), concepts (c), and labels (y).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Generate random vectors from a normal distribution
    emb_size = 2
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    if log:
        logger.info(f"v1: {v1.shape}")
        logger.info(f"v2: {v2.shape}")
        logger.info(f"v3: {v3.shape}")
        logger.info(f"v4: {v4.shape}")

    # Concatenate vectors to form input features
    x = np.hstack([v1 + v3, v1 - v3])

    # Define concepts based on dot product results
    c = np.stack([
        np.dot(v1, v2).ravel() > 0,
        np.dot(v3, v4).ravel() > 0,
    ]).T

    if log:
        logger.info(f"x: {x.shape}")
        logger.info(f"x: {x[:5]}")
        logger.info(f"c: {c.shape}")
        logger.info(f"c: {c[:5]}")

    # Define labels based on element-wise multiplication of vectors
    y = ((v1 * v3).sum(axis=-1) > 0).astype(np.int64)

    if log:
        logger.info(f"y: {y.shape}")
        logger.info(f"y: {y[:5]}")

    # Convert numpy arrays to PyTorch tensors
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.Tensor(y)

    return x, c, y.unsqueeze(-1)


if __name__ == "__main__":
    xor(10, log=True)
    # print('--------------------------------')
    # trigonometry(10, log=True)
    # print('--------------------------------')
    # dot(10, log=True)
    # print('--------------------------------')
    # mux41(10, log=True)

"""
This script demonstrates the implementation of the Binary Step activation function.

Activation functions are crucial components in artificial neural networks. They introduce non-linearity
into the model, allowing it to learn complex patterns. The Binary Step function is one of the simplest
activation functions. It outputs 1 if the input is greater than or equal to 0, and 0 otherwise.
This mimics the behavior of a neuron that is either "firing" (activated) or not.

Key properties of the Binary Step function:
- It's a step function, hence the name.
- It's not differentiable at x = 0, which can cause issues with gradient-based optimization methods
  like backpropagation, as the derivative is 0 everywhere except at 0 where it's undefined.
- Due to its binary nature, it's often used in simple models or for thresholding, but not in deep
  learning where smooth, differentiable functions like sigmoid or ReLU are preferred.
- Range: {0, 1}
- It's saturating, meaning it can lead to vanishing gradients in multi-layer networks.

This function is mentioned in the Wikipedia article on Activation Functions:
https://en.wikipedia.org/wiki/Activation_function

The script uses NumPy for efficient array operations, as neural networks often deal with vectors
and matrices of data.

Imports:
- numpy: For numerical computations and array handling.
"""

import numpy as np


def validate_input(vector):
    """
    Validates the input for the binary step function.

    Checks if the input is a NumPy array and contains numeric values.

    Parameters:
        vector: The input to validate.

    Returns:
        bool: True if valid, False otherwise.

    Raises:
        TypeError: If input is not a NumPy array.
        ValueError: If array contains non-numeric values.
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if not np.issubdtype(vector.dtype, np.number):
        raise ValueError("Array must contain numeric values.")
    return True


def apply_binary_step(vector):
    """
    Applies the Binary Step activation function to the input vector.

    Parameters:
        vector (np.ndarray): A NumPy array of numeric values.

    Returns:
        np.ndarray: The result after applying binary step.
    """
    validate_input(vector)
    return np.where(vector >= 0, 1, 0)


def print_result(input_vector, output_vector):
    """
    Prints the input and output vectors in a formatted way.

    Parameters:
        input_vector (np.ndarray): The original input.
        output_vector (np.ndarray): The result after binary step.

    Returns:
        None
    """
    print(f"Input:  {input_vector}")
    print(f"Output: {output_vector}")
    print("-" * 40)


def run_example(vector, description=""):
    """
    Runs an example by applying binary step and printing results.

    Parameters:
        vector (np.ndarray): The input vector.
        description (str): Optional description of the example.

    Returns:
        None
    """
    if description:
        print(f"{description}")
    result = apply_binary_step(vector)
    print_result(vector, result)


def run_doctests():
    """
    Runs doctests for the binary_step function.

    Returns:
        None
    """
    import doctest
    print("Running doctests...")
    doctest.testmod(verbose=True)
    print("Doctests completed.\n")


def demonstrate_binary_step():
    """
    Demonstrates the Binary Step function with multiple examples.

    Calls various small functions to show different scenarios.
    """
    print("Demonstrating the Binary Step Activation Function")
    print("=" * 50)
    
    # Example 1: Basic mixed vector
    run_example(np.array([-1.2, 0, 2, 1.45, -3.7, 0.3]), "Example 1: Mixed values")
    
    # Example 2: All positive
    run_example(np.array([1, 2, 3, 4]), "Example 2: All positive")
    
    # Example 3: All negative
    run_example(np.array([-1, -2, -3, -4]), "Example 3: All negative")
    
    # Example 4: Zeros and floats
    run_example(np.array([0.0, -0.5, 0.5, -1.0, 1.0]), "Example 4: Zeros and floats")
    
    # Run doctests
    run_doctests()


if __name__ == "__main__":
    """
    Main entry point for the script.

    Calls the demonstration function to show the binary step in action.
    """
    demonstrate_binary_step()

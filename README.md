# Pytorch MNIST MLP Variants Comparison

This notebook trains and compares four different Multilayer Perceptron (MLP) models on the MNIST dataset. The goal is to analyze the impact of various architectural and regularization techniques on performance.

## Models Compared

The notebook defines and trains four distinct MLP configurations:

*   **Model A — Baseline (Shallow, Light):** `784 → 128 → 10`, uses ReLU activation, no Batch Normalization, and no Dropout.
*   **Model B — Deep + BN + Dropout:** `784 → 512 → 256 → 128 → 10`, uses ReLU activation, includes BatchNorm layers, and applies Dropout with a rate of 0.3.
*   **Model C — Wide + GELU + strong Dropout:** `784 → 1024 → 10`, uses GELU activation, and applies strong Dropout with a rate of 0.5.
*   **Model D — Compact + Reg + Label Smoothing:** `784 → 256 → 256 → 10`, uses ReLU activation, applies Dropout with a rate of 0.4, uses AdamW optimizer with weight decay, and incorporates Label Smoothing with a factor of 0.05.

## Key Features

*   **Configurable MLP Model:** A flexible `MLP` class allows easy definition of models with varying depth, width, activation functions, Batch Normalization, and Dropout.
*   **Cross-Entropy with Label Smoothing:** Implementation of a custom loss function that incorporates label smoothing.
*   **Reproducibility:** Includes a function to set random seeds for Python, NumPy, and PyTorch.
*   **Experiment Configuration:** Defines configurations for each of the four models using a dataclass.
*   **Training and Evaluation Loops:** Standard functions for training one epoch and evaluating the model performance.
*   **Learning Rate Schedulers:** Support for StepLR and CosineAnnealingLR schedulers.
*   **Optimizer Builder:** Helper function to build SGD or AdamW optimizers.
*   **Utility Functions:** Includes functions to count parameters, calculate accuracy from logits, generate confusion matrices, and compute per-class accuracy.
*   **Results Visualization:** Plots training/validation loss and accuracy curves, and generates confusion matrices.
*   **Experiment Runner:** A function to run a single experiment end-to-end (train, validate, test, save artifacts).
*   **Summary Generation:** Runs all defined experiments and prints a summary table of results.

## Setup and Usage

1.  **Clone the repository (if applicable) or open the notebook in Google Colab.**
2.  **Ensure you have the necessary libraries installed.** The notebook uses `torch`, `torchvision`, `numpy`, and `matplotlib`. These are standard in Colab or can be installed via pip.
3.  **Run the cells sequentially.** The notebook is structured to download the MNIST dataset, define models and utility functions, configure experiments, run them, and display a summary.
4.  **Results** will be saved in the `./results` directory, including model weights, per-class accuracy JSON files, loss/accuracy plots, and confusion matrix plots. A `summary.json` file will also contain a summary of all experiment results.

## Analysis

After running the experiments, you can analyze the generated plots and the summary table to compare the performance of the different MLP variants and understand the impact of the architectural and regularization choices.

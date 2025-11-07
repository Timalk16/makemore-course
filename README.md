# Makemore - Character-Level Language Model

A character-level language model implementation for generating names, based on Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) course. This project demonstrates the fundamentals of neural language modeling using PyTorch.

## Overview

This project implements a character-level bigram language model that learns to generate names by predicting the next character given the previous character. The model is trained on a dataset of names and uses a simple neural network architecture to learn character-level patterns.

## Features

- **Bigram Character Model**: Implements a character-level bigram model that learns character transition probabilities
- **Neural Network Implementation**: Uses PyTorch to build and train a neural network for character prediction
- **Name Generation**: Generates new names based on learned patterns from the training data
- **Visualization**: Includes visualizations of character transition matrices

## Project Structure

- `makemore.ipynb` - Main Jupyter notebook containing the complete implementation

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Timalk16/makemore-course.git
cd makemore-course
```

2. Install the required dependencies:
```bash
pip install torch numpy matplotlib jupyter
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook makemore.ipynb
```

2. Run the cells sequentially. The notebook will:
   - Download the names dataset
   - Preprocess the data
   - Build character mappings
   - Train a neural network model
   - Generate new names

## What's Inside

The notebook covers:

1. **Data Loading**: Downloads and loads a dataset of names
2. **Character Encoding**: Maps characters to indices and vice versa
3. **Bigram Counting**: Counts character bigram frequencies
4. **Probability Modeling**: Creates probability distributions for character transitions
5. **Neural Network**: Implements a simple feedforward neural network
6. **Training**: Uses gradient descent to optimize the model
7. **Generation**: Samples new names from the trained model

## Model Architecture

The model uses a simple architecture:
- Input: One-hot encoded character vectors (27 dimensions: 26 letters + special start/end token)
- Hidden Layer: Linear transformation with learned weights
- Output: Probability distribution over next characters (softmax)

## Training

The model is trained using:
- Negative log-likelihood loss
- Gradient descent optimization
- 100 training iterations

## Results

After training, the model can generate novel names that follow similar patterns to the training data, demonstrating the ability to learn character-level language patterns.

## References

- [Makemore by Andrej Karpathy](https://github.com/karpathy/makemore)
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)

## License

This project is for educational purposes.


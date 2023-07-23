# PyTorch to ONNX Conversion Project

This project demonstrates how to convert a PyTorch model to the ONNX format using the IMDb movie reviews dataset and the BERT model for sentiment analysis.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to showcase the process of converting a PyTorch model to the ONNX format. We use the IMDb movie reviews dataset for sentiment analysis and the BERT model, which is a popular Transformer-based language model developed by Google.

The steps involved in the project are as follows:
1. Training a BERT model using PyTorch on the IMDb movie reviews dataset.
2. Converting the trained PyTorch model to the ONNX format, which is a standard format for representing machine learning models.
3. Demonstrating how to incorporate the speed comparison when running the ONNX model.
4. Implementing quantization, which is a technique to reduce the memory footprint and speed up inference of the ONNX model.
5. Visualizing the model to gain insights into its architecture and layers.

## Requirements

Before running the project, ensure you have the following prerequisites:

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- ONNX

You can install the required libraries using the following command:

```bash
pip install torch transformers onnx



# Transformer Training Pipeline for LLaMA with FLOPs Utilization and Scaling Insights
**Description:** This project aims to enhance understanding and efficiency in training large language models, particularly by exploring the inner workings of transformers within models like LLaMA. Our initial work focuses on setting up and refining a training pipeline for the LLaMA 3.2-1B model.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Reproducibility](#reproducibility)

---

### Overview

This project centers on uncovering the inner workings of transformers within large language models (LLMs), with a specific focus on enhancing our comprehension and efficiency in training models like LLaMA.

#### Abstract

This project centers on uncovering the inner workings of transformers within large language models (LLMs), with a specific focus on enhancing our comprehension and efficiency in training models like LLaMA. In the initial phase of this fall quarter, we have delved into the fundamentals of transformer architectures and thoroughly analyzed their mathematical framework, starting with the forward pass and advancing to gradient calculations for backpropagation. Building on this foundation, we explored pretraining datasets commonly used for LLMs, especially LLaMA, and set up a functional training pipeline for the LLaMA 3.2-1b model, observing promising improvements in the loss curve. As we progress, our primary goal is to optimize Model FLOPs Utilization (MFU), targeting an increase from an initial 10% to 50% by quarter's end. Additionally, we aim to deepen our understanding of model scaling laws, specifically the impact of parameter selection and pretraining token volume on model performance. This exploration lays the groundwork for our continued work on LLM efficiency and scalability, setting us up to align future models with scaling law predictions effectively.

---

### Requirements

- **Python version**: 3.8 or higher
- **Jupyter Notebook**: Required to run the `.ipynb` file
- **Libraries**:
  - `torch` (PyTorch for model handling)
  - `transformers` (for handling the LLaMA model)
  - `datasets` (for loading and preprocessing datasets)
  - `peft` (for efficient model training)
  - `trl` (For SFTTrainer, a tool for supervised fine-tuning)
  - `huggingface_hub` (To access Hugging Face's model hub)
  - `wandb` (Weights & Biases for experiment tracking)

### Installation

1. Clone this repository:

  ```bash
  git clone https://github.com/nljumaoas/dsc180a.git
  ```
2. Navigate into the project directory:

  ```bash
  cd dsc180a
  ```
3. Install required dependencies:

  ```bash
  pip install -r requirements.txt
  ```

### Usage
Open the Jupyter Notebook:

  ```bash
  jupyter notebook llama1b.ipynb
  ```

Execute each cell sequentially in the notebook to preprocess data, configure the model, and initiate training. Ensure cells are run in order for reproducible results.

### Reproducibility
The results of this project are reproducible by following the order of cells in the provided notebook. The notebook is configured to yield consistent results when executed in sequence, enabling users to recreate the training process for the LLaMA 3.2-1B model.

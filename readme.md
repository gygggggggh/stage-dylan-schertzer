# SimCLR and Logistic Regression for Time Series Classification

This project implements SimCLR (Simple Framework for Contrastive Learning of Visual Representations) and Logistic Regression for time series classification tasks.
Two encoder backbones are available: ResNet and InceptionTime.

## Project Structure

- `data/`: Data directory
- `graph.py`: Display accuracy results from logs in a table
- `main.py`: Main script for training and testing
- `simCLR_resnet/`: SimCLR with ResNet backbone
- `simclr_IT/`: SimCLR with InceptionTime backbone
- `LR/`: Logistic Regression implementation
 
## Key Components

1. **SimCLRModuleRN**: PyTorch Lightning module for SimCLR with ResNet backbone
2. **SimCLRModuleIT**: PyTorch Lightning module for SimCLR with InceptionTime backbone
3. **NPYDataset** and **NPYDatasetAll**: Custom dataset classes for NumPy array data
4. **Logistic Regression**: GPU-accelerated implementation using cuML

## Data

- Training: x_train (40000, 100, 60, 12), y_train (40000,)
- Testing: x_test (10000, 100, 60, 12), y_test (10000,)

## Setup and Execution

1. Install dependencies: PyTorch, PyTorch Lightning, NumPy, scikit-learn, cuML, tqdm, matplotlib, pandas
2. Move data files in `data/` directory
3. Run: `python main.py`

## Configuration

Key parameters (adjust in respective scripts):
- Batch size: 1024
- Max epochs: 100
- Learning rate: 0.02
- N values for few-shot learning: [5, 10, 50, 100]

## Evaluation

- Few-shot learning performance (5, 10, 50, 100 samples per class)
- Majority voting over 100 time series per sample (5, 10, 50, 100 samples per class). Each sample is an agricultural parcel with 100 time series.

## Logging

- `testRN.log`: SimCLR+ResNet logs
- `testLR.log`: Logistic Regression logs
- `testIT.log`: SimCLR+InceptionTime logs

## Results

Check log files for accuracy results. Use `graph.py` for visualization in a table.

## Note

Designed for research on high-dimensional time series data. Requires significant computational resources, especially GPU memory.

## References

- [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939)
- [InceptionTime-Pytorch](https://github.com/TheMrGhostman/InceptionTime-Pytorch/blob/master/inception.py)
- [Lightly SSL ](https://docs.lightly.ai/self-supervised-learning/index.html)
## CNN Learning Project - README

This project aims to implement and train a Convolutional Neural Network (CNN) on the TinyImageNet30 and CIFAR10 datasets using the PyTorch library. The project also includes a comparison between the performance of the CNN model and a pretrained AlexNet model on the TinyImageNet30 dataset.

### Function Implementation

The project includes the following PyTorch classes:

- **Dataset** and **DataLoader** classes for loading and preprocessing the datasets.
- **Model** classes for a simple MLP model and a simple CNN model.

### Model Training

The CNN model is trained on the TinyImageNet30 dataset using strategies for tackling overfitting, such as data augmentation, dropout, and hyperparameter tuning (changing learning rate). Confusion matrices and ROC curves are generated to evaluate the performance of the model.

### Model Fine-tuning on CIFAR10 Dataset

The CNN model is fine-tuned on the CIFAR10 dataset. The model is initialized with pretrained weights and is fine-tuned with both frozen and unfrozen convolution layers. A comparison is made between the results of complete model retraining with pretrained weights and with frozen layers.

### Model Comparison

The project includes a comparison between the CNN model and a pretrained AlexNet model on the TinyImageNet30 dataset. The pretrained AlexNet model is fine-tuned until model convergence. The performance of both models is evaluated using loss graphs, confusion matrices, top-1 accuracy, and execution time.

### Interpretation of Results

Grad-CAM is used on both the CNN model and AlexNet to visualize and compare the results. The project includes a discussion of why the network predictions were correct or incorrect and what can be done to improve the results further.

Thank you for reviewing this project.
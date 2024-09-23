#Master thesis

**Self-Supervised Pre-Training For
Efficient Hierarchical Image
Classification: A Study On Unlabeled
Data Utilization**


[![Python](https://img.shields.io/badge/python-3.8+-%233776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-1.7%2B-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Implementation Strategy](#implementationstrategy)
- [What is Momentum Contrast Model?](#quickstart)
- [Stages in Azure](#stagesinazure)
- [Classifier](#classifier)
- [Code](#code)
- [Results & Evaluation](#Results-and-Evaluation)
- [Conclusion](#conclusion)
- [Future Directions and Enhancements](#future-work)



## Introduction 

This master’s thesis explores the potential of self-supervised pre-training for feature extraction in hierarchical image classification tasks, focusing on using large amounts of unlabeled images. Traditional approaches train separate models on limited data at each hierarchy level, leading to higher computational costs and potentially reduced performance.

We propose using a domain-specific backbone Convolutional Neural Network (CNN), pre-trained on a large dataset of unlabeled images, to produce high-quality embeddings, thereby reducing computational costs across all hierarchy levels. Our approach involves:
- Gathering and pre-processing all available unlabeled images.
- Training a self-supervised image embedding model, **Momentum Contrast V2 (MoCoV2)**.
- Fine-tuning the model for hierarchical image classification.

This study highlights the robustness of the trained embedding space, which is less sensitive to class distribution imbalances and can be used for image similarity comparisons to assist labeling. The method aims to enhance the efficiency and performance of hierarchical image classification by leveraging self-supervised pre-training.


## Features

- Self-supervised pre-training with MoCoV2.
- Robust against class distribution imbalances.
- Applicable for hierarchical image classification.
- Can fine-tune with labeled data using transfer learning.
- Efficient and scalable on large datasets.

## Implementation Strategy

The implementation phase involves:
1. Creating the Momentum Contrast (MoCo) model.
2. Modifying the existing classifier model.
3. Applying judicious transfer learning methodologies.


   <img width="800" alt="impl_con_1 (1)" src="https://github.com/user-attachments/assets/01f7b7bc-430d-493b-b896-8f2e10df79b4">

## What is Momentum Contrast Model?

The momentum contrast model created uses unlabelled data for learning meaningful
representations with the help of contrastive learning. It can be viewed as a dynamic dictionary
look-up process by integration of a queue and moving-averaged encoder. The dynamic
dictionary created on the fly with these encoders provides an evolving source of negative images
crucial for contrastive learning as well as providing the stability to model with a moving average
encoder enabling the model to learn intricate visual patterns without too much dependence on
labeled data. In rather simple terms, MoCo is a technique enabling a machine learning model to
glean insights from unlabeled images by constructing a visual pattern "dictionary" and
employing an intelligent approach to image comparison for learning. The figure shown below
illustrates the steps involved in creating the Moco V2 model and its detailed implementation is
discussed in model training. It can be easily explained with the flowchart below:   

![Moco_process (1)](https://github.com/user-attachments/assets/a7de4a47-5e0c-4837-94b2-956f5e0b3551)

- **Training of Momentum Contrast Model**
  - For this initial study, the **MoCo model** was trained using a relatively smaller neural network. The model uses **ResNet18** as the backbone for both query and key encoders, and their respective **MLP heads** for the projection layers. The total trainable parameters consist of both the backbone and projection head parameters, approximately **11.5M** parameters.
- **Trained Moco Model**:
  - After several iterations finally, MoCo model was trained on all unlabeled data in the cosmos_db database. It took almost 10 days to train on all the unlabeled data (300,000 images) :open_mouth:
### MoCo Model Components and Parameters

| Name                   | Type                | Params  |
|------------------------|---------------------|---------|
| **Backbone**            | Sequential          | 11.2M   |
| **Projection Head**     | MoCoProjectionHead  | 328K    |
| **Backbone Momentum**   | Sequential          | 11.2M   |
| **Projection Head Momentum** | MoCoProjectionHead  | 328K    |
| **Criterion**           | NTXentLoss          | 0       |
| **Trainable Params**    |                     | 11.5M   |
| **Non-trainable Params**|                     | 11.5M   |
| **Total Params**        |                     | 23.0M   |
| **Training Time**       |                     | 6 hrs   |

### Key Training Parameters

- **Input Image Size / Crop Size**: 100
- **Memory Bank Size**: 2048
- **Learning Rate**: 6e-2
- **Momentum**: 0.9
- **Weight Decay**: 5e-4
- **Number of Images**: 15,000
- **Epochs**: 100
- **Batch Size**: 128
- **GPU**: 1 x NVIDIA Tesla T4

The choice of parameter values is based on a variety of factors, including available computational resources, performances of previously known models, and insights from literature reviews. 
The most crucial metric to monitor during training is the behavior of **contrastive loss**. As seen in the figure below, the contrastive loss decreases steadily with an increase in epochs.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/cbde2ebc-6fa3-4107-a120-68f4c1324222">

## Stages In Azure:

1. **Connecting to Database** (`connect_db_MoCo.py`)
2. **Data Preparation** (`data_prep_moco.py`)
3. **Model Training**

To orchestrate the workflow, an `execute_moco.py` file is used, which manages:
- Pipeline parameters.
- Authentication.
- Configuration settings.
- Creation of a Docker container for setting up the Python environment for machine learning tasks.


1. **Connecting to Database** (`connect_db.py`)

The `connect_db.py` script plays a critical role in preparing the unlabeled dataset by connecting to the **Azure Cosmos Database**. It accepts command-line arguments and uses a custom query for **Cosmos DB** passed as a pipeline parameter from the execution file.

- **Functionality**:
  - Connects to the Azure Cosmos Database.
  - Executes custom queries to fetch the necessary image data.
  - Outputs a training dataset as a CSV file containing the image IDs of the unlabeled data.

This dataset is the foundation for subsequent steps, as it forms the base input for data preparation and model training.

---

2. **Data Preparation** (`data_prep_moco.py`)

The `data_prep_moco.py` script takes the output from the database connection step and prepares the data for training. This script:
  
- **Downloads and pre-processes images** from an **Azure Blob Storage** container.
- Contains the following key functions:
  - **`load_data`**: Reads the CSV file containing the image IDs produced by the database connection step.
  - **`get_image_size`**: Determines the required image size for model training based on the architecture of the PyTorch model.
  - **`get_image`**: Asynchronously downloads and resizes images from the Azure Blob Storage, preparing them for input into the model.

This data preparation step is essential to ensure that the images are correctly formatted and ready for training in the **MoCoV2** model. This can be explained by this flow chart:

![Data_prep (1)](https://github.com/user-attachments/assets/67042aac-d782-4b35-9b0f-697af3be6b79)


3. **Model Training**

The model training consists of the following steps:

   - **Data Loading and Transformation** (`datamodule_moco.py`):
     - Loads images and applies random transformations.
     -Generates query and key images for contrastive learning.
  
   - **Momentum Contrast Model** (`moco_model.py`):
        - Builds the MoCo model.
        - To understand the concept in detail please refer to section 5.1.3 in the thesis of kushal shah. The MoCo algorithm is implemented as follows
        - ![image-20240229-155842](https://github.com/user-attachments/assets/cb9f4186-d7f1-46a6-b9de-e465287c20cb)

   - **Encapsulating and Saving the Model** (`training_moco_pipeline.py`):
        - Configures training epochs, monitors training, and saves the trained model.
        - The MoCo model created is saved in the Azure model store which can be later used as the base model for the classifier.
        - The output folder contains a checkpoint file with details about the weights and parameters of each layer.

## Classifier:

The classifier model serves as the concluding component of the implementation phase. The learned representations from the pre-trained MoCo model are passed to this model with the help of transfer learning for the further classification process.
It consists of connected layers for the downstream classification task. The goal is to make the first-level classifier for the following categories:

![image](https://github.com/user-attachments/assets/93c58ddb-5623-407b-a62e-5c8d86c9fb47)
<img width="800" alt="image" src="https://github.com/user-attachments/assets/cb1f1490-1019-4ec1-a8f1-6318fb11b4a0">

The `connect_db.py` and `data_prep.py` steps work in the same way as they do for the MoCo model, but now we handle **labeled data** for the classifier model.

---

### Model Training for classifier

Model training consists of two key scripts:

1. **`evaluation_moco.py`**
2. **`training_training.py`**

The `evaluation_moco.py` script is primarily a modified version of `transfer_model.py`. Instead of building the model from scratch using the **Tim library**, it downloads the **MoCo V2** model from the **Azure model store**. The specific version of the model to be used is defined in the `build` function. The desired version of the MoCoV2 model is downloaded as a checkpoint file and saved in the output path of the training step.

#### Key Training Steps:
- Only the **query encoder** (without an MLP head) is attached to the linear classification head from the checkpoint. Other parameters from the checkpoint are not used.
- **PyTorch** does not save model hyperparameters in the checkpoint file. Therefore, the same hyperparameters used during pre-training are manually passed as arguments when instantiating the query model class.
- The **base model** (MoCoV2) has its weights frozen, and only the fully connected layers of the classification head are trained using the labeled data.

---

### Transfer Learning and Model Architecture

**Transfer learning** is crucial for successful representation learning. This process relies on a thorough understanding of the base model's architecture. The output dimensions or features of the last layer of the base model are used as the input for the first fully connected layer in the classifier.

The classification head includes:
- **One linear layer**
- **ReLU activation**
- **Dropout**
- **Another linear layer**

The architecture remains relatively simple and unchanged because this project is a **preliminary study**, and the number of classes to identify is small.

---

### Model Evaluation

In the evaluation phase:
- If the training process uses a **base model** (resnet-18) from the **PyTorch library**, it will evaluate the performance of a conventional model.
- If it uses the **MoCo V2** model (resnet-18 as backbone), it will evaluate the results with **pre-training**.

The **evaluation metrics** and other aspects of both models remain the same, ensuring a consistent comparison between the two approaches.

## Code
Code files for the above mentioned script cannot be provided in this repository as it is the property of Geberit Gmbh. However to test the working of MoCo a jupyter notebook is provided in the code section of this repository. The code is based on this notebook as well. It is developed by 
AI lighty and highly useful repository for self-supervised training research.

## Results and Evaluation

This thesis has successfully demonstrated the effectiveness of **MoCo** in image classification. Through experimental analysis, the study confirms that **MoCo** as a **self-supervised learning** approach can efficiently learn meaningful representations from unlabeled data, outperforming traditional supervised learning methods in several key performance metrics.

### Key Findings:
- **Enhanced Generalization**
- **Improved F1 Score**
- **Better Accuracy, Precision, and Recall**

These results, as summarized in the table below, show that MoCo-based self-supervised learning, especially with **oversampling**, achieves superior performance compared to traditional supervised learning approaches.

| Method                     | Precision | Recall  | F-1 Score | Accuracy |
|----------------------------|-----------|---------|-----------|----------|
| **Self-supervised**         | 0.72      | 0.73    | 0.72      | 0.82     |
| **Self-supervised with oversampling** | 0.74      | 0.75    | 0.74      | 0.86     |
| **Supervised Learning**     | 0.54      | 0.50    | 0.46      | 0.78     |

For a more detailed analysis, please refer to **Chapter 6** of the thesis.

---

### Conclusion:
The experimental results demonstrate that **MoCo's self-supervised learning** approach is not only more robust in terms of generalization but also outperforms supervised learning across key performance indicators such as F1 score, accuracy, precision, and recall.

## Future Directions and Enhancements

### Improved GPU Capabilities and Labeled Data Quality
In the future, **enhanced GPU capabilities** and better quality of labeled data could significantly aid in improving model performance. One potential approach for optimizing computation could be the use of **data parallel processing techniques**.

### Scaling Up Training with Horovod
By scaling up the training script using the [Horovod](https://horovod.readthedocs.io/en/stable/summary_include.html#why-horovod) package, the training process can be distributed across a **cluster of nodes**. This parallelization technique could greatly reduce the time required for training large datasets. For example:
- Training **300,000 images** on a single GPU node may take 10 days.
- By utilizing **5 nodes**, this time can be potentially reduced to **2 days**, with a constant cost, dependent on the number of nodes employed.

### Fine-Tuning and Enhancing MoCo
Exploring the fine-tuning of both the **classification head** and **MoCo pipeline** could lead to better results. Some key ideas include:
- **Incorporating Batch Normalization**: This could help stabilize and accelerate the training process.
- **Using Wider Networks**: Findings from SimCLR research suggest that wider networks improve contrastive loss, so using broader networks as backbones in future experiments might yield better performance.

### Integration with Hierarchical Classifiers
An immediate future step involves integrating the MoCo pipeline into **hierarchical classifiers**. Running inferences across different levels of the hierarchy will allow further assessment of the model’s performance.


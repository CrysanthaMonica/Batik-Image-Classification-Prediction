# Batik Motif Image Classification using Deep Learning

## Overview
This project focuses on building and evaluating deep learning models to classify Indonesian batik motifs. Indonesian batik is globally recognized as a valuable cultural heritage, and automatic recognition of batik patterns can aid both cultural preservation and creative industries.

Using various models, including custom CNNs, MobileNetV2, and EfficientNetB2, this project attempts to classify batik motifs into several categories. The goal is to identify the best-performing model based on accuracy and other metrics.

## Table of Contents
- [Introduction](#introduction)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Challenges Faced](#challenges-faced)
- [Results](#results)
- [Further Improvements](#further-improvements)
- [Conclusion](#conclusion)

## Introduction
Batik patterns are intricate and diverse, requiring expertise to recognize manually. With advancements in technology, we explore how deep learning can assist in the automatic classification of batik patterns. This project aims to classify several batik motifs, including Batik Kawung, Parang, and Mega Mendung, among others.

## Models Used
Three deep learning models were used for the batik classification task:
1. **Custom CNN Model**: Built from scratch, used as a baseline.
2. **MobileNetV2**: A pre-trained model, further fine-tuned with our batik dataset.
3. **EfficientNetB2**: A highly efficient model designed for classification tasks with advanced parameter tuning and data augmentation.

Each model was evaluated on several metrics, including accuracy, precision, recall, and F1-score.

## Dataset
The dataset consists of images of different batik motifs, obtained from [Kaggle](https://www.kaggle.com/dionisiusdh/indonesian-batik-motifs). Key characteristics of the dataset:
- **Variability**: Images differ in size, resolution, and quality.
- **Categories**: The dataset contains batik patterns like Kawung, Parang, and Mega Mendung, with some classes having more samples than others.
- **Preprocessing**: Images were resized to 128x128 pixels, normalized, and augmented to improve model robustness.

## Methodology
1. **Data Preparation**: Cleaned the dataset by removing irrelevant images and resizing all images to a uniform size.
2. **Model Development**: Built three models with different complexities, using regularization techniques like dropout and batch normalization to prevent overfitting.
3. **Training and Evaluation**: Models were trained using the Adam optimizer, with validation splits and early stopping to avoid overfitting. The performance of each model was evaluated using confusion matrices and key metrics such as precision, recall, and F1-score.

## Challenges Faced

### 1. Dataset Quality and Variability
- **Image Quality**: The dataset included images with varying quality, from high-resolution images to blurry and low-resolution ones, making it difficult for the model to learn meaningful features. These noisy images added complexity to the training process.
  
- **Resolution and Size Variability**: The batik images had varying resolutions, requiring resizing to a uniform 128x128 pixels. While this helped with consistency, resizing could also result in the loss of fine details crucial for pattern recognition.

- **Class Imbalance**: Some categories had significantly more samples than others, such as **Batik Parang** having more images compared to **Batik Sogan**. This imbalance led to biased predictions, where the model was better at classifying well-represented categories while struggling with under-represented ones.

- **Diverse Image Sources**: The dataset included images from different sources (Google Images, Instagram, local databases), resulting in inconsistent lighting, orientation, and backgrounds. Data augmentation like flipping, zooming, and lighting adjustments was applied to help improve the model's generalization.

### 2. Overfitting
- **Custom CNN Overfitting**: The initial custom CNN model showed clear signs of overfitting, as the training loss decreased while the validation loss fluctuated and increased. This indicated that the model was memorizing the training data instead of learning to generalize well.

- **MobileNetV2 Overfitting**: Even with a pre-trained MobileNetV2 model, overfitting persisted, especially with under-represented categories like **Batik Sogan**. To address this, dropout and batch normalization were added, though the results still left room for improvement.

### 3. Model Generalization
- **Data Augmentation**: To prevent overfitting and improve generalization, various augmentation techniques were applied, such as rotation, flipping, and brightness adjustment. While this improved overall performance, tuning the augmentation techniques took significant effort to ensure they didn’t degrade performance on certain categories.

- **Class-specific Challenges**: Certain batik motifs like **Batik Sogan** posed particular challenges, as the model consistently struggled with lower precision and recall for this class due to the limited number of samples and subtle differences between motifs.

### 4. Computational Constraints
- **Training Time**: The most advanced model, EfficientNetB2, required significant computational resources and time due to its large number of parameters and complex architecture. Running multiple experiments with data augmentation made the training process time-intensive, particularly on limited local resources.

## Results
The performance of the models was as follows:
- **Custom CNN Model**: Achieved an accuracy of 35%, showing overfitting with poor generalization.
- **MobileNetV2**: Improved accuracy to 42%, but struggled with some classes, especially Batik Sogan.
- **EfficientNetB2**: Achieved the best accuracy of 69%, with balanced precision and recall across most categories.

| Model           | Accuracy | Precision (Best Class) | Recall (Best Class) | F1-Score (Best Class) |
|-----------------|----------|------------------------|---------------------|-----------------------|
| Custom CNN      | 35%      | Batik Betawi (0.44)     | Batik Sogan (0.38)  | Batik Betawi (0.44)   |
| MobileNetV2     | 42%      | Batik Betawi (0.75)     | Batik Sogan (0.89)  | Batik Sogan (0.53)    |
| EfficientNetB2  | 69%      | Batik Betawi (1.00)     | Batik Bali (0.62)   | Batik Betawi (0.80)   |

## Further Improvements

1. **Addressing Class Imbalance**:
   - Use SMOTE to generate synthetic data for under-represented classes like Batik Sogan.
   - Adjust class weights to prioritize minority classes and balance the dataset.

2. **Data Augmentation and Preprocessing**:
   - Apply advanced augmentation techniques such as elastic transformations and cutout augmentation to enrich the dataset.
   - Implement super-resolution techniques to improve low-quality images, enhancing the model's learning capability.

3. **Model Generalization and Overfitting**:
   - Use L2 regularization and k-fold cross-validation to prevent overfitting and improve model generalization.
   - Combine models via ensemble techniques like stacking for more robust classification results.

4. **Hyperparameter Tuning**:
   - Perform grid search or Bayesian optimization to fine-tune parameters like learning rate, batch size, and dropout rates.
   - Implement learning rate schedulers to dynamically adjust the learning rate during training.

5. **Transfer Learning Enhancement**:
   - Fine-tune deeper layers of pre-trained models (MobileNetV2, EfficientNetB2) to extract more relevant features.
   - Explore transfer learning from models trained on fashion or textile-specific datasets to better align with the batik domain.

6. **Model Complexity and Customization**:
   - Develop deeper, more customized CNN architectures for texture recognition, tailored to batik motifs.
   - Integrate attention mechanisms to focus on key parts of batik patterns for improved classification.

7. **Leveraging External Datasets**:
   - Expand the dataset by incorporating more batik motifs from additional sources to increase training data.
   - Enhance data annotations to provide more metadata, improving the richness and depth of the training set.

8. **Evaluation and Interpretation**:
   - Use Grad-CAM to visualize the model's focus during classification and better understand misclassifications.
   - Focus on precision-recall metrics to get a clearer sense of performance, especially for under-represented classes.

## Conclusion
EfficientNetB2 was the most effective model, achieving the highest accuracy and generalizing well across different batik patterns. However, there is still room for improvement, especially in recognizing lesser-known patterns like Batik Sogan.

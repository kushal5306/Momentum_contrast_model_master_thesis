
## Abstract
This master’s thesis explores the potential of self-supervised pre-training for feature extraction in hierarchical image classification tasks, focusing on using large amounts of unlabeled images. 
Traditional approaches train separate models on limited data at each hierarchy level, leading to higher computational costs and potentially reduced performance.

We propose using a domain-specific backbone Convolutional Neural Network (CNN), pre-trained on a large dataset of unlabeled images, to produce high-quality embeddings, thereby reducing computational costs across all hierarchy levels. Our approach involves:
- Gathering and pre-processing all available unlabeled images.
- Training a self-supervised image embedding model, **Momentum Contrast V2 (MoCoV2)**.
- Fine-tuning the model for hierarchical image classification.

This study highlights the robustness of the trained embedding space, which is less sensitive to class distribution imbalances and can be used for image similarity comparisons to assist labeling. 
The method aims to enhance the efficiency and performance of hierarchical image classification by leveraging self-supervised pre-training.

---

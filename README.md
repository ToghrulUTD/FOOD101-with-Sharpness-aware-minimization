# Project description
In this project, we fine-tune transfer learning model to classify pictured dishes into 101 classes. The main objective is to beat the performance of [DeepFood](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment) paper using fine-tuning and sharpness minimization techniques. See the data at [`food101`](https://www.tensorflow.org/datasets/catalog/food101) (dataset comprised of 75000 train and 25000 test images). 

T

he notebook convers 5 different experiments involving transfer learning, fine-tuning, data augmentation, sharpness aware minimization, and techniques to reduce uncertainity. Below are the steps of the project.

  1. Use TensorFlow Datasets ([tfds.laod](https://www.tensorflow.org/datasets/api_docs/python/tfds/load)) to download and explore data
  2. Creating preprocessing function for our data
  3. Create data pipeline with tf.data (making our datasets run fast with [tf.data API](https://www.tensorflow.org/guide/data_performance))
  4.  Define TensorBoard and Model Checkpoint, Early Stopping, Learning Rate callbacks
  5. Setting up [Mixed Precision](https://www.tensorflow.org/guide/mixed_precision) policy for fast training 
  6. Building and customizing RESNET model.
  7. Fine-tuning the feature extractor - experiment models with and without the ideas outlined in [MCdropout](https://arxiv.org/abs/1506.02142) and [Sharpness Aware Minimization](https://arxiv.org/abs/2010.01412).
    * Stochastic predictions with MCdropout
    * Sharpness aware minimization for better generalization
  8. Tracking experiment results on TensorBoard

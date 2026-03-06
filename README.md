# satellite-images-segmentation-using-unet

Project on Semantic Segmentation of Satellite Images. This work was conducted as part of the _Normes et Nouvelles Générations de Compression_ class given by Mrs. Dorsaf SEBAI, Eng. Dr. HDR. in Computer Science during my second year of master's studies (Data Science Master @ENSI, January 2026).

## Project overview
This project aims to train a deep learning model that will be capable to classify different types of land cover from satellite images. 

## Data

### DeepGlobe Land Cover Classification Dataset

The [DeepGlobe Land Cover Classification Challenge Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset/), from 2018, was the first public dataset to offer high-resolution, sub-meter saletille imagery with a focus on rural areas. The images have a resolution of 50 cm per pixel and contain three channels **(Red, Green, and Blue).** 

### Key facts
- **Domain:** High-resolution satellite land-cover segmentation.
- **Images:** 1,146 RGB tiles (2448x2448 pixels).
- **Resolution:** 0.5 m per pixel.
- **Splits:** 803 images for the training set, 171 images for the validation set, and 172 images for the test set.

### Data decription

Each satelitte image is paired with a mask image for land cover annotation. The mask is an RGB image with **7 classes** that follow the Anderson Classification system:

- **Urban land:** Man-made, built-up areas with human artefacts.
- **Agriculture land:** Farms, any planned (i.e., regular) plantation, cropland, and ornamental horticultural areas.
- **Rangeland:** Any non-forest, non-farm, green land, grass.
- **Forest land:** Any land with at least 20% tree crown density plus clear cuts.
- **Water:** Rivers, oceans, lakes, wetland, ponds.
- **Barren land:** Mountain, rock, dessert, beach, land with no vegetation.
- **Unknown:** Clouds and others.

Insert an image.

Due to the variety of land cover types and the density of annotations, this dataset is more challenging than many of its predecessors. For more information, you can refer to the original [paper](https://arxiv.org/abs/1805.06561).

## Model
In this project, we _fine-tuned_ a **U-Net** with a **ResNet18** encoder pretained on **ImageNet dataset**.

We work with some common tricks, including:
1. Working with smaller **patches (256x256 pixels)** extracted from the large satellite images.
2. Using **minibatch learning (32)**, where we show the model a small number of patches at a time.
3. Running the optimisation for several **epochs (30)**, that is, we repeatedly show the model the entire dataset during training.

We used these tricks to efficiently train our model (without consuming all our memory).

## Requirements
We used the popular deep learning framework **PyTorch** for the models and the training as well as**PyTorch Lightning**, which is useful library containing many functions to make training models easier (especially on GPUs). Finally, we used **Segmentation Models PyTorch**, which makes it easy to automatically download and set-up pretrained segmentation models for PyTorch.

So you will have to run this following commands before starting the notebook.

```markdown
```python
try: 
    import pytorch_lightning as pl
except ImportError:
    !pip install -U pytorch-lightning -q
    
try:
    import segmentation_models_pytorch as smp
except ImportError:
    !pip install -U segmentation-models-pytorch -q
    
try:
    import albumentations as A
except ImportError:
    !pip install -U albumentationsx -q
```
## Training pipeline
The better summary of our training pipeline is :
1. Load satellite images.
2. Load corresponding maks.
3. Train the unet model.

## Evaluation
We used the **Dice loss** to evaluate our model. This function is based on the [Dice-Sørensen](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient) coefficient and is very common in image segmentation tasks because it is more robust to the class imbalances than other common loss functions.

We also used the **Intersection over Union (IoU)** metric, which is also a common metric used for semantic segmentatin because it penalize false positives.

Other metrics, like **Precision, Recall, Accuracy, and F1-score** are also used.

## Results

## References
1. Demir, K. Koperski, D. Lindenbaum, G. Pang, J. Huang, S. Basu, F. Hughes, D. Tuia et R. Raskar, **[DeepGlobe 2018 : A Challenge to Parse the Earth Through Satellite Images](https://arxiv.org/abs/1805.06561)**, in _proceeding of The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops_, 2018.
2. R. Olaf, F. Philipp et B. Thomas, **[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)**, arXiv, 2015.
   
## Note
Some parts of this work were inspired by this [Kaggle notebook](https://www.kaggle.com/code/kstensbo/deep-learning-for-land-cover-classification/notebook).



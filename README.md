﻿# 3D MRI Transformations

<p align="center">
  <img src="./data/Brain_Brodmann_blend.gif" alt="Your Image" width="400" height="250">
</p>

<h3 align = "center">How to extract information from 3D MRI for CNN?</h3>
Extracting useful information for 3D MRI scan to train efficient CNN models can be tricky. Why?

1. 3D CNN are difficult to train due to high computation times. And retraining models can be exhausting.
2. To train 3D CNN, amount of labelled-data required is very high and is not suitable if the sample size is small. 

<h3 align = "center"> Some transformers.. </h3>
<p><a href="https://github.com/pradhanhitesh/3D-MRI-Transformations/blob/main/notebooks/3D-Mean-Transformations.ipynb">3D Mean Transformation</a></p>

<p align="center">
  <img src="./data/mean-transformation.png" alt="Your Image" width="600" height="450">
</p>

<p><a href="https://github.com/pradhanhitesh/3D-MRI-Transformations/blob/main/notebooks/3D-DWT-Transformation.ipynb">3D Discrete Wavelet Transformation</a></p>

1. Chaplot, S., Patnaik, L. M., & Jagannathan, N. R. (2006). Classification of magnetic resonance brain images using wavelets as input to support vector machine and neural network. Biomedical signal processing and control, 1(1), 86-92.

<p align="center">
  <img src="./data/dwt-transfomation.png" alt="Your Image" width="800" height="200">
</p>

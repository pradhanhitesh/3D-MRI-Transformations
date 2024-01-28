# 3D MRI Transformations

<p align="center">
  <img src="./data/Brain_Brodmann_blend.gif" alt="Your Image" width="400" height="250">
</p>

<h3 align = "center">How to extract information from 3D MRI for CNN?</h3>
Extracting useful information for 3D MRI scan to train efficient CNN models can be tricky. Why?

1. 3D CNN are difficult to train due to high computation times. And retraining models can be exhausting.
2. To train 3D CNN, amount of labelled-data required is very high and is not suitable if the sample size is small. 

<h3 align = "center"> Some transformers.. </h3>
<p>1 . 3D Mean Transformation</p>
<p align="center">
  <img src="./data/mean-transformation.png" alt="Your Image" width="600" height="450">
</p>

<p>2 . 3D Discrete Wavelet Transformation</p>
<p align="center">
  <img src="./data/dwt-transfomation.png" alt="Your Image" width="800" height="200">
</p>

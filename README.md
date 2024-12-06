<h1 align="center"> Enhancing Video Text Detection through Dual-Dimensional Attention and Swin Transformer</h1>

# Overview
Video text detection poses significant challenges due to the dynamic nature of video content. To address these challenges, we propose an advanced framework that integrates a dual-dimensional attention mechanism and the Swin Transformer.By leveraging the Swin Transformer’s hierarchical structure and ability to capture both local and global dependencies, our model significantly improves feature extraction efficiency. The dual-dimensional attention mechanism further enhances detection, matching, and tracking capabilities by focusing on salient channel and spatial features. Additionally, we replace the conventional multi-layer perceptron with Kolmogorov-Arnold Networks (KAN) to improve the precision of text instance matching.

# Key Features
Experimental evaluations on the ICDAR15-video and DSText datasets demonstrate that our approach outperforms state-of-the-art models, achieving substantial gains in accuracy. Our work highlights the effectiveness of combining the Swin Transformer and dualdimensional attention for robust video text detection.

# Installation

```
Python_3.8 + PyTorch_1.9.0 + CUDA_11.1 + Detectron2_v0.6
pip install -r requirements.txt
```

# Dataset

Videos in [ICDAR15-video](https://rrc.cvc.uab.es/?ch=3&com=downloads) and [DSText](https://rrc.cvc.uab.es/?ch=22&com=downloads).

# TRAIN

ICDAR15

```python
python train_net.py --num-gpus 1 --config-file configs/GoMatching_ICDAR15.yaml
```

DSText

```python
python train_net.py --num-gpus 1 --config-file configs/GoMatching_DSText.yaml
```

# Evaluation

**ICDAR15**

```python
python eval.py --config-file configs/GoMatching_ICDAR15.yaml --input ./datasets/ICDAR15/frame_test/ --output output/icdar15 --opts MODEL.WEIGHTS trained_models/ICDAR15/xxx.pth
```

**DSText**

```python
python eval.py --config-file configs/GoMatching_DSText.yaml --input ./datasets/DSText/frame_test/ --output output/dstext --opts MODEL.WEIGHTS trained_models/DSText/xxx.pth
```

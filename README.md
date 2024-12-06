<h1 align="center"> Enhancing Video Text Detection through Dual-Dimensional Attention and Swin Transformer</h1>
Continuously updated

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

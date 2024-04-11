# U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation [ISPRS Annals, 2024] 

This repo is the official implementation of [U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation](https://link.springer.com/article/10.1007/s41064-024-00280-4), which was accepted for publication in the ISPRS Annals and will be presented at ISPRS Technical Commission II Symposium in Las Vegas, June 11-14, 2024.

## Motivation
In this work, we present a novel Uncertainty-aware Cross-Entropy loss (U-CE) that incorporates dynamic predictive uncertainties into the training process by pixel-wise weighting of the well-known cross-entropy loss (CE). With U-CE, we manage to train models that not only improve their segmentation performance but also provide meaningful uncertainties after training.

## Requirements
### Environment
First, pull the following Docker Image:
```shell
docker pull nvcr.io/nvidia/pytorch:23.03-py3
```

Second, start the Docker Container with the Docker Image:
```shell
docker run --name <container_name> [--docker_options] --volume /your_path_to/U-CE:/workspace <image_name> bash
docker attach <container_name>
```

Finally, install additional requirements:
```shell
pip install -r requirements.txt
```

## Acknowledgement
U-CE is partly built upon [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch). We want to thank [Pavel Iakubovskii](https://github.com/qubvel) for building, publishing, and maintaining it.

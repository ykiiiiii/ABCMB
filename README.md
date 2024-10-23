# ABCMB: Deep Delensing Assisted Likelihood-Free Inference from CMB Polarization Maps


ABCMB is a deep learning framework for CMB B-mode polarization delensing and tensor-to-scalar ratio inference. This repository contains the implementation of the method described in our [paper](https://www.arxiv.org/abs/2407.10013).

## Installation


### Environment Setup

Create a conda environment using our provided requirements:

```bash
# Create and activate environment
conda create --name ABCMB --file requirements.txt
conda activate ABCMB
```


## Getting Started

### Download Pre-trained Models

Download our pre-trained model [weights](https://drive.google.com/file/d/1o9K4-QeXbYrFvCUUfLK-1GOQft-N9jim/view?usp=drive_link)


### Data Generation

Generate simulation data for training or inference:

```bash
# Generate validation data
python lensing_sim.py --output_path 'output/val/all_map/'

# Generate training data
python lensing_sim.py --output_path 'output/train/all_map/'
```


### Running Inference

For detailed examples of delensing Q/U maps to obtain B-mode maps and computing power spectra, see our [inference notebook](inference.ipynb).




## Citation 
If you consider our codes and datasets useful, please cite:
```
@article{yi2024ab,
  title={AB $$\backslash$mathbb $\{$C$\}$ $ MB: Deep Delensing Assisted Likelihood-Free Inference from CMB Polarization Maps},
  author={Yi, Kai and Fan, Yanan and Hamann, Jan and Li{\`o}, Pietro and Wang, Yuguang},
  journal={arXiv preprint arXiv:2407.10013},
  year={2024}
}
```

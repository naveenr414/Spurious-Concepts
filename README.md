# Do Concept Bottleneck Models Respect Localities? 

![Paper Overview](img/Locality_Abstract_CBM.png)

This repository contains the implementation for the paper ["Do Concept Bottleneck Models Respect Localities?"](https://arxiv.org/abs/2401.01259), published in TMLR 2025.

This work was done by [Naveen Raman](https://naveenraman.com/), [Mateo Espinosa](https://hairyballtheorem.com/), [Juyeon Heo](https://sites.google.com/view/juyeonheo/), and [Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/). 

#### TL;DR
Concept-based methods require accurate concept predictors, yet the faithfulness of existing concept predictors to their underlying concepts is unclear.
To better understand this, we investigate the faithfulness of Concept Bottleneck Models (CBMs), a popular family of concept-based architectures, by looking at whether they respect "localities" in datasets. 
Localities involve using only relevant features when predicting concepts. 
When localities are not considered, concepts may be predicted based on spuriously correlated features, degrading performance and robustness. 
This work examines how CBM predictions change when perturbing model inputs, and reveals that CBMs may not capture localities, even when independent concepts are localised to non-overlapping feature subsets. 
Our empirical and theoretical results demonstrate that datasets with correlated concepts may lead to accurate but uninterpretable models that fail to learn localities. 

## Citation
If you use our code for your research, please cite this as
```
@article{
raman2025do,
title={Do Concept Bottleneck Models Respect Localities?},
author={Naveen Janaki Raman and Mateo Espinosa Zarlenga and Juyeon Heo and Mateja Jamnik},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=4mCkRbUXOf},
note={}
}
```

## Installation and Datasets
### Installation
To install dependencies and setup folders, run the following
```
$ conda env create --file environment.yaml
$ pip install -e .
$ bash scripts/bash_scripts/create_folders.sh
```

### Datasets
To setup datasets, download and unzip the dataset skeleton from <a href="https://drive.google.com/file/d/1Z_exiW6SRd7rSq0YRgMLNPO_cMFzRLjE/view?usp=sharing">here</a>. 
This consists of the metadata for CUB, COCO, and synthetic datasets, and the images for the synthetic datasets. 
The images for CUB are located at <a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">here</a>; move ```CUB_200_2011/images``` to ```datasets/CUB/images```. 
The images for COCO are located at <a href="https://cocodataset.org/#home">here</a>; use the 2014 train and validation datasets and move all images in ```train2014``` and ```val2014``` to ```datasets/coco/images```.  

### Models
To download trained versions of each of our models, use the link <a href="https://cmu.box.com/s/usy4h34vp51bsoeafot7jf79i9uc8cg5">here</a>. 

Similarly, to download our results, use the link <a href="https://cmu.box.com/s/8mvnvpwzfrowgbs53ua2biyn7t3wc6cu">here</a>. 

## Directory Structure
The repository consists of four components: 
1. Code to train different combinations of models is located in ```scripts/bash_scripts/train_models```. Each bash script uses ```locality/cbm_variants/ConceptBottleneck/train_cbm.py``` to train CBMs. For example, ```bash scripts/bash_scripts/train_models/synthetic/train_synthetic.sh``` trains all the synthetic dataset models. Code for training ProbCBM and CEM models can be found in the ```locality/cbm_variants/prob-cbm``` folder. 
2. Notebooks in ```scripts/notebooks``` are used to conduct analyses of models according to the leakage, masking, and intervention metrics. Each notebook also has a corresponding python .py file, which is used when running larger-scale analyses. 
3. Scripts in ```scripts/bash_scripts/analyze_models``` use the python files in ```scripts/notebooks``` to conduct larger-scale analyses. For example, running ```bash scripts/bash_scripts/analyze_models/synthetic_run_synthetic_leakage.sh``` analyzes synthetic models according to the leakage metric. All results are written to the ```results``` folder. 
4. Plots can be found in the ```scripts/notebooks/Plotting.ipynb``` folder. 
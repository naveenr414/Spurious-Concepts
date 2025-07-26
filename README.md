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
To install dependencies, run the following
```
$ conda env create --file environment.yaml
$ python setup.py install
```

### Datasets
To setup datasets, download and unzip the dataset skeleton from <a href="https://drive.google.com/file/d/1Z_exiW6SRd7rSq0YRgMLNPO_cMFzRLjE/view?usp=sharing">here</a>. 
This consists of the metadata for CUB, COCO, and synthetic datasets, and the images for the synthetic datasets. 
The images for CUB are located at <a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">here</a>; move ```CUB_200_2011/images``` to ```datasets/CUB/images```. 
The images for COCO are located at <a href="https://cocodataset.org/#home">here</a>; use the 2014 train and validation datasets and move all images in ```train2014``` and ```val2014``` to ```datasets/coco/images```.  

### Models
To download trained versions of each of our models, use the link <a href="https://cmu.box.com/s/usy4h34vp51bsoeafot7jf79i9uc8cg5">here</a>. 

Similarly, to download our results, use the link <a href="https://cmu.box.com/s/8mvnvpwzfrowgbs53ua2biyn7t3wc6cu">here</a>. 

## Synthetic Experiments
Our synthetic experiments are largely built off of our Jupyter notebook: ```scripts/notebooks/Synthetic Experiments.ipynb```. 
This notebook first loads in a synthetic model, and evaluates it according to the locality leakage metric. 

These experiments can be automatically run through scripts titled: ```scripts/bash_scripts/analyze_models/run_synthetic...```

Training models is done through: ```ConceptBottleneck/train_cbm.py```. 

Scripts to train models can be found: ```scripts/bash_scripts/train_models.sh```

All trained models will be stored in the ```models/``` folder. 

## Locality Masking
The notebook corresponding to our locality masking experiments can be found at: ```scripts/notebooks/CUB Analysis.ipynb``` and ```scripts/notebooks/Coco Analysis.ipynb```. 

The corresponding bash scripts: ```scripts/bash_scripts/analyze_models/run_cub.sh``` and ```scripts/bash_scripts/analyze_models/run_coco.sh```

Models for CUB and COCO can be trained through the scripts: ```scripts/bash_scripts/train_models/train_cub.sh``` and ```scripts/bash_scripts/train_models/train_coco.sh```

## Locality Intervention
The notebook corresponding to our locality intervention experiments can be found at: ```scripts/notebooks/Synthetic Correlation.ipynb```

The corresponding bash script is: ```scripts/bash_scripts/analyze_models/run_synthetic_correlation.sh```. 

## Mitigating Locality-Based Issues
We construct experiments to analyze the impact of different methods upon improving the locality-related properties of CBMs. 
1. **Pruning**: ```scripts/notebooks/CUB Pruning.ipynb``` and ```scripts/notebooks/Coco Pruning.ipynb```
2. **Label Free CBMs**: ```scripts/notebooks/CUB Label Free.ipynb```
3. **Training Modifications**: ```scripts/notebooks/CUB Analysis.ipynb``` and ```scripts/notebooks/Coco Analysis.ipynb```
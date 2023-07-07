# Spurious Correlation Project (R255 Final Project)
Can we detect known and unknown correlations using Concept Bottleneck Models? 

## Training Models
To set up a sample dataset, run the following commmand
```python
python src/dataset.py -s synthetic_simple -n 100
```
In this case, `synthetic_simple` is the name of the dataset, with 100 random image data points. 

Then to train a CBM model based on this dataset, run the following commmand
```bash
bash train_cbm.sh synthetic_simple joint 1 42 1
```
where the arguments refer to the number of attributes, seed, and epochs. 
The resulting model is saved to `results/synthetic_simple/joint`. 

# Introduction

Accurate metabolite annotation remains a major bottleneck in Spatial Metabolomics, particularly in MALDI-MSI workflows where orthogonal descriptors (e.g., retention time, collision cross-section) are absent. Typically, less than 10% of detected features are successfully identified in untargeted analyses.
DeepGCN-FP is a computational framework designed to enhance metabolite annotation confidence by leveraging adduct ion formation patterns as an orthogonal constraint. Our approach combines Deep Graph Convolutional Networks (https://github.com/kangqiyue/DeepGCN-RT) with Transfer Learning to extract informative molecular fingerprints, which are then used to predict the propensity of metabolites to form specific adduct ions under fixed matrix conditions.



# Build conda environment

Use the provided 

```
environment.yml
```

 file and conda to create an environment capable of running the project code.
All code was developed using Python 3.8

# Usage and examples

The code for transfer learning , predictors , visualization ，importance analysiscan ，annotation can be run by executing the following command

```
notebooks/pipline.ipynb
```

 in notebooks directory

 


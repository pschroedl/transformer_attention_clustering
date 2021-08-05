# transformer attention clustering

![pipeline](./img/pipeline_w_images.png)
## Overview:
A pre-trained Bert for Question-Answering Model (huggingface) is fine-tuned on the Squad2 Dataset, using a max-token size of 384.

Each example, when evaluated on the fine-tuned model, produces a 12x12 matrix of attentions, each consisting of attention weights mapping each token to all tokens ( 384x384 ).

A Trained Barlow twins model ( with the last, linear classifier layer removed ) is fed a scaled version (0-255) of each of the layer and heads (12x12) attention weights to produce vectors of 2048 32 bit floats.

Using the 131,944 squad2 training set examples downloaded from [SQuAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/) [(direct file link)](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json) the resulting dataset is over 18M data points and takes up a little more than 400GB as 26 separate CSVs ~16gb apiece.  The dataset is public online at https://storage.googleapis.com/representations_central/ under the folder /datasets/7-18-21/.

Initially using Intel Python enhanced ScikitLearn is investigated to scale, Dask + cuML were settled on for their extraordinary performance.  Dimensionality reduction and clustering algorithms distributed across multiple GPUs is performed.

Resultant cluster labels are then correllated with their position in the 12 x 12 layer/head matrix corresponding to attentions  created when evaluating a single Squad2 QA example.

Clustering using DBScan largely produced a single dominant cluster favoring the earlier layers of the bert architecture.  Noise was primarily made up of later layers.  This suggests that variance increases as the input data travels through subsequent layers.  Additional, much smaller clusters were also found in which data points corresponded to a single or few common heads within groupings of adjacent examples in the squad2 dataset.  The squad2 examples were sampled sequentially so neighboring examples share the same provided context with differing questions.


An additional clustering step was also performed, taking the largest cluster and re-running dbscan.  This produced a larger number of 'satellite' clusters containing a much more evenly distributed group of heads.





A Springboard Capstone Project: [Proposal](proposal.md)

## Contents:

• bert_fine_tuning/

[run_squad.py](https://github.com/pschroedl/transformer_attention_clustering/blob/main/bert_fine_tuning/run_squad.py) - modified version of huggingface provided script  
[fine_tuning.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/bert_fine_tuning/fine-tune.ipynb) - a wrapper around the cli call to run_squad.py  
[fine_tune.sh](https://github.com/pschroedl/transformer_attention_clustering/blob/main/bert_fine_tuning/fine-tune.sh) - script calling run_sqad for fine-tuning including cli parameters  
[eval.sh](https://github.com/pschroedl/transformer_attention_clustering/blob/main/bert_fine_tuning/eval.sh) - script calling run_squad for evaluation including cli parameters  
[eval.log](https://github.com/pschroedl/transformer_attention_clustering/blob/main/bert_fine_tuning/eval.log) - evaluation output for our model  

• exploration/

(These exploratory files are included, as well as the separate extract/transform steps in protoyping the pipeline, in order to provide some insight into the mechanisms of transformation used)

[attention_exploration.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/exploration/attention_exploration.ipynb) - visualizing various huggingface Transformer outputs  
[transform_prototyping.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/exploration/transform_prototyping.ipynb) - prototyping transformation to representations
[transform_representation.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/exploration/transform_representation_exploration.ipynb) - more exploration of process of data transformation  

• pipeline/

![pipeline](/img/pipeline_files.png)

[extract_attentions.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/exploration/extract_attentions.ipynb) - modified run_squad.py to output raw attention matrices
[transform_attentions.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/pipeline/transform_attentions.ipynb) - prototyping raw attentions for inital clustering exploration  
[extract_transform_attentions.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/pipeline/extract_transform_attentions.ipynb) -  from bert evaluation to dataset output to csv  
[dataset_partitioning.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/pipeline/dataset_partitioning.ipynb) - sub sampling a cross section of the dataset  
[remove_index.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/pipeline/remove_index.ipynb) - batch repair to 400GB dataset necessary because of human error

    
• clustering/

[intel_python_clustering.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/intel_python_clustering.ipynb) - benchmarking Intel Python accelerated scikit-learn  
[cuML_PCA_variance.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_PCA_variance.ipynb) - exploring PCA with attention representations  
[cuML_PCA_visualization.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_PCA_visualization.ipynb) - exploring PCA with attention representations  
[cuML_Dask_grid_search.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_dbscan_grid_search.ipynb) - determining optimal epsilon and min_samples for DBSCAN  
[cuML_Dask_optimalK.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_Dask_optimalK.ipynb) - determining optimal k for kMeans  
[cuML_Dask_kMeans_full.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_Dask_kMeans_full.ipynb) - clustering segmented dataset with Dask + cuML kMeans  
[cuML_Dask_dbscan_full.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_dbscan_full.ipynb) - clustering segmented dataset with Dask + cuML DBSCAN

## Running the code

If you're running a local Jupyter instance, You'll want to set up your enviroment

``` bash
$ python3 -m venv myenv
$ source myenv/bin/activate
$ pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install transformers==2.5.1
$ pip install wikipedia==1.4.0
```

I used Docker and the nvidia-docker image, particularly when working with jupyter notebooks

```
insert nvidia docker command here
```

Short of building a new custom image, or if working in a cloud environment like colab or paperspace, I typically used pip the following to install torch, transformers, and other necessary libraries as needed

```
!pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
!pip install transformers
```

For gpu accelerated clustering and dimensionality reduction, I leveraged cuML and Dask - which can be easily utilized with the Rapids Docker image, which also hosts a jupyter server:

```bash
run --gpus all -it --rm -p 8888:8888 -p 8787:8787 -p 8786:8786 -v "$(pwd):/rapids/notebooks/host/" rapidsai/rapidsai:21.06-cuda11.0-runtime-ubuntu18.04-py3.7

```

## Extract attentions from fine-tuned model during evaluation

In exploration/attention_exploration I investigated working with BERT models and visualization.  Comparing and contrasting heads and layers at this level of granularity led motivated working with a larger set to be able to see if we could utilize traditional machine learning methods to make some sense of their relationship, if any could be foun. 

The relevant code from squad.py file was used in exploration/extract_attentions.ipynb to produce raw attention matrices stored as torch .pkl files.  This process would use nearly 100gb of our available 128gb of memory to produce 100 examples worth of attention matrices at a time ( 100 examples x 12 layers @ 12 heads - 14400 384x384 matrices ), each file taking up ~10GB apiece.

## Transforming attention matrices to lower dimensional representations

While clearly this wasn't going to be possible for us to produce the entire 130k+ examples dataset worth of raw attentions due to space limitation, this gave us the opportunity to exlore our options and informed the decision to use the barlow twins model for feature extraction.

Our pipeline consisted of the following steps:

• Feed squad2 example to fine-tuned BERT  
• Scale attention values to 0-255  
• Reshape to (1, 3, 384, 384) tensor  
• Extract features using modified Barlow Twins  
• Flatten 12x12 representations  
• Convert tensors to dataframe columns  
• Append to dataframe  

To stay within memory constraints ( under 100gb of RAM ), examples were batched into representation format and added to an in-memory dataframe 350 at a time.  Every 5000 examples, this dataframe was written out to file, resulting in (26) ~16gb files representing all 130k squad2 training examples.


## Evaluating clustering algorithms on transformed attentions

We knew scaling was going to be of primary concern because we wanted to be able to cover as much as possible of the squad2 dataset.  Intel makes a [patch for Scikit-learn](https://github.com/intel/scikit-learn-intelex) which allows for the use of multiple cores, and while this looked like a promising alternative, the speed-up was not as significant as I had hoped.

In [intel_python_clustering.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/intel_python_clustering.ipynb) a smaller size dataset of 100 squad2 examples ( 14400 attention heads ), kMeans on 8 cores shows a roughly linear improvement from 10 minutes to 1.5 minutes.

On a larger set - the 2000 squad2 example output from pipeline/transform_attentions.ipynb - speedup was similar (~7x) and took over 45x longer to cluster with only 2x the # of rows.

[cuML](https://github.com/rapidsai/cuml) promised and delivered _much_ faster results - With one drawback - our total sample size was limited by the amount of GPU VRAM available.  Combining Dask and cuML allows us to parallelize kMeans and DBSCAN across multiple GPUs on the same machine.  This increased our capacity to 48gb across two Nvidia RTX-3090s.

In order to get a representational cross-section of the squad2 examples and more easily fine tune the size of our dataset to fit into VRAM, in [extract_attentions.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/exploration/extract_attentions.ipynb) 300 examples were extracted from each of the 26 files making up our whole dataset.

Based on findings in [cuML_Dask_optimalK.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_Dask_optimalK.ipynb) we performed kMeans clustering on our dataset in [cuML_Dask_kMeans_full.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_Dask_kMeans_full.ipynb)

Layers and heads columns are added for each row of the resulting cluster labels so that we can investigate correlations between clusters, heads and layers.

Running a grid search to find optimal parameters for DBscan in [cuML_Dask_grid_search.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_dbscan_grid_search.ipynb), we proceeded to cluster in [cuML_Dask_dbscan_full.ipynb](https://github.com/pschroedl/transformer_attention_clustering/blob/main/clustering/cuML_dbscan_full.ipynb), also adding columns to the result for analysis and visualization.

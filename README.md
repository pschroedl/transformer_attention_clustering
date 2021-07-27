# transformer attention clustering
A Springboard Capstone Project

## Proposal:
The Transformer architecture has pushed SOTA NLP  to new heights in the past few years.  As with many Deep Learning architectures, explainability is low and getting insight into why these mechanisms work so well in specific situations can be problematic.

Projects such as BertViz ( https://github.com/jessevig/bertviz ) and Tensor2Tensor (Tensor2Tensor visualization tool) work towards solving this problem by providing tools to visualize the attention patterns produced by one or more attention heads across layers of a Transformer, or the query and key vectors that are used to compute attention heads.

While these type of visualization of the attention heads is interesting, it is very difficult for the human eye to correlate and find meaning in the patterns in these images - especially across multiple input examples.  I propose to utilize a pipeline of transformations and traditional machine learning analysis and clustering methods to identify patterns across all of the attention heads produced across a dataset. This, I believe, will give some insight into the ability of transformers ability to encode information into the attention mechanism, and information about the locality in the layer/head matrix.

The attention computed by each head in each layer of the Transformer architecture is a matrix of values made up elements corresponding to the mapping of each token in the input sequence to every other token.  The proposed architecture will consist of collecting these attention matrices, and transforming them into a more computationally feasible dimensional space by utilizing a SOTA image classification model for feature extraction.  The output will then be analyzed by more traditional machine learning methods such as KMeans.  Learned clusters can then be correlated with their corresponding head, layer, and input example for insight. 


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


## Fine tuning a model


HF provides a script, run_squad.py that fine-tunes a Transformer model on one of the two SQuAD (stanford question answering) datasets.  Squad v2 adds the possibilty of an answer not being present in the context provided.  We had to modify this file to use BertForQuestionAnswering instead of AutoModel to load the pre-trained model.  AutoModel acts as a wrapper to accomodate a number of different architectures, giving flexibility to huggingface users looking to fine-tune on the QA task.  Unfortunately this wrapper doesn't respect the 'output_attentions' flag passed into the model config, and simply outputs the string 'attention', which we had to find out the head-scratchingly hard way.  It would appear that the most recent run_squad no longer uses auto_model, but at the time this was a necessary workaround to produce the attention maps for analysis.

The dataset for Squad2 training is found here:
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

The training and eval script and pipeline notebook are called with a data_dir parameter ( 'data/squad' ) where it expects to find this file ( and the dev set if evaluating ).

You can run fine tuning from a notebook (link to /bert-fine-tuning/fine-tune.ipynb)), or /bert-fine-tuning/fine-tune.sh . The trained model with associated config files will be output to 'models/bert'.

Evaluating via the eval.sh script will log associated results files to 'models/bert' and to the console.  Our eval run produced these scores:

Results:{
    'exact': 72.95544512760044,
    'f1': 76.28942043046025,
    'total': 11873,
    'HasAns_exact': 71.42375168690958,
    'HasAns_f1': 78.10126328793105,
    'HasAns_total': 5928,
    'NoAns_exact': 74.48275862068965,
    'NoAns_f1': 74.48275862068965,
    'NoAns_total': 5945,
    'best_exact': 72.95544512760044,
    'best_exact_thresh': 0.0,
    'best_f1': 76.28942043046025,
    'best_f1_thresh': 0.0
} 

## Extract attentions from fine-tuned model during evaluation

In exploration/attention_exploration I investigated working with BERT models and visualization.  Comparing and contrasting heads and layers at this level of granularity led motivated working with a larger set to be able to see if we could utilize traditional machine learning methods to make some sense of their relationship, if any could be foun. 

The relevant code from squad.py file was used in exploration/extract_attentions.ipynb to produce raw attention matrices stored as torch .pkl files.  This process would use nearly 100gb of our available 128gb of memory to produce 100 examples worth of attention matrices at a time ( 100 examples x 12 layers @ 12 heads - 14400 384x384 matrices ), each file taking up ~10GB apiece.

## Transforming attention matrices to lower dimensional representations

While clearly this wasn't going to be possible for us to produce the entire 130k+ examples dataset worth of attentions due to space limitation, it would give us an opportunity to exlore alternatives and informed the decision to use a resnet model - specifically barlow twins, with the last linear layer removed - to produce a lower dimensionality representation. 

Feasability and methods for transforming and analyzing these attention matrices were examing exploration/representations_cpu_exploration.  First transforming the 0-1 floating point values of the attention maps to 0-255, we then fed them to the barlow twins model to reduce our 384x384 sized vectors to a 1x2048 representation, and finally took a look swing at using traditional machine learning methods - KMeans and DBScan.  The amount of time to perform clustering at this scope ( ~2hrs for 2000 squad2 examples ) on CPU proved we would need to look to more computationally efficient methods.

At this point I decided to combine the separate pieces of attention matrix extraction and transformation to 2048 value representations.  pipeline/transform_attentions and exploration/transform_prototyping were combined into pipeline/extract_transform_attentions.  To stay within memory constraints, examples were batched into representation format and added to an in-memory dataframe 350 at a time.  Every 5000 examples, this dataframe was written out to file, resulting in 26 ~16gb files representing all 130k squad2 examples.  This took a little ove 5 days running on a machine with a 3090 and i7-9800x.  Unfortunately I had forgotten to add 'index=False' to the to_csv call, and had to run all of the files through /pipeline/remove_index.ipynb to remove the added first index column.

## Evaluate clustering algorithms on transformed attentions

Scaling was a large concern because I wanted to be able to process as much as possible of the squad2 dataset for clustering.  Intel makes a python distribution [link] which allows patching of SKLearn to use multiple cores, and while this looked like a promising alternative, the speed up was not as significant as I had hoped.  In clustering/intel_python_clustering I ran some small experiments and found it ran close to a linear speed-up per additional CPU ( <10x on my home 8 core machine - it appears to support multi-core, but not multi-threading ).  A high compute cloud cloud solution might be a possible solution, but I continued to look elsewhere 

cuML [Link] promised and delivered much greater results - the same small dataset that took >20 minutes on 8 CPUs took less than a minute to perform KMeans and DBScan.  Our total sample size was limited by the amount of GPU VRAM available and Dask allows us to parallelize across multiple GPUs on the same machine.   Ideally we would like to scale this to many more GPUs on one or more cloud instances, but unfortunatly it hasn't been yet possible to get our GPU quotas on Google Cloud approved for a personal project.  While pricy, a few A100s could give us enough VRAM to allow processing a very large subset of our full ~400GB dataset, if not in its entirety.

In order to get a representational subset of the squad2 examples, I extracted the first 300 squad2 examples from each of the 26 representation files using a snippet of code in clustering/segmentation.ipynb.  300 examples corresponds to 43,200 of the representation exported by the pipeline and takes up ~1GB of disk space.  Dask recommends that the dataset be split into 1-2gb sections before loading, so this also took care of that step for us.

##  Analysis of clustering

In clustering/cuML_clustering_Dask_full, we take a look at the clusters identified by kmeans.  Layers and heads columns are added for each row of our dataset so that we can identify possible correlations between clusters, heads and layers.  [ TODO: add example # and # of tokens in example ]




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


## Step one : fine tuning a model


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

## Step two : extract attentions from fine-tuned model during evaluation

## Step three : transform attentions

## Step four : evaluate clustering algorithms on transformed attentions
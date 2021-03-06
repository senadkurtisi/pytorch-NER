## Named Entity Recognition in PyTorch using Transformers 💻💥

**Quick Intro** <br></br>
This is an implementation of **Named Entity Recognition** model in **PyTorch**.

For traning [CoNLL2003](https://huggingface.co/datasets/conll2003) dataset was used. Dataset was acquired by leveraging the [HuggingFace datasets API](https://huggingface.co/datasets).

This repo contains custom implementation of the attention mechanism. Using the PyTorch built-in Transformer encoder is a viable option, I just implemented this as practice. 🧐


Table of Contents:
1. [Problem Formulation](#problem-formulation)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
4. [Model Performance](#model-performance)
5. [Instructions](#setup-and-instructions)
6. [Acknowledgements](#acknowledgements)

## Problem Formulation

Named Entity Recognition is an NLP problem in which we have a body of text (example: a sentence) and we try to classify if each word/subword of this input sequence represents a *Named Entity*.
This can be further expanded by also trying to predict not only if an input token is a named entity, but also to which class that named entity belongs to. For example: a person, name of the city etc. <br>
An example for this would be:
<br>
<br>
<p align="center">
  <img src="imgs\ner_example.jpg" />
</p>

## Dataset
[CoNLL2003](https://huggingface.co/datasets/conll2003) is one of the standard datasets used in the area of named entity recognition.

### Dataset split 
The dataset is originaly split into *train, validation and test* subsets.

<p align="center">
  <img src="imgs\subsets.jpg" />
</p>

### Labels
Each sample in the dataset is defined by input sequence and labels for each element of the sequence. These labels fall into **9 different categories**. <br>

These categories can be described by the following list:
* NULL - not an entity
* PER - Person
* ORG - Organization
* LOC - A location
* MISC

We also need to consider that some entities consist out of more than one token. For example *New York*. Therefore each of categories defined above (besides NULL), is split into two subcategories, which carry the same name as the original category with prefixes *B* and *I*. Example: **LOC** ➡ **B-LOC** & **I-LOC**.

Below we can see the distribution of classes in the training set. As we can see the dataset is **highly unbalanced**. The situation is similar for the other two subsets as well.

<p align="center">
  <img src="imgs\sample_distrib.jpg" />
</p>

## Architecture

Model used in this implementation is the **Encoder part** of the famous **Transformer** architecture

This repo contains custom implementation of the self-attention mechanism originally presented in the "***Attention is All You Need***" [paper](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et al

Using the [Transformer Encoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) module implemented in PyTorch is a viable and a high quality option. I implemented this from scratch for practice purposes.

The architecture contains some modifications which were implemented since they improved the performance.


## Model Performance

Due to the dataset being highly unbalanced [F1 score](https://en.wikipedia.org/wiki/F-score) was used as a primary metric for model evaluation.<br>
Since majority of the dataset tokens belongs to the **O** class which corresponds to the non-named entity model performance was evalauted in two ways.


Performance comparison is given below for each of three subsets, on two slices of data:
* All input tokens
* Named Entity tokens i.e. tokens which belong to classes which aren't ***O*** class. 

<br>
<p align="center">
  <img src="imgs\f1_score.jpg" />
</p>

We can see the problem difficulty illustrated in the bar plot shown above. Model is having tough time maintaining generalization power to due to dataset being highly imbalanced. We can see that the gap between training set and validation and test sets is smaller in the case when we use all of the tokens.

## Setup and Instructions
1. Open Anaconda Prompt and navigate to the directory of this repo by using: ```cd PATH_TO_THIS_REPO ```
2. Execute ``` conda env create -f environment.yml ``` This will set up an environment with all necessary dependencies. 
3. Activate previously created environment by executing: ``` conda activate ner-pytorch ```
4. Download GloVe embeddings from the following [link](https://nlp.stanford.edu/projects/glove/). Choose the one marked as **"glove.6B.zip"**
5. In the [configuration file](config.json) modify the ``` glove_dir ``` entry and change it to the path to directory where you have previoulsy downloaded the **GloVe** embeddings.
6. Run ``` python prepare_dataset.py ```. This will perform the following steps:
    * Download the **CoNLL2003** dataset using the **HuggingFace dataset API**
    * Process train, validation and test subsets and save them to ``` dataset ``` directory
    * Generate the vocabulary using tokens from the training set
    * Extract **GloVe** embeddings for tokens present in previously created vocabulary
7. Run ``` python main.py ``` to initiate the training of the model </br>
   
## Acknowledgements
These resources were very helpful for me:
* [Official PyTorch Attention Mechanism Implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
* [Official Scale Norm Implementation](https://github.com/tnq177/transformers_without_tears)
* [Aleksa's Transformer implementation](https://github.com/gordicaleksa/pytorch-original-transformer)

## Licence
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

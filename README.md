# Named Entity Recognition in PyTorch using Transformers 💻

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

Named Entity Recognition is an NLP problem in which we have a body of text (example: a sentence) and we try to classify if each word/subword of this input sequence represents a *Named Entity*. <br><br>
An example for this would be:
<br>
<br>
<p align="center">
  <img src="imgs\ner_example.jpg" />
</p>

This can be further expanded by also trying to predict not only if an input token is a named entity, but also to which class that named entity belongs to. For example: a person, name of the city etc.

## Dataset
[CoNLL2003](https://huggingface.co/datasets/conll2003) is one of the standard datasets used in the area of named entity recognition.

### Dataset split 
The dataset is originally split into three subsets:
* training
* validation
* test

<p align="center">
  <img src="imgs\subsets.jpg" />
</p>

### Labels
Each sample in the dataset is defined by input sequence and labels for each element of the sequence. These labels fall into **10 different categories**. <br>

These categories are:
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

Model used in this implementation is the **Encoder part** of the famous **Transformer** architecture. <br>This repo contains custom implementation of the self-attention mechanism originally presented in the "***Attention is All You Need***" [paper](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et al. <br>Using the [Transformer Encoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) module implemented in PyTorch is a viable and a high quality option. I implemented this from scratch for practice purposes.

The architecture contains some modifications which were implemented since they improved the performance.


## Model Performance

Due to the dataset being hihgly unbalanced [F1 score](https://en.wikipedia.org/wiki/F-score) was used as a primary metric for model evaluation.<br>
Besides that, the majority of the dataset tokens belongs to the **O** class which corresponds to the non-named entity.<br><br>
Therefore, a performance comparison is given below for each of three subsets, on two slices of data:
* All input tokens
* Named Entity tokens i.e. tokens which belong to classes which aren't ***O*** class. 

<br>
<p align="center">
  <img src="imgs\f1_score.jpg" />
</p>

We can see that the problem difficulty illustrated in the bar plot shown above. Model is having tough time maintaining generalization power to due to high imbalance of the dataset. We can see that the gap between training set and validation and test sets is smaller in the case when we use tokens which belong to the ***O*** class.

## Setup and Instructions
1. Open Anaconda Prompt and navigate to the directory of this repo by using: ```cd PATH_TO_THIS_REPO ```
2. Execute ``` conda env create -f environment.yml ``` This will set up an environment with all necessary dependencies. 
3. Activate previously created environment by executing: ``` conda activate ner-pytorch ```
4. Run ``` main.py ``` in your IDE or via command line by executing ``` python main.py ```. </br>
   
## Acknowledgements
These resources were very helpful for me:
* [Official PyTorch Attention Mechanism Implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
* [Official Scale Norm Implementation](https://github.com/tnq177/transformers_without_tears)
* [Aleksa's Transformer implementation](https://github.com/gordicaleksa/pytorch-original-transformer)

## Licence
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

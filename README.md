# Conditional sentence generation by using a force word 

This is a repository for a university project (University of Twente) for the course Natural Language Processing.

## Short Project Description

The main idea for the research project is to create a sentence generator. This sentence generator shall generate conditional sentences out of only word together with its’ given speech tag. In exceptional cases without any part of speech tag, the most common (with the highest probability) tag shall be used for the particular word. The structure in which the given word shall be implemented in, will be a simple subject-verb-object structure, e.g., <subject> <adverb for frequency> <verb> <adverb for place or time>. 

We think that with these given features, the model should be able to generate a sentence that makes sense. 

## Research Questions

* How to identify the grammatical part in the sentence for the given word? 
* How to make sure that the generated sentences actually make sense and are based on a continuous bag of words (CBOW)? 
* How can we force our model to generate the sentence that contains a given word? 
* Optional: Does the order of adverbs as well as the sentence order influence the performance of the model? 

## Datasets 

* https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5150

## References 

Almutiri, T., & Nadeem, F. (2022). Markov models applications in natural language processing: a survey. Int. J. Inf. Technol. Comput. Sci, 2, 1-16. 
Cao, S., Kitaev, N., & Klein, D. (2020). Unsupervised parsing via constituency tests. arXiv preprint arXiv:2010.03146. 

## Setup and usage

After downloading the code and installing the requirements you have to download the corpus from the link provided in the datesets, create a folder called "Data" in the root directory of the project and place there the UD_English-EWT folder.

The program can be executed with the `main.py` command.

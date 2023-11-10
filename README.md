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

## Literature 

In previous research, grammatical models have already been used to provide unsupervised parsing based on the linguistic notion of a constituency test. Therefore, a set of sentences will be transformed by adding words intentionally. Afterwards, these sentences shall be judges towards their grammatical correctness (Cao et al., 2020). 

Sentence generation has been done quite frequently in the past with the help of Markov chains and hidden Markov models (Almutiri et al., 2022). Due to this information, we feel confident that this can be an approach to proceed further on with. 

## Methods 

1. Data preprocessing: tag the corpus with the respective SOV-tags (we couldn’t find a corpus pre-tagged with sentence part tags) 
2. Calculate the most probable SOV and POS tag for every word with the help of hidden Markov models 
3. Extend the model on test set 

## Datasets 

* https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5150

## References 

Almutiri, T., & Nadeem, F. (2022). Markov models applications in natural language processing: a survey. Int. J. Inf. Technol. Comput. Sci, 2, 1-16. 
Cao, S., Kitaev, N., & Klein, D. (2020). Unsupervised parsing via constituency tests. arXiv preprint arXiv:2010.03146. 


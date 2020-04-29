# Language Modeling with Transformer-XL

## Introduction

I include here the code and data to:
- Access a Transformer-XL Language Model and obtain the prediction of the next word
- Train a Transformer-XL on a specified text corpus. 
It can include a Wikipedia Dump, and it also works in languages other than English.

## Next-word prediction
- To select the dataset:\
`import InputFacilities.Input as Input`\
 `dataset = Input.Dataset.DANISH`
 
- To create the Language Model object and predict the next word, with the input context written in manually:\
`LM = TW.LM_TransformerXL(dataset, flag_text_or_manual=False)`\
`LM.predict("Jeg kan godt lige rødbeder ")`

- Alternative: To create the Language Model object and predict the next word, with the input context in the text file:\
`import TW`\
`LM = TW.LM_TransformerXL(dataset, flag_text_or_manual=True)`\
`LM.predict()`\
\
It will predict from the '|' character in the text file.\
As in Write-Assistant, a caret immediately after the end of a word means that we are completing that word - 
and thus, operating with a prefix - whereas a caret after a space, or adjacent to a punctuation sign, will ask for the next word.\
`Hvad er klokken?|`: Next Word\
`Jeg taler eng|`: Prefix: completing 'eng'\
`I morgen bliver jeg  |`: Next Word)\

### Notes on necessary files & folders
To deal with the different datasets - or any other that you may want to add - the current directory structure is as follows:
- Datasets > [dataset name] >  model file, corpus files used to train the model > Sources > wikiDump.bz2, and other source files if present

- Example:\
Datasets > Danish > model.pt, train.txt, valid.txt, test.txt > dawiki-latest-pages-articles.xml.bz2\

`model.pt` is the trained Transformer-XL\
The text files are also necessary because they are used to create the tokenizer for the given transformer

### Script
The Python3 script `run_LM_prediction-py` is currently a stub, 
that could be expanded upon in order to run the LM as a script or daemon service.

## Training
The `Training.py` module provides the code to access a Transformer-XL instrument.

It works on a the Python (version 3) console of a computer that has a GPU.\
You also need either a .bz2 wikidump or text files
in the Datasets/[dataset name]/Sources folder.\
For example, the commands are:\
`import Input; import Training`\
`dataset= Input.Dataset.SPANISH`\
`Training.train(dataset)`
 


## Credits and Acknowledgements
I included - and where needed, slightly modified - the following GitHub projects:
- The implementation of the Transformer-XL instrument, found at: https://github.com/kimiyoung/transformer-xl\
Based on the paper "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" was made by Zihang Dai et al., 2019


- WikiExtractor - 
  Python script that extracts and cleans text from a Wikipedia database dump. From: https://github.com/attardi/wikiextractor/wiki
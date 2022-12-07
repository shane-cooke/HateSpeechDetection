# Detecting Hate Speech

## Description

A python-based machine learning text classification project created to classify comments collected from online social media platforms as either non-hateful, or hateful.

## Contents

#### Reports :
 A folder containg a full thesis report on the project and a published academic paper on the project.

#### Data :
 A folder containg three 1000 post single-platform datasets, and the 3000 post multi-platform dataset.
 Full dataset published for public use: https://figshare.com/articles/dataset/Labelled_Hate_Speech_Detection_Dataset_/19686954

#### WebScrapers :
 A folder containing three .py files for scraping the Reddit, 4chan and Twitter social media platforms.

#### HateTesting.ipynb :
A jupyter notebook project which contains a full suite for the testing of machine learning classifiers and word embedding methods.

#### BERT.py :
A python program for the testing of machine learning classifiers using Google's BERT word embeddings.

#### ParamaterOptimisation.ipynb :
A jupyter notebook project which contains a full suite for the parameter optimisation of machine learning classifiers.

#### EmbeddingOptimisation.ipynb :
A jupyter notebook project which contains a full suite for the parameter optimisation of word embedding methods.

#### HateChecker.py :
A python-based streamlit application for the testing of user inputted strings against a collection of accurate hate speech detection models.

## Usage

#### HateTesting.ipynb :
```bash
jupyter notebook
```
Once jupyter notebook has launched, upload and open the file using the Jupyter Notebook user interface.

#### ParamaterOptimisation.ipynb :
```bash
jupyter notebook
```
Once jupyter notebook has launched, upload and open the file using the Jupyter Notebook user interface.

#### EmbeddingOptimisation.ipynb :
```bash
jupyter notebook
```
Once jupyter notebook has launched, upload and open the file using the Jupyter Notebook user interface.

#### WebScrapers :
```bash
python3 RedditScraper.py
```
```bash
python3 TwitterScraper.py
```
```bash
python3 4ChanScraper.py
```

#### BERT.py :
```bash
python3 BERT.py
```

#### HateChecker.py :
```bash
streamlit run HateChecker.py
```
![hatecheckinit](https://user-images.githubusercontent.com/120044490/206256266-4cb34a55-8c3b-49ab-b4fe-54c8f4ab1340.png)

## Authors and Acknowledgements
#### Author: Shane Cooke (17400206)
#### Acknowledgments: Soumyabrata Dev (Supervisor)

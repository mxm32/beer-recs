# AIPI 540 Recommender System Module Project
# Beer Recommender
**Miranda Morris and Rob Baldoni for Duke AIPI 540 2022**

Predicting Beer Ratings using [Beer Advocate Dataset](https://data.world/socialmediadata/beeradvocate)

<p align="center">
  <img src="https://fixcom.azureedge.net/assets/content/19743/craft-beer-header.png" width="400" /> 
</p>

# About

In this project, we trained a deep-learning recommender system tool using collaborative filtering and hybrid approaches  to predict how a user would rate a beer (1-5) based on their prior rating history. We used the "overall impression" ratings from the Beer Advocate dataset, consisting of about 1.5 million beer reviews.

# Instructions:
Type in your terminal:
1. `git clone https://github.com/mxm32/beer-recs.git`

Navigate to the new directory, then:

2. Go to [google drive data link] and download the dataset file
3. Unzip the zip file and then place it in the project's root directory

Then enter the following in your terminal:

4. `python -m venv <VirtualEnvironmentName>`
5. `source <VirtualEnvironmentName>/bin/activate`
6. `make install`
7. `python main.py`

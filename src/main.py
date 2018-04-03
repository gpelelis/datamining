#!/usr/bin/env python
"""
Example usage of wordcloud
===============

Generating the word cloud for the small dataset
"""

from os import path
from wordcloud import WordCloud
from Categories import Categories
# import matplotlib.pyplot as plt

TRAIN_SET_FILE = '../datasets/small_train_set.csv'
IMAGE_FOLDER = '../output/images/'

categories = Categories(TRAIN_SET_FILE)

text = categories.get_body_as_string('Film')
wordcloud = WordCloud().generate(text)

# display the image
image = wordcloud.to_file(IMAGE_FOLDER + 'test.png')

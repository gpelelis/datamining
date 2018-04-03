#!/usr/bin/env python
"""
Example usage of wordcloud
===============

Generating the word cloud for the small dataset
"""

from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt


TRAIN_SET_FILE = '../datasets/test_wordcloud.txt'
text = open(TRAIN_SET_FILE).read()
wordcloud = WordCloud().generate(text)

# display the image
plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")


# # lower max_font_size
# wordcloud = WordCloud(max_font_size=40).generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
plt.show()

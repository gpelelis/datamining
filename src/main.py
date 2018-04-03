#!/usr/bin/env python
"""
Example usage of wordcloud
===============

Generating the word cloud for the small dataset
"""
from wordcloud import WordCloud
import random

from Categories import Categories

# CONSTANTS
TRAIN_SET_FILE = '../datasets/small_train_set.csv'
IMAGE_FOLDER = '../output/images/'

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

# FUNCTIONS
def generate_word_cloud():
  # get the file from train set and prepare an object that keeps all the information for the files
  categories = Categories(TRAIN_SET_FILE)

  for category_name in categories.categories:
    text = categories.get_body_as_string(category_name)
    wordcloud = WordCloud().generate(text)
    default_colors = wordcloud.to_array()
    wordcloud.recolor(color_func=grey_color_func, random_state=3)

    # display the image
    image = wordcloud.to_file(IMAGE_FOLDER + category_name + '.png')

generate_word_cloud()
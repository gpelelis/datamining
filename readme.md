### wordcloud
Here you can find the [documentation of wordcloud API](http://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud). For further usage see [examples from github](https://github.com/amueller/word_cloud/tree/master/examples)

**usage example:**
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

TRAIN_SET_FILE = '../datasets/test_wordcloud.txt'
text = open(TRAIN_SET_FILE).read()
wordcloud = WordCloud().generate(text)

# display the image
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()
```

**tips**: 
to make it work you may need to run 
- ```conda install -c matplotlib ```
- on macos run the app as pythonw (not python)
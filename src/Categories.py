from pandas import read_csv

class Categories:
    def __init__(self, file):
      self.file = file
      self.categories = [
        'Politics',
        'Film',
        'Football',
        'Business',
        'Technology'
      ]
      self.contents = {}
      self.titles = {}

      # read_csv return a dataframe https://pandas.pydata.org/pandas-docs/stable/api.html#dataframe
      file_contents = read_csv(file, sep='\t')

      for category_name in self.categories:
        self.contents[category_name] = file_contents[file_contents["Category"] == category_name][["Content"]]
        self.titles[category_name] = file_contents[file_contents["Category"] == category_name][["Title"]]


    def get_body_as_string(self, category_name):
      if category_name not in self.categories:
        return ''
      else:
        return str(self.contents[category_name])

      
from pandas import read_csv

class Categories:
    def __init__(self, file):
      self.file = file
      self.categories = {
        'Politics': [],
        'Film': [],
        'Football': [],
        'Business': [],
        'Technology': []
      }

      # read_csv return a dataframe https://pandas.pydata.org/pandas-docs/stable/api.html#dataframe
      file_contents = read_csv(file, sep='\t')

      for index, row in file_contents.iterrows():
        target_category = self.categories[row['Category']]
        doc_to_add = {
          'id': row['Id'],
          'title': row['Title'],
          'content': row['Content']
        }
        target_category.append(doc_to_add)

    def get_body_as_string(self, category_name):
      str = ''

      target_category = self.categories[category_name]
      for index in range(len(target_category)):
        str += target_category[index]['content']

      return str

      
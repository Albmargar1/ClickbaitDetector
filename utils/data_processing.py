import pandas as pd
from sklearn.model_selection import train_test_split
import datasets

def process_file(data_dir, inputs):
  data = pd.read_csv(data_dir)
  data.rename(columns = {'truthClass':'labels'}, inplace = True)

  data.loc[data['labels'] == 'clickbait', 'labels'] = 1
  data.loc[data['labels'] == 'no-clickbait', 'labels'] = 0

  for n in data.columns:
    data.loc[data[n].isna(), n] = ''

  feature = []
  for i in range(data.shape[0]):
    feature.append('')
    for label in inputs:
      feature[i] += data[label][i]
      feature[i] += '[SEP]' # Esto es trampita y tengo que buscar el token de cada arquitectura
  data['feature'] = feature

  return data

def to_Dataset(dataset, clip=[0,0], split=False, train_val_split = 0):
  
  def clip_dataset_mean(dataset):
    return dataset[(dataset['truthMean']<clip[0]) | (dataset['truthMean']>=clip[1])].reset_index(drop=True)

  # Add inputs together
  if split:
    train, val = train_test_split(dataset, test_size=train_val_split)
    train = clip_dataset_mean(train)
    data_dic = datasets.DatasetDict({'train': datasets.Dataset.from_pandas(train),
                                     'val': datasets.Dataset.from_pandas(val)})
  else:
    test = dataset
    data_dic = datasets.DatasetDict({'test': datasets.Dataset.from_pandas(test)})    

  return data_dic

def tokenize_dataset(dataset, tokenizer):
  def tokenize_function(dataset):
    return tokenizer(dataset['feature'], 
                    truncation=True)

  return dataset.map(tokenize_function, batched=True)

import pandas as pd

def process_data(data_dir):
  data = pd.read_csv(data_dir)

  for n in data.columns:
    data.loc[data[n].isna(), n] = ''

  return data
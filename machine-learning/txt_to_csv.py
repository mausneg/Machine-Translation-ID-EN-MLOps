import pandas as pd
import os

def txt_to_csv():
    data = []
    for file in os.listdir('machine-learning/data'):
        if file.endswith('.txt'):
            with open(f'machine-learning/data/{file}', 'r') as f:
                for line in f:
                    data.append(line.strip().split('\t'))
    df = pd.DataFrame(data, columns=['english', 'indonesian', 'license'])
    df.to_csv('machine-learning/data/final/data.csv', index=False)

if __name__ == '__main__':
    txt_to_csv()
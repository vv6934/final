import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

label_names = ['aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'diningtable',
               'dog',
               'horse',
               'motorbike',
               'person',
               'pottedplant',
               'sheep',
               'sofa',
               'train',
               'tvmonitor']

ap = np.load('./ap.npy')

df = pd.DataFrame(columns=['label', 'ap'])
df['label'] = label_names
df['ap'] = ap
df.sort_values(by='ap', ascending=False, inplace=True)

sns.set_style('darkgrid')
plt.figure(figsize=(20, 18), dpi=150)
sns.lineplot(x='label', y='ap', data=df, marker='o', linewidth=5.)

plt.xticks(rotation=45, size=20)
plt.yticks(size=20)
plt.xlabel('class',size=25)
plt.ylabel('ap', size=25)
plt.savefig('ap.jpg')

import numpy as np
import matplotlib.pyplot as plt

import preprocessing


train_distribution = np.zeros((7), dtype=int)
test_distribution = np.zeros((7), dtype=int)

y = preprocessing.read_csv("data/digitStruct_train.csv")
y1 = preprocessing.read_csv("data/digitStruct_test.csv")

for image_name in y:
    train_distribution[len(y[image_name]["DigitLabel"])] += 1
        
for image_name in y1:
    test_distribution[len(y1[image_name]["DigitLabel"])] += 1

distribution = [x + y for x, y in zip(train_distribution, test_distribution)]
   
width = 0.6
ind = np.arange(6)
plt.rcParams['figure.figsize'] = (11.0, 6.82)
fig, ax = plt.subplots()
ax.bar(ind, distribution[1:], width, color='r')
ax.set_ylabel('Number of occurrences')
ax.set_title('Distribution of the digit sequence length')
ax.set_xticklabels(('1', '2', '3', '4', '5', '>5'))
ax.set_xlabel('Length of the digit sequence')
ax.set_xticks(ind + width*0.5)
plt.margins(0.1)
plt.ylim(ymin=0)
plt.show()

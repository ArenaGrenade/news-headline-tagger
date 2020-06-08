import json
import matplotlib.pyplot as plt
import random
import time
import csv

random.seed(time.time())

allowed_categories = ["BUSINESS", "ENTERTAINMENT", "SCIENCE", "TECH", "SPORTS"]
general_categories = ["WORLD NEWS", "IMPACT", "POLITICS", "COMEDY", "MEDIA"]
healthy_categories = ["WELLNESS", "HEALTHY LIVING"]
all_cat = {}

cat_indices = {"BUSINESS": 0,
               "ENTERTAINMENT": 1,
               "GENERAL": 2,
               "HEALTH": 3,
               "SCIENCE": 4,
               "SPORTS": 5,
               "TECH": 6
               }

data = []
with open("News_Category_Dataset_v2.json") as base_dataset:
    for line in base_dataset:
        linedict = json.loads(line)
        if linedict["category"] in allowed_categories:
            data.append([linedict["headline"], linedict["short_description"], cat_indices[linedict["category"]]])
        elif linedict["category"] in general_categories:
            data.append([linedict["headline"], linedict["short_description"], cat_indices["GENERAL"]])
        elif linedict["category"] in healthy_categories:
            data.append([linedict["headline"], linedict["short_description"], cat_indices["HEALTH"]])

"""
Found data to be skewed. So, randomly removing articles to balance.
"""
new_data = data
data = []

for x in new_data:
    if x[2] in [4, 6]:
        data.append(x)
    if x[2] == 1:
        if random.random() < 0.63:
            continue
    if x[2] == 3:
        if random.random() < 0.72:
            continue
    if x[2] == 2:
        if random.random() < 0.74:
            continue
    data.append(x)

for d in data:
    if d[2] not in all_cat.keys():
        all_cat[d[2]] = 1
    else:
        all_cat[d[2]] += 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(all_cat.keys(), all_cat.values())
print(all_cat)

"""
Exporting data
"""
random.shuffle(data)
print(len(data))
partition = int(len(data) * 0.2)

with open('./test_data/test_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data[:partition])

with open('./training _data/training_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data[partition:])

if __name__ == '__main__':
    plt.show()

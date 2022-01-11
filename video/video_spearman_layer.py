import json
from scipy.stats import spearmanr

f = open("video_result_layer.json", 'r', encoding='utf-8')
result_layer = dict()
for line in f.readlines():
    tmp = json.loads(line)
    for (key, value) in tmp.items():
        result_layer[key] = value
f.close()

ff = open("video_layer.json", 'r', encoding='utf-8')
true_layer = dict()
for line in ff.readlines():
    tmp = json.loads(line)
    for  (key, value) in tmp.items():
        true_layer[key] = value
ff.close()
name = []
result_l = []
true_l = []
for (key, value) in result_layer.items():
    name.append(key)
    result_l.append(value)
    true_l.append(true_layer[key])
coef, p = spearmanr(result_l, true_l)
print(coef)
for i in range(len(name)):
    print(name[i], ' ', result_l[i], ' ', true_l[i])

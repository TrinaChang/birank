import json
import re
import math
f = open("video_games_reviews.json", 'r', encoding='utf-8')
product_review = dict()
reviews = []
print('stage1')
for line in f.readlines():
    dic = json.loads(line)
    product_review[dic['asin']] = product_review.get(dic['asin'], 0) + 1
    reviews.append(dic)
f.close()

layer_qry = dict()
user_qry = dict()
user_qry_review = dict()
product_qry = dict()
product_cate = dict()
product_layer = dict()
product = dict()
product2 = []
print('stage2')
f1 = open("Video_Games_short.json", 'r', encoding='utf-8')
for line in f1.readlines():
    dic = json.loads(line)
    if dic['asin'] not in product2:
        product2.append(dic['asin'])
        # product.append(dic)
        product_cate[dic['asin']] = dic['category']
        product_layer[dic['asin']] = dic['brand']
        # product[dic['asin']] = product_review.get(dic['asin'], 0)
        product_qry[dic['category']] = product_qry.get(dic['category'], 0) + product_review.get(dic['asin'], 0)
        layer_qry[dic['brand']] = layer_qry.get(dic['brand'], 0) + product_review.get(dic['asin'], 0)
f1.close()
print('stage3')
w = dict()
for review in reviews:
    string = '{layer},{user}'.format(layer = product_layer[review['asin']], user = review['reviewerID'])
    user_qry_review[string] = user_qry_review.get(string, 0) + 1
    if 'vote' in review:
        vote = int(re.sub("[^0-9]", "", review['vote']))
        user_qry[string] = user_qry.get(string, 0) + vote
    else:
        user_qry[string] = user_qry.get(string, 0) + 1
        
    u_p = review['reviewerID'] + ',' + product_cate[review['asin']] + ',' + product_layer[review['asin']]
    overall = review['overall']
    if u_p in w:
        w[u_p][0] += overall
        w[u_p][1] = w[u_p][1] + 1
    else:
        w[u_p] = [overall, 1]

with open('video_weight.txt', 'a+') as file:
    for ele in w.items():
        arr = ele[0].split(",")
        string = '{user},{product},{weight},{layer}'.format(user = arr[0], product= arr[1], weight = ele[1][0], layer = arr[2])
        file.write(string + '\n')

with open('video_product.txt', 'a+') as file:
    for ele in product_qry.items():
        file.write(json.dumps(ele) + '\n')

with open('video_user.txt', 'a+') as file:
    for ele in user_qry.items():
        tmp = dict()
        fac = ele[1]
        if (ele[1] <= 1):
            fac = 3
        tmp[ele[0]] = math.log(fac) * math.log(user_qry_review[ele[0]], 2)
        file.write(json.dumps(tmp) + '\n')

with open('video_layer.txt', 'a+') as file:
    for ele in layer_qry.items():
        tmp = dict()
        tmp[ele[0]] = ele[1]
        file.write(json.dumps(tmp) + '\n')

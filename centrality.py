from operator import le
from typing import OrderedDict
from numpy.lib import user_array
from numpy.lib.arraysetops import unique
from numpy.lib.ufunclike import isposinf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.coo import coo_matrix
import json

def cross_layer(listOfW, pCrossWeight, uCrossWeight, theta=0.425, delta=0.425, phi=0.425, lamb=0.425, max_iter=200):
        # initialize the lists that would be used later
        WTList = []
        spList = []
        sdList = []
        p0List = []
        u0List = []
        p_lastList = []
        u_lastList = []
        # iterate over the weight matrix of the layers
        for W in listOfW:
            W = W.astype('float', copy=False)
            # get the transpose of the weight matrix and append to the list for later use
            WT = W.T
            WTList.append(WT)
            # dp and du(sum of edges)
            kp = np.array(W.sum(axis=1)).flatten()
            ku = np.array(W.sum(axis=0)).flatten()
            # avoid divide by 0
            kp[np.where(kp==0)] += 1
            ku[np.where(ku==0)] += 1

            kp_ = spa.diags(1/kp)
            ku_ = spa.diags(1/ku)

            kp_bi = spa.diags(1/np.lib.scimath.sqrt(kp))
            ku_bi = spa.diags(1/np.lib.scimath.sqrt(ku))
            # get the S matirx and its transpose
            S = ku_bi.dot(WT).dot(kp_bi)
            spList.append(S.T)
            sdList.append(S)
            # query vector
            p0 = np.repeat(1 / kp_.shape[0], kp_.shape[0])
            p0List.append(p0)
            p_lastList.append(p0.copy())
            u0 = np.repeat(1 / ku_.shape[0], ku_.shape[0])
            u0List.append(u0)
            u_lastList.append(u0.copy())
        
        pCrossWeight = pCrossWeight.astype('float', copy=False)
        uCrossWeight = uCrossWeight.astype('float', copy=False)
        crossList = []
        # weighted degree for cross layers
        crossList.append(np.array(pCrossWeight.sum(axis=1)).flatten()) # col alpha dpA
        crossList.append(np.array(pCrossWeight.sum(axis=0)).flatten()) # dpB
        crossList.append(np.array(uCrossWeight.sum(axis=1)).flatten()) # duA
        crossList.append(np.array(uCrossWeight.sum(axis=0)).flatten()) # duB
        for i in range(len(crossList)):
            crossList[i][np.where(crossList[i]==0)] += 1
            crossList[i] = spa.diags(1/np.lib.scimath.sqrt(crossList[i]))
            # the S matrix
        spAB = crossList[1].dot(pCrossWeight).dot(crossList[0])
        suAB = crossList[3].dot(uCrossWeight).dot(crossList[2])
        #inintialise
        p = [0]*len(listOfW)
        u = [0]*len(listOfW)
        factorP = 0
        factorU = 0
        # iterate for max_iter time
        for i in range(max_iter):
            # iterate over the layers
            for layer in range(len(listOfW)):
                # different weighted degree in different layers
                if (i != 0):
                    if (layer == 0):
                        factorP = spAB.dot(p[1])
                        factorU = suAB.dot(u[1])
                    else:
                        factorP = spAB.dot(p[0])
                        factorU = suAB.dot(u[0])
                p[layer] = theta * spList[layer].dot(u_lastList[layer]) + delta * factorP + (1 - theta - delta) * p0List[layer]
                u[layer] = phi * sdList[layer].dot(p_lastList[layer]) + lamb * factorU + (1 - phi - lamb) * u0List[layer]
                # normalise
                p[layer] /= sum(p[layer])
                u[layer] /= sum(u[layer])
                p_lastList[layer] = p[layer]
                u_lastList[layer] = u[layer]
        return p, u

def centrality(listOfW, listOfZ, theta=0.475, delta=0.475, phi=0.475, gamma=0.475, lamb=0.475, xi=0.475, max_iter=200, pQuery=None, uQuery=None):
    # initialize the lists that would be used later
    WTList = []
    spList = []
    suList = []
    p0 = []
    u0 = []
    p_lastList = []
    u_lastList = []
    betaP = []
    betaU = []
    totalW = 0
    # retrieve the initial state
    # iterate over the weight matrix of the layers
    np.set_printoptions(threshold=np.inf)
    count = 0
    for W in listOfW:
        W = W.astype('float', copy=False)
        # get the transpose of the weight matrix and append to the list for later use
        WT = W.T
        WTList.append(WT)
        # dp and du(sum of edges)
        du = np.array(W.sum(axis=1)).flatten()
        dp = np.array(W.sum(axis=0)).flatten()
        totalW += sum(du)
        # beta for calculating z later
        betaU.append(du)
        betaP.append(dp)
        # avoid divide by 0
        kp_ = spa.diags(np.divide(1, dp, out=np.zeros_like(dp), where=dp!=0))
        ku_ = spa.diags(np.divide(1, du, out=np.zeros_like(du), where=du!=0))
        du = np.divide(1, np.lib.scimath.sqrt(du), out=np.zeros_like(du), where=du!=0)
        dp = np.divide(1, np.lib.scimath.sqrt(dp), out=np.zeros_like(dp), where=dp!=0)
        # get the S matrix and its transpose
        kp_bi = spa.diags(dp)
        ku_bi = spa.diags(du)
        S = ku_bi.dot(W).dot(kp_bi)
        # print(S.toarray())
        spList.append(S.T)
        suList.append(S)
        count += 1
    betaU[:] = [x / totalW for x in betaU]
    betaP[:] = [x / totalW for x in betaP]

    # query vector
    if pQuery is not None:
        p0 = pQuery.copy()
    else:
        p0 = np.random.randint(1, 2, size=kp_.shape[0])
    if uQuery is not None:
        u0 = uQuery.copy()
    else:
        u0 = np.random.randint(1, 2, size=ku_.shape[0])

    p0 = p0.astype('float')
    u0 = u0.astype('float')

    if (sum(p0) != 0):
        p0 /= sum(p0)
    if (sum(u0) != 0):
        u0 /= sum(u0)
    p_lastList = np.tile(p0, (len(listOfW), 1))
    u_lastList = np.tile(u0, (len(listOfW), 1))
    p = [0]*len(listOfW)
    u = [0]*len(listOfW)
    z0 = listOfZ.copy()
    z = listOfZ

    # print(max_iter)
    # iterate for max_iter time
    for i in range(max_iter):
    # for i in range(50):
        # iterate over the layers
        # print("iter", i)
        for layer in range(len(listOfW)):
            # print("layer num", layer)
            p[layer] = np.array(theta * spList[layer].dot(u_lastList[layer]) + delta * np.array(np.matrix(p_lastList).T.dot(np.array(z)))[0] + (1 - theta - delta) * p0)
            u[layer] = np.array(phi * suList[layer].dot(p_lastList[layer]) + gamma * np.array(np.matrix(u_lastList).T.dot(np.array(z)))[0] + (1 - phi - gamma) * u0)
            z[layer] = lamb * betaU[layer].dot(u_lastList[layer]) + xi * betaP[layer].dot(p_lastList[layer]) + (1 - lamb - xi) * z0[layer]
            # if (layer == 97 or layer == 98):
            #     print(layer)
            #     print(betaU[layer].dot(u_lastList[layer]))
            #     print(betaP[layer].dot(p_lastList[layer]))
            #     print(z[layer])
                # print(u_lastList[layer])
                # print(p_lastList[layer])
            # if layer == 2:
            #     print('good ', z[layer])
            #     print(betaU[layer])
            #     print(u_lastList[layer])
            # if layer == 5:
            #     print('bad ', z[layer])
            #     print(betaU[layer])
            #     print(u_lastList[layer])
            # normalise
            p[layer] /= sum(p[layer])
            u[layer] /= sum(u[layer])
            p_lastList[layer] = p[layer]
            u_lastList[layer] = u[layer]
        z /= sum(z)
    # exit()
    return p, u, z

class BipartiteNetwork:
    def init(self):
        self.data = OrderedDict()
    def set_edgelist(self, df, top_col, bottom_col, layer_col, importance_col=None, dfImportance=None, weight_col='weight'):
        self.df = df
        self.dfImportance = dfImportance
        self.top_col = top_col
        self.bottom_col = bottom_col
        self.weight_col = weight_col
        self.layer_col = layer_col
        self.importance_col = importance_col
        self._index_nodes()
        self._generate_adj()
    def set_cross_layer(self, dfProduct, dfUser, product1_col, product2_col, user1_col, user2_col, cross_product_col=None, cross_user_col=None):
        self.product1_col = product1_col
        self.product2_col = product2_col
        self.user1_col = user1_col
        self.user2_col = user2_col
        self.dfProduct = dfProduct
        self.dfUser = dfUser
        self.cross_product_col = cross_product_col
        self.cross_user_col = cross_user_col
        self.set_cross_layer_product()
        self.set_cross_layer_user()

    def set_cross_layer_product(self):
        for i in range(len(self.top_ids[self.top_col])):
            self.dfProduct[self.product1_col] = self.dfProduct[self.product1_col].replace(self.top_ids[self.top_col].get(i), i)
            self.dfProduct[self.product2_col] = self.dfProduct[self.product2_col].replace(self.top_ids[self.top_col].get(i), i)
        weight = self.dfProduct[self.cross_product_col]
        W = spa.coo_matrix(
                (
                    weight,
                    (self.dfProduct[self.product1_col].values, self.dfProduct[self.product2_col].values)
                )
            )
        self.pCrossWeight = W

    def set_cross_layer_user(self):
        for i in range(len(self.bottom_ids[self.bottom_col])):
            self.dfUser[self.user1_col] = self.dfUser[self.user1_col].replace(self.bottom_ids[self.bottom_col].get(i), i)
            self.dfUser[self.user2_col] = self.dfUser[self.user2_col].replace(self.bottom_ids[self.bottom_col].get(i), i)
        weight = self.dfUser[self.cross_user_col]
        W = spa.coo_matrix(
                (
                    weight,
                    (self.dfUser[self.user1_col].values, self.dfUser[self.user2_col].values)
                )
            )
        self.uCrossWeight = W
        
    def _index_nodes(self):
        self.top_ids = pd.DataFrame(self.df[self.top_col].unique(), columns=[self.top_col]).reset_index()
        self.bottom_ids = pd.DataFrame(self.df[self.bottom_col].unique(), columns=[self.bottom_col]).reset_index()
        self.top_ids = self.top_ids.rename(columns={'index': 'top_index'})
        self.bottom_ids = self.bottom_ids.rename(columns={'index': 'bottom_index'})
        self.df = self.df.merge(self.top_ids, on=self.top_col)
        self.df = self.df.merge(self.bottom_ids, on=self.bottom_col)

    def _generate_adj(self):
        data = OrderedDict()
        for element in self.df[self.layer_col].unique():
            newDf = self.df[self.df[self.layer_col] == element]
            # print(newDf)
            weight = newDf[self.weight_col]
            W = spa.coo_matrix(
                    (
                        weight,
                        (newDf['top_index'].values, newDf['bottom_index'].values)
                    ),
                    shape=(len(self.top_ids), len(self.bottom_ids))
                )
            if self.importance_col != None:
                data[element] = (W, self.dfImportance[(self.dfImportance[self.layer_col] == element)][self.importance_col].values[0])
            else:
                data[element] = (W, 0)
        self.data = data

    def generate_birank(self, **kwargs):
        listOfW = []
        listOfZ = []
        for (W, importance) in self.data.values():
            listOfW.append(W)
            listOfZ.append(importance)
        p, u, z = centrality(listOfW, listOfZ, **kwargs)
        self.p = p
        self.u = u
        self.z = z
        self.print_result(p, u, z)
        # self.graph(p, u, z)

    def generate_birank_cross_layer(self):
        listOfW = []
        for (W, importance) in self.data.values():
            listOfW.append(W)
        p, u = cross_layer(listOfW, self.pCrossWeight, self.uCrossWeight)
        # self.print_result(p, u)
        
    def print_result(self, p, u, z=None):
        top_df = self.top_ids.copy()
        bottom_df = self.bottom_ids.copy() 
        counter=0
        for ele in self.data.keys():
            if z is not None:
                print(ele, "layer influence:", z[counter])
            else:
                print(ele)
            top_df[self.top_col + '_birank'] = u[counter]
            bottom_df[self.bottom_col + '_birank'] = p[counter]
            print(top_df[[self.top_col, self.top_col + '_birank']])
            print(bottom_df[[self.bottom_col, self.bottom_col + '_birank']])
            counter += 1
    def graph(self, p, u, z=None):
        top_df = self.top_ids.copy()
        bottom_df = self.bottom_ids.copy() 
        counter=0
        # for ele in self.data.keys():
        y = p[0].tolist()
        x = top_df[self.top_col].to_numpy().tolist()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, alpha=0.3, color='red')
        ax.scatter(x, p[1].tolist(), alpha=0.3, color='blue')
        plt.show()

    def get_p(self):
        return self.p
    def get_u(self):
        counter=0
        counter 
        result = dict()
        top_df = self.top_ids.copy()
        for ele in self.data.keys():
            user = []
            user = top_df['user'].to_list()
            user_result = self.u[counter]
            counter2 = 0
            for i in user:
                string = '{layer},{user}'.format(layer = ele, user = i)
                result[string] = user_result[counter2]
                counter2 += 1
            counter += 1
        # with open('bb.txt', 'a+') as file:
        #     for ele in result.items():
        #         tmp = dict()
        #         tmp[ele[0]] = ele[1]
        #         file.write(json.dumps(tmp) + '\n')

        # print(self.u[0])
        # counter=0
        # result = []
        # for ele in self.data.keys():
        #     dic = dict()
        #     dic[ele] = self.z[counter]
        #     result.append(dic)
        #     counter += 1
    def get_z(self):
        counter=0
        result = []
        for ele in self.data.keys():
            dic = dict()
            dic[ele] = self.z[counter]
            result.append(dic)
            counter += 1
    
        with open('rr.txt', 'a+') as file:
            for ele in result:  
                file.write(json.dumps(ele) + '\n')


            
# testing for centrality
# df = pd.read_csv('data.csv', names=['color', 'product', 'weight', 'country'])
# dfImportance = pd.read_csv('importance.csv', names=['country', 'importance'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'color', 'product', 'country', 'importance', dfImportance, 'weight')
# bn.generate_birank()

# top = user(color)
# bottom = product
# not every layer has the same number of item
# df = pd.read_csv('data_with0.csv', names=['color', 'product', 'weight', 'country'])
# dfImportance = pd.read_csv('importance.csv', names=['country', 'importance'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'color', 'product', 'country', 'importance', dfImportance, 'weight')
# bn.generate_birank()

# video
df = pd.read_csv('video_weight.csv', names=['user', 'product', 'weight', 'brand'])
bn = BipartiteNetwork()
bn.set_edgelist(df, 'user', 'product', 'brand', weight_col='weight')
print('generating birank')
bn.generate_birank()
bn.get_u()
bn.get_z()

#debugging
# df = pd.read_csv('debug.csv', names=['user', 'product', 'weight', 'brand'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'user', 'product', 'brand', weight_col='weight')
# # bn.generate_birank(max_iter=200)
# bn.generate_birank(max_iter=200, pQuery=np.array([0.005, 0.015, 0.01]), uQuery=np.array([0.01, 0.005]))

# DTI
# df = pd.read_csv('newData.csv', names=['user', 'item', 'weight', 'method'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'user', 'item', 'method', weight_col='weight')
# bn.generate_birank()

# Amazon
# df = pd.read_csv('amazon.csv', names=['user', 'item', 'weight', 'method'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'user', 'item', 'method', weight_col='weight')
# bn.generate_birank()

# alibaba
# df = pd.read_csv('alibaba.csv', names=['user', 'item', 'weight', 'method'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'user', 'item', 'method', weight_col='weight')
# bn.generate_birank()

#testing for cross_layer
# df = pd.read_csv('twoLayerWeight.csv', names=['color', 'user', 'weight', 'country'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'color', 'user', 'country', weight_col='weight')
# dfProduct = pd.read_csv('twoLayerCrossProduct.csv', names=['colour1', 'colour2', 'weightProduct'])
# dfUser = pd.read_csv('twoLayerCrossUser.csv', names=['user1', 'user2', 'weightUser'])
# bn.set_cross_layer(dfProduct, dfUser, 'colour1', 'colour2','user1', 'user2', 'weightProduct', 'weightUser')
# bn.generate_birank_cross_layer()

# df = pd.read_csv('personalized recommendation2.csv', names=['user', 'product', 'weight', 'platform'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'user', 'product', 'platform', weight_col='weight')
# bn.generate_birank()

# testing for popularity prediction
# platform A: u1=500 u2=10
# platform B: u1=0   u2=30
# p0List=[np.array([500, 10]),np.array([0, 30])]
# p0List=[np.array([500, 40]),np.array([500, 40])]
# p0List=[np.array([500/540, 10/540]),np.array([0, 30/540])]
# p0List=[np.array([500/510, 10/510]),np.array([0, 1])]

# platform A: u1=20 u2=10
# platform B: u1=0   u2=30
# p0List=[np.array([2/6, 1/6]),np.array([0, 1/2])]
# p0List=[np.array([2/3, 1/3]),np.array([0, 1])]
# df = pd.read_csv('popularity prediction.csv', names=['user', 'product', 'weight', 'platform'])
# dfImportance = pd.read_csv('popularity importance.csv', names=['platform', 'importance'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'user', 'product', 'platform', 'importance', dfImportance, weight_col='weight')
# bn.generate_birank(pQuery=[np.array([500/540, 10/540]),np.array([0, 30/540])])
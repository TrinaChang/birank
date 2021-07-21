from numpy.lib.arraysetops import unique
import pandas as pd
import numpy as np
import scipy
import scipy.sparse as spa
from scipy.sparse.coo import coo_matrix

def cross_layer(listOfW, pCrossWeight, uCrossWeight, theta=0.425, delta=0.425, phi=0.425, lamb=0.425, max_iter=200):
        WTList = []
        spList = []
        sdList = []
        p0List = []
        u0List = []
        p_lastList = []
        u_lastList = []
        for W in listOfW:
            W = W.astype('float', copy=False)
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
   
            S = ku_bi.dot(WT).dot(kp_bi)
            spList.append(S.T)
            sdList.append(S)

            p0 = np.repeat(1 / kp_.shape[0], kp_.shape[0])
            p0List.append(p0)
            p_lastList.append(p0.copy())
            u0 = np.repeat(1 / ku_.shape[0], ku_.shape[0])
            u0List.append(u0)
            u_lastList.append(u0.copy())
        
        pCrossWeight = pCrossWeight.astype('float', copy=False)
        uCrossWeight = uCrossWeight.astype('float', copy=False)
        crossList = []
        crossList.append(np.array(pCrossWeight.sum(axis=1)).flatten()) # col alpha dpA
        crossList.append(np.array(pCrossWeight.sum(axis=0)).flatten()) # dpB
        crossList.append(np.array(uCrossWeight.sum(axis=1)).flatten()) # duA
        crossList.append(np.array(uCrossWeight.sum(axis=0)).flatten()) # duB
        for i in range(len(crossList)):
            crossList[i][np.where(crossList[i]==0)] += 1
            crossList[i] = spa.diags(1/np.lib.scimath.sqrt(crossList[i]))
        spAB = crossList[1].dot(pCrossWeight).dot(crossList[0])
        suAB = crossList[3].dot(uCrossWeight).dot(crossList[2])
        p = [0]*len(listOfW)
        u = [0]*len(listOfW)
        factorP = 0
        factorU = 0
        for i in range(max_iter):
            for layer in range(len(listOfW)):
                if (i != 0):
                    if (layer == 0):
                        factorP = spAB.dot(p[1])
                        factorU = suAB.dot(u[1])
                    else:
                        factorP = spAB.dot(p[0])
                        factorU = suAB.dot(u[0])
                p[layer] = theta * spList[layer].dot(u_lastList[layer]) + delta * factorP + (1 - theta - delta) * p0List[layer]
                u[layer] = phi * sdList[layer].dot(p_lastList[layer]) + lamb * factorU + (1 - phi - lamb) * u0List[layer]
                p[layer] /= sum(p[layer])
                u[layer] /= sum(u[layer])
                p_lastList[layer] = p[layer]
                u_lastList[layer] = u[layer]
        return p, u

def centrality(listOfW, listOfZ, theta=0.425, delta=0.425, phi=0.425, gamma=0.425, lamb=0.425, xi=0.425, max_iter=200):
    # List for different layers
    WTList = []
    spList = []
    sdList = []
    p0List = []
    u0List = []
    p_lastList = []
    u_lastList = []
    betaP = []
    betaU = []

    # retrieve the initial state
    # weight for different layers
    for W in listOfW:
        W = W.astype('float', copy=False)
        WT = W.T
        WTList.append(WT)
        # dp and du(sum of edges)
        kp = np.array(W.sum(axis=1)).flatten()
        ku = np.array(W.sum(axis=0)).flatten()
        totalW = sum(kp)
        betaU.append(kp / totalW)
        betaP.append(ku / totalW)
        # avoid divide by 0
        kp[np.where(kp==0)] += 1
        ku[np.where(ku==0)] += 1

        kp_ = spa.diags(1/kp)
        ku_ = spa.diags(1/ku)

        kp_bi = spa.diags(1/np.lib.scimath.sqrt(kp))
        ku_bi = spa.diags(1/np.lib.scimath.sqrt(ku))
        S = ku_bi.dot(WT).dot(kp_bi)

        spList.append(S.T)
        sdList.append(S)

        p0 = np.repeat(1 / kp_.shape[0], kp_.shape[0])
        p0List.append(p0)
        p_lastList.append(p0.copy())
        u0 = np.repeat(1 / ku_.shape[0], ku_.shape[0])
        u0List.append(u0)
        u_lastList.append(u0.copy())

    p = [0]*len(listOfW)
    u = [0]*len(listOfW)
    z = [1]*len(listOfW)
    for i in range(max_iter):
        # layer from a,b, ....
        for layer in range(len(listOfW)):
            p[layer] = np.array(theta * spList[layer].dot(u_lastList[layer]) + delta * np.matrix(p_lastList).T.dot(np.array(z)) + (1 - theta - delta) * p0List[layer])[0]
            u[layer] = np.array(phi * sdList[layer].dot(p_lastList[layer]) + gamma * np.matrix(u_lastList).T.dot(np.array(z)) + (1 - phi - gamma) * u0List[layer])[0]
            z[layer] = lamb * betaP[layer].dot(u_lastList[layer]) + xi * betaU[layer].dot(p_lastList[layer]) + (1 - lamb - xi) * listOfZ[layer]
            p[layer] /= sum(p[layer])
            u[layer] /= sum(u[layer])
            p_lastList[layer] = p[layer]
            u_lastList[layer] = u[layer]
        z /= sum(z)
        
    return p, u, z

class BipartiteNetwork:
    def init(self):
        self.data = {}
    def set_edgelist(self, df, top_col, bottom_col, layer_col, importance_col=None, dfImportance=None, weight_col=None):
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
            dfProduct[self.product1_col] = dfProduct[self.product1_col].replace(self.top_ids[self.top_col].get(i), i)
            dfProduct[self.product2_col] = dfProduct[self.product2_col].replace(self.top_ids[self.top_col].get(i), i)
        weight = dfProduct[self.cross_product_col]
        W = spa.coo_matrix(
                (
                    weight,
                    (dfProduct[self.product1_col].values, dfProduct[self.product2_col].values)
                )
            )
        self.pCrossWeight = W

    def set_cross_layer_user(self):
        for i in range(len(self.bottom_ids[self.bottom_col])):
            dfUser[self.user1_col] = dfUser[self.user1_col].replace(self.bottom_ids[self.bottom_col].get(i), i)
            dfUser[self.user2_col] = dfUser[self.user2_col].replace(self.bottom_ids[self.bottom_col].get(i), i)
        weight = dfUser[self.cross_user_col]
        W = spa.coo_matrix(
                (
                    weight,
                    (dfUser[self.user1_col].values, dfUser[self.user2_col].values)
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
        data = {}
        for element in self.df[self.layer_col].unique():
            newDf = self.df[self.df[self.layer_col] == element]
            weight = newDf[self.weight_col]
            W = spa.coo_matrix(
                    (
                        weight,
                        (newDf['top_index'].values, newDf['bottom_index'].values)
                    )
                )
            if self.importance_col != None:
                data[element] = (W, self.dfImportance[(self.dfImportance[self.layer_col] == element)][self.importance_col].values[0])
            else:
                data[element] = (W)
        self.data = data

    def generate_birank(self,  **kwargs):
        listOfW = []
        listOfZ = []
        for (W, importance) in self.data.values():
            listOfW.append(W)
            listOfZ.append(importance)
        p, u, z = centrality(listOfW, listOfZ, **kwargs)
        self.print_result(p, u, z)
    
    def generate_birank_cross_layer(self):
        # print(self.df['color'].unique())
        # for ()
        listOfW = []
        for (W) in self.data.values():
            listOfW.append(W)
        p, u = cross_layer(listOfW, self.pCrossWeight, self.uCrossWeight)
        self.print_result(p, u)

    def print_result(self, p, u, z=None):
        top_df = self.top_ids.copy()
        bottom_df = self.bottom_ids.copy() 
        counter=0
        for ele in self.data.keys():
            if z is not None:
                print(ele, "layer influence:", z[counter])
            else:
                print(ele)
            top_df[self.top_col + '_birank'] = p[counter]
            bottom_df[self.bottom_col + '_birank'] = u[counter]
            print(top_df[[self.top_col, self.top_col + '_birank']])
            print(bottom_df[[self.bottom_col, self.bottom_col + '_birank']])
            counter += 1

            
# testing for centrality
# df = pd.read_csv('data.csv', names=['color', 'product', 'weight', 'country'])
# dfImportance = pd.read_csv('importance.csv', names=['country', 'importance'])
# bn = BipartiteNetwork()
# bn.set_edgelist(df, 'color', 'product', 'country', 'importance', dfImportance, 'weight')
# bn.generate_birank()

#testing for cross_layer
df = pd.read_csv('twoLayerWeight.csv', names=['color', 'user', 'weight', 'country'])
bn = BipartiteNetwork()
bn.set_edgelist(df, 'color', 'user', 'country', weight_col='weight')
dfProduct = pd.read_csv('twoLayerCrossProduct.csv', names=['colour1', 'colour2', 'weightProduct'])
dfUser = pd.read_csv('twoLayerCrossUser.csv', names=['user1', 'user2', 'weightUser'])
bn.set_cross_layer(dfProduct, dfUser, 'colour1', 'colour2','user1', 'user2', 'weightProduct', 'weightUser')
bn.generate_birank_cross_layer()
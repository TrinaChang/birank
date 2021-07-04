import pandas as pd
import numpy as np
import scipy
import scipy.sparse as spa

def centrality(listOfW, listOfZ, theta=0.85, delta=0.85, phi=0.85, gamma=0.85, lamb=0.85, xi=0.85, max_iter=200):
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
        betaP.append(kp / totalW)
        betaU.append(kp / totalW)
        # avoid divide by 0
        kp[np.where(kp==0)] += 1
        ku[np.where(ku==0)] += 1

        kp_ = spa.diags(1/kp)
        ku_ = spa.diags(1/ku)

        kp_bi = spa.diags(1/np.lib.scimath.sqrt(kp))
        ku_bi = spa.diags(1/np.lib.scimath.sqrt(ku))
        Sp = kp_bi.dot(WT).dot(ku_bi)
        spList.append(Sp)
        sdList.append(Sp.T)

        p0 = np.repeat(1 / kp_.shape[0], kp_.shape[0])
        p0List.append(p0)
        p_lastList.append(p0.copy())
        u0 = np.repeat(1 / ku_.shape[0], ku_.shape[0])
        u0List.append(u0)
        u_lastList.append(u0.copy())
    p = [0]*len(listOfW)
    u = [0]*len(listOfW)
    z = [0]*len(listOfW)
    for i in range(max_iter):
        # layer from a,b, ....
        for layer in range(len(listOfW)):
            p[layer] = theta * spList[layer].dot(u_lastList[layer]) + delta * p_lastList[layer] * listOfZ[layer] + (1 - theta - delta) * p0List[layer]
            u[layer] = phi * sdList[layer].dot(p_lastList[layer]) + gamma * u_lastList[layer] * listOfZ[layer] + (1 - phi - gamma) * u0List[layer]
            z[layer] = lamb * betaP[layer].dot(u_lastList[layer]) + xi * betaU[layer].dot(p_lastList) + (1 - lamb - xi) * listOfZ[layer]
            
            p_lastList[layer] = p[layer]
            u_lastList[layer] = u[layer]
            
    
    return p, u, z

class BipartiteNetwork:
    def init(self):
        self.data = {}
    def set_edgelist(self, df, top_col, bottom_col, layer_col, importance_col, dfImportance, weight_col=None):
        self.df = df
        self.dfImportance = dfImportance
        self.top_col = top_col
        self.bottom_col = bottom_col
        self.weight_col = weight_col
        self.layer_col = layer_col
        self.importance_col = importance_col
        self._index_nodes()
        self._generate_adj()
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
            data[element] = (W, self.dfImportance[(self.dfImportance[self.layer_col] == element)][self.importance_col].values[0])
        self.data = data

    def generate_birank(self,  **kwargs):
        listOfW = []
        listOfZ = []
        for (W, importance) in self.data.values():
            listOfW.append(W)
            listOfZ.append(importance)
        p, u, z = centrality(listOfW, listOfZ, **kwargs)
        top_df = self.top_ids.copy()
        bottom_df = self.bottom_ids.copy() 
        counter=0
        for ele in self.data.keys():
            print(ele)
            top_df[self.top_col + '_birank'] = p[counter]
            bottom_df[self.bottom_col + '_birank'] = u[counter]
            print(top_df[[self.top_col, self.top_col + '_birank']])
            print(bottom_df[[self.bottom_col, self.bottom_col + '_birank']])
            counter += 1

df = pd.read_csv('data.csv', names=['color', 'product', 'weight', 'country'])
dfImportance = pd.read_csv('importance.csv', names=['country', 'importance'])
bn = BipartiteNetwork()
bn.set_edgelist(df, 'color', 'product', 'country', 'importance', dfImportance, 'weight')
bn.generate_birank()
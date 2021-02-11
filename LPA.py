import pandas as pd
import numpy as np
from scipy.spatial import distance


def create_dvr(x):
    """ Creates the DVR table of the domain"""
    dvr = x.iloc[:,[0,2]].groupby('element').sum().sort_values(by='frequency_in_category', ascending=False)
    tot = sum(dvr['frequency_in_category'])
    dvr['global_weight'] = dvr/tot
    dvr.reset_index(inplace=True)
    dvr['rnk'] = range(1,len(dvr)+1)
    return dvr

def create_vector(df):
    """ Creates a vector for every category in the domain"""
    counts = df.groupby(['element','category_id']).sum().reset_index()
    sums = df.groupby(['category_id']).sum().reset_index()
    sums.columns = ['category_id','total_weight']
    
    vector = pd.merge(counts, sums, on='category_id')
    vector['local_weight'] = vector['frequency_in_category']/vector['total_weight']
    vector = vector.sort_values(['category_id','local_weight'], ascending=[True,False])

    lrnk = vector.groupby('category_id').rank(method='first', ascending=False)['local_weight']
    lrnk.name = 'lrnk'

    vector = pd.merge(vector,lrnk,left_index=True,right_index=True)
    return vector.reset_index(drop=True)

def calc_distance_summary(dvr_vec,sample_vec):
    """ Auxillary func for distance summary"""
    return sum(((sample_vec-dvr_vec)*np.log10(sample_vec/dvr_vec)))

def distance_summary(dvr,vecs):
    """ Calculates the distance of every category from the domain"""
    dvr_vec = np.array(dvr.sort_values('element')['global_weight'])
    dist_sum = pd.DataFrame(index = vecs.index, columns=['distance_summary'], data=0)

    for i in range(len(vecs)):
        sample_vec = np.array(vecs.iloc[i,:])
        dist_sum.iloc[i,:] = calc_distance_summary(dvr_vec,sample_vec)
    f_max = round(max(dist_sum['distance_summary']) + 0.005,3)
    if f_max > 1:
        dist_sum['distance_summary'] = dist_sum['distance_summary'] / f_max
    return dist_sum


def signatures(dvr, vecs, n, epsilon):
    """ Creates the signature for every category in the domain. Default signature length is 500"""
    dvr_vec = np.array(dvr.sort_values('element')['global_weight'])
    to_concat = []
    for j in range(len(vecs)):
        sample_vec = np.array(vecs.iloc[j,:])
        KL_vec = np.zeros(len(sample_vec))
        existing_vec = np.zeros(len(sample_vec))
        for i in range(len(KL_vec)):
            KL_vec[i] = ((sample_vec[i]-dvr_vec[i])*np.log10(sample_vec[i]/dvr_vec[i]))
            if sample_vec[i] != epsilon:
                existing_vec[i] = 1
        ind = np.argpartition(KL_vec,-n)[-n:]
        ind = ind[np.argsort(KL_vec[ind])]
        sigs = pd.DataFrame(columns=['category_id','element','KL', 'existing_element_flag'], index=range(n))
        sigs['category_id'] = vecs.index[j]
        sigs['element'] = np.array(vecs.columns)[ind][::-1]
        sigs['KL'] = KL_vec[ind][::-1]
        sigs['existing_element_flag'] = existing_vec[ind][::-1]
        to_concat.append(sigs)
    return pd.concat(to_concat)


def KLDS(signatures):
    """ Auxillary func for Sock Puppet Distance Calculation"""
    signatures['existing_element_flag'].replace(0,-1,inplace=True)
    signatures['KLR'] = signatures['KL'] * signatures['existing_element_flag']
    signatures['KLR'] = signatures['KLR'] + signatures['existing_element_flag']
    return signatures

def cross_categories(df):
    """ Auxillary func for Sock Puppet Distance Calculation"""
    a = list(df['category_id'].unique())
    b = list(df['category_id'].unique())
    index = pd.MultiIndex.from_product([a, b], names = ["user1", "user2"])
    return pd.DataFrame(index = index).reset_index()

def SPD(klds, cc):
    """ Calculates the L1 distance between every pair of categories in the domain (Sock Puppet Distance)"""
    spd = cc.copy()
    spd['distance_between_users'] = 0

    ## This code can be made more efficient by not calculating twice when user1 and user2 are interchanged, using the other masks
    for i in range(len(cc)):
        user1 = cc.iloc[i,0]
        user2 = cc.iloc[i,1]

        mask1 = klds['category_id'] == user1
        mask2 = klds['category_id'] == user2
        x = klds[mask1 | mask2]

        x = x.pivot(index='category_id', columns='element', values='KLR')
        csize = len(x.columns)

        #using just the data in klds and the rest is 0
        vec_a = np.array(x.fillna(0).iloc[0,:])
        vec_b = np.array(x.fillna(0).iloc[-1,:])

        spd_mask1 = spd['user1'] == user1
        spd_mask2 = spd['user1'] == user2
        spd_mask3 = spd['user2'] == user1
        spd_mask4 = spd['user2'] == user2

        spd.loc[spd[(spd_mask1  & spd_mask4)].index[0],'distance_between_users'] = distance.cityblock(vec_a,vec_b)/csize

    f_max = round(max(spd['distance_between_users']) + 0.005,3)
    if f_max > 1:
        spd['distance_between_users'] = spd['distance_between_users'] / f_max
    return spd

def SockPuppetDistance(signatures, df):
    """ Wrapper func for L1 Distance Calculation"""
    klds = KLDS(signatures)
    cc = cross_categories(df)
    return SPD(klds, cc)

def create_signatures(df,epsilon_frac=2,sig_length=500):
    """ Prepares the raw data and creates signatures for every category in the domain.
    Epsilon frac defines the size of epsilon, default is 1/(corpus size * 2)
    Sig_length defines the length of the signature, default is 500 """
    
    dvr = create_dvr(df)
    dvr_dic = dict(dvr.iloc[:,[0,2]].values)
    epsilon = 1/ (len(dvr) * epsilon_frac)

    vector = create_vector(df)
    num_of_elements = vector.iloc[:,[1,-1]].groupby('category_id').max()
    num_of_elements['beta'] = 1-((len(dvr)-num_of_elements['lrnk']) * epsilon)
    vecs = vector.pivot_table(values='local_weight',index='category_id',columns='element').fillna(0)
    vecs.replace(0,epsilon,inplace=True)
    
    return signatures(dvr,vecs,sig_length, epsilon)

def distance_from_world(df):
    """ Prepares the data and returns the distance of every category in the domain"""
    dvr = create_dvr(df)
    dvr_dic = dict(dvr.iloc[:,[0,2]].values)
    epsilon = 1/ (len(dvr) *2)

    vector = create_vector(df)
    num_of_elements = vector.iloc[:,[1,-1]].groupby('category_id').max()
    num_of_elements['beta'] = 1-((len(dvr)-num_of_elements['lrnk']) * epsilon)
    betas = num_of_elements.reset_index().iloc[:,[0,-1]]
    vecs = vector.pivot_table(values='local_weight',index='category_id',columns='element').fillna(0)

    beta_vecs = vecs.copy()
    for i in range(len(beta_vecs)):
        beta_vecs.iloc[i,:] = beta_vecs.iloc[i,:] * betas.iloc[i,1]  
    vecs.replace(0,epsilon,inplace=True)
    beta_vecs.replace(0,epsilon,inplace=True)

    return distance_summary(dvr,beta_vecs)

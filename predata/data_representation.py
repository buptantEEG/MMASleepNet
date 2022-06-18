from re import sub
from turtle import pos
import scipy
from scipy import signal
import numpy as np
import mne
import scipy.interpolate as spi

def EEG_Spectral_spatial_representation(x):
    # x shape [channel*time*samples]
    #channel :1->'F3-M2', 2->'F4-M1', 3->'C3-M2', 4->'C4-M1',5->'O1',6->'O2'
    print(x.shape)
    map_delta=np.zeros((16,16))
    map_theta=np.zeros((16,16))
    map_alpha=np.zeros((16,16))
    map_beta=np.zeros((16,16))
    map_gamma=np.zeros((16,16))
    
    for i in range(x.shape[0]):
        position0,position1=electrodes_mapping(i)
        # print(position0)
        delta,_=mne.time_frequency.psd_array_multitaper(x[i,:],100,0,4)
        theta,_=mne.time_frequency.psd_array_multitaper(x[i,:],100,4,8)
        alpha,_=mne.time_frequency.psd_array_multitaper(x[i,:],100,8,13)
        beta,_=mne.time_frequency.psd_array_multitaper(x[i,:],100,13,30)
        gamma,_=mne.time_frequency.psd_array_multitaper(x[i,:],100,30,50)
        delta=np.sum(np.array(delta))
        theta=np.sum(np.array(theta))
        alpha=np.sum(np.array(alpha))
        beta=np.sum(np.array(beta))
        gamma=np.sum(np.array(gamma))


        
        map_delta[position0,position1]=delta
        
        map_theta[position0,position1]=theta
        map_alpha[position0,position1]=alpha
        map_beta[position0,position1]=beta
        map_gamma[position0,position1]=gamma
    map=np.array([map_delta,map_theta,map_alpha,map_beta,map_gamma]) #shape:[5,16,16]
    del map_alpha,map_beta,map_delta,map_theta,map_gamma
    # map=np.expand_dims(map,axis=-1)

    return map

def electrodes_mapping(channel):
    if channel==0:
        return 4,3
    if channel==3:
        return 4,12
    if channel==1:
        return 7,4
    if channel==4:
        return 7,11
    if channel==2:
        return 14,5
    if channel==5:
        return 14,10
    

# def multi_model_representation():
    
    
# def EEG_Temporal_representation():
if __name__=='__main__':
    subject_map=[]
    for i in range(1000):
        map_delta=np.zeros((16,16))
        map_theta=np.zeros((16,16))
        map_alpha=np.zeros((16,16))
        map_beta=np.zeros((16,16))
        map_gamma=np.zeros((16,16))
        map=np.array([map_delta,map_theta,map_alpha,map_beta,map_gamma]) #shape:[5,16,16]
        # print(map.shape)
        map=np.expand_dims(map,axis=-1)
        if i ==0:
            subject_map=list(map)
        else:
            subject_map.append(list(map))
        print(map.shape)
    subject_map=np.concatenate(subject_map)
    print(subject_map.shape)
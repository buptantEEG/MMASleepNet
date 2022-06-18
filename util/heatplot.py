import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
#sleepedf-20
# data = np.array([[ 7592,   462,   127,    36,    68],
# [  307,  1564,   564,     8 ,  361],
# [  189 ,  490 ,15836,   674  , 610],
# [    5 ,    7,   409,  5281,     1],
# [   90,   386 ,  573 ,    7  ,6661]])
# font2 = {
#     'weight' : 'normal',
#     'size' : 13
#     }
# sns.heatmap(data=data,annot=True,cmap='Blues',fmt='d')
# x = [0.5,1.5,2.5,3.5,4.5]
# label = ['W','N1','N2','N3','REM']
# plt.yticks(x,label,rotation=0)
# plt.xticks(x,label,rotation=0)
# plt.xlabel('target',font2,labelpad = 8.5)
# plt.ylabel('predict',font2,labelpad = 8.5)
# plt.show()
# plt.savefig('./MMASleep_sleepedf_20.jpg')

# sleepedf-78
data = np.array( [[60198,  4547,   562,   114,   530],
 [ 2640, 10820,  6153,   104 , 1805],
 [  314 , 4627 ,59527 , 2312,  2352],
 [   26 ,   42 , 2260 ,10708 ,    3],
 [  460,  2294,  2673,   138 ,20270]])

font2 = {
    'weight' : 'normal',
    'size' : 13
    }
sns.heatmap(data=data,annot=True,cmap='Greens',fmt='d')
x = [0.5,1.5,2.5,3.5,4.5]
label = ['W','N1','N2','N3','REM']
plt.yticks(x,label,rotation=0)
plt.xticks(x,label,rotation=0)
plt.xlabel('target',font2,labelpad = 8.5)
plt.ylabel('predict',font2,labelpad = 8.5)
plt.show()
plt.savefig('./MMASleep_sleepedf_78.jpg')


#isruc
# data = np.array([[1486,  146,   28 ,   7  ,  7],
#  [ 117,  725,  214 ,   6,  155],
#  [  29 , 205, 2145 , 170  , 67],
#  [   8 ,   2 , 252 ,1754,    0],
#  [   7,  110  , 21  ,  2  ,926]])
# font2 = {
#     'weight' : 'normal',
#     'size' : 13
#     }
# sns.heatmap(data=data,annot=True,cmap='Oranges',fmt='d')
# x = [0.5,1.5,2.5,3.5,4.5]
# label = ['W','N1','N2','N3','REM']
# plt.yticks(x,label,rotation=0)
# plt.xticks(x,label,rotation=0)
# plt.xlabel('target',font2,labelpad = 8.5)
# plt.ylabel('predict',font2,labelpad = 8.5)
# plt.show()
# plt.savefig('./MMASleep_isruc.jpg')
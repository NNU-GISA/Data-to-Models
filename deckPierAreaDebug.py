# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:13:23 2019

@author: Alan
"""


#Run working function, then run these
pierAreaBroke = np.load("pierAreaBroke.npy")
deckAreaBroke = np.load("deckAreaBroke.npy")
res = []
for da, dab in zip(deckArea, deckAreaBroke):
    r1 = da == dab
    res.append(np.any(r1 & False))
    
print(np.any(res))

res = []
for pa, pab in zip(pierArea, pierAreaBroke):
    r1 = pa == pab
    res.append(np.any(r1 & False))
    
print(np.any(res))
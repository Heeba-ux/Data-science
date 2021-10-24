# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 00:46:57 2021

@author: DELL
"""


word = "Grow Gratitude"
letter=word[0]
letter
len(word)
print(word.count('G'))

word1 = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
print(word1[-1])
print (word1[0:3])


word2 = "stay positive and optimistic"
a =word2.split(' ')
a
word2.startswith("H")
word2.endswith("d")
word2.endswith("c")

print( " 🪐 "*108)

word4 = "Grow Gratitude"

word4.replace("Grow", "Growth of")

dir(string)

string = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"
print (''.join(reversed(string)))


import pandas as pd
import numpy as np
from scipy import stats
df= pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/wight score.csv",encoding='latin1')
df
df.describe

df.Points.mean()
df.Score.mean()
df.Weigh.mean() 

df.Points.median()
df.Score.median()
df.Weigh.median()

stats.mode(df.Points)
stats.mode(df.Score)
stats.mode(df.Weigh)

df.Points.mode()
df.Score.mode()
df.Weigh.mode()

df.Points.var()
df.Score.var()
df.Weigh.var() # variance

df.Points.std()
df.Score.std()
df.Weigh.std() #standard deviation

range = max(df.Points) - min(df.Points) # range
range
range = max(df.Score) - min(df.Score) # range
range
range = max(df.Weigh) - min(df.Weigh) # range
range

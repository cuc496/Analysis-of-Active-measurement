# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:24:08 2019

@author: cuc496
"""

def getcost(H, costs):
    res = 0
    for s in H:
        res += costs[s]
    return res

def getweightfromsNotInH(H, s, weights, sets):
    res = 0
    eH = {}
    for i in range(len(H)):
        for e in sets[H[i]]:
            eH[e] = True
    for ele in sets[s]:
        if ele not in eH:
            res += weights[ele]
    return res

def greedy(H, X, U, weights, costs, sets, L):
    currentCost = getcost(H, costs)
    while len(U)>0:# and len(X)!=0:
        #greedy: select the best set
        a = float("-inf")
        selecteds = 0
        for s in U:
            c = costs[s]
            if c==0:
                c=10e-10
            sa = getweightfromsNotInH(H, s, weights, sets)/c
            if sa>=a:
                a = sa
                selecteds = s
        if currentCost+costs[selecteds]<=L:
            currentCost += costs[selecteds]
            H.append(selecteds)
        #for element in sets[selecteds]:
        #    if element in X:
        #        X.remove(element)
        U.remove(selecteds)
    return H

def getweightfromH(H, weights, sets):
    res = 0
    eDic = {}
    for s in H:
        for e in sets[s]:
            eDic[e] = True
    #print(eDic)
    for e in eDic:
        res += weights[e]
    return res
"""
L: limit of the sum of cost
X=[e1, e2, ..., en]
weights={e1 => w1, e2 => w2, ...}
S=[s1, s2, ..., sm]
sets={s1=>[e1,e2], s2=>[], }
costs={s1 => c1, s2 => c2, ...}
H = a subset of S
U = a subset of S
"""
def getWMCSet(S, X, weights, sets, costs, L):
    wH1 = float("-inf")
    H1 = []
    #the reason of running |subsets|==2 first: we want to make |subsets| as less as possible.
    #if both |subsets|==1 and |subsets|==2 can cover all element in X, |subsets|==1 will be the solution of H1
    #|subsets|==2
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            H = [S[i], S[j]]
            if getcost(H, costs)<=L:
                w = getweightfromH(H, weights, sets)
                if wH1<=w:
                    wH1=w
                    H1=H
    print("done with |subsets|==2")                
    #|subsets|==1
    for s in S:
        H = [s]
        if getcost(H, costs)<=L:
            w = getweightfromH(H, weights, sets)
            if wH1<=w:
                wH1=w
                H1=H
    print("done with |subsets|==1")
    print("weight(H1):{}, budget-cost(H1):{}".format(wH1, getcost(H1, costs)))
    
    
    wH2 = float("-inf")
    H2 = []
    
    #|subsets|==3
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                H = [S[i], S[j], S[k]]
                if getcost(H, costs)<=L:
                    U = S[:]
                    U.remove(S[i])
                    U.remove(S[j])
                    U.remove(S[k])
                    H = greedy(H, X[:], U, weights, costs, sets, L)
                    w = getweightfromH(H, weights, sets)
                    if wH2<=w:
                        wH2=w
                        H2=H

    print("weight(H2):{}, budget-cost(H2):{}".format(wH2, getcost(H2, costs)))
    #print("H2: {}".format(H2))
    if wH1>wH2:
        return H1
    else:
        return H2
    
    
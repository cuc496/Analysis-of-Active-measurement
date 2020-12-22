from pygraphml import Graph
from pygraphml import GraphMLParser
import numpy as np
import optimizer as opt
import itertools
import greedy as gd
import operator
import plottopology as plg
import plotbargraph as plb
import CKR
import TmLP
import pickle
import dijkstra as dij
import importMatlab as ipM
import pathHandler as paH
import WMCappro as WMC
import datetime
import sys
import OPTsolver
from gurobipy import GRB
import multiprocessing
import time
import random
from numpy import linalg as LA
import OPTsolver

def printDelta(delta):
    paths = []
    for i in range(len(delta)):
        if delta[i]==1:
            paths.append(i)
    return paths

def intersectwithPd(Upaths, datapaths):
    linksPd = []
    res = 0
    for p in datapaths:
        for l in p:
            if l not in linksPd:
                linksPd.append(l)

    for p in Upaths:
        isinPd = False
        for l in p:
            if l in linksPd:
                isinPd = True
                break
        if isinPd:
            res += 1
    return res

def checkpaths(paths, datapaths, delta, datapaths_index):
    isVal = True
    for i in range(len(datapaths)):
        if datapaths[i] not in paths:
            isVal = False
    if not isVal:
        print("some datapaths are NOT included in paths")
        print("paths:{}".format(sorted(printDelta(delta))))
        print("datapaths:{}".format(sorted(datapaths_index)))
        raise

def random_delta(datapaths, upaths, cost_d, budget_d):
    len_P = len(upaths)
    delta = np.zeros(len_P)
    candiP = [i for i in range(len_P)]
    for i in range(len_P):
        if upaths[i] in datapaths:
            candiP.remove(i)
            delta[i] = 1

    while len(candiP)>0:
        selected = candiP[np.random.randint(0, len(candiP))]
        if (cost_d[selected] <= budget_d):
            delta[selected] = 1
            budget_d -= cost_d[selected]
        candiP.remove(selected)
    return delta

def getArr(paths, edges):
    arr = np.zeros((len(paths), len(edges)))
    for i in range(len(paths)):
        for e in paths[i]:
            j = edges.index(e)
            arr[i,j] = 1
    return arr

def getVec(weight, edges):
    vec = np.zeros(len(edges))
    for j in range(len(edges)):
        vec[j] = weight[edges[j]]
    return vec

def getPaths(delta, UPaths):
    paths = []
    for i in range(len(delta)):
        if delta[i] == 1:
            paths.append(UPaths[i])
    return paths


def getMaxDelayBydictLmx0(datapaths, paths, lm, t, tmax, x0, edges):

    PdM = getArr(datapaths, edges)
    Pn = []
    for i in range(len(paths)):
        if not paH.isCompromisedPath(paths[i], lm):
            Pn.append(paths[i])
    x0M = []
    lmM = []
    
    for edge in edges:
        x0M.append(x0[edge])
        if edge not in lm:
            lmM.append(0)
        else:
            lmM.append(1)
    if lm == []:
        return 0, np.array(x0M)
    PnM = getArr(Pn, edges)
    return OPTsolver.maxdam(PdM, PnM, lmM, t, tmax, x0M)

def getMaxDelay(datapaths, paths, lmM, t, tmax, x0, edges):
    PdM = getArr(datapaths, edges)
    lm = paH.getlm(lmM, edges)
    Pn = []
    for i in range(len(paths)):
        if not paH.isCompromisedPath(paths[i], lm):
            Pn.append(paths[i])

    PnM = getArr(Pn, edges)
    return OPTsolver.maxdam(PdM, PnM, lmM, t, tmax, x0)

def Tmdualy(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges, d):
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    delta0 = np.full(len(upaths), d)
    y_, tm0 = OPTsolver.minTmDualByDelta(delta0, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
    y = []
    for i in range(len(y_)):
        y.append(y_[i])
    delta, _ = OPTsolver.minTmDualByY(np.array(y), uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
    return delta

def DPTmdual(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    if budget_d == float('inf'):
        return np.full(len(upaths), 1)
    budget_d = int(budget_d)
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    f = [[ 0 for _ in range(len(upaths)+1)] for _ in range(budget_d + 1)]
    delta_arr = [[ np.zeros(len(upaths)) for _ in range(len(upaths)+1) ] for _ in range(budget_d + 1)]
    for i in range(1, len(upaths) + 1):
        for j in range(len(cost_d)):
            if cost_d[j]==0:
                delta_arr[0][i][j] = 1

    _, tm0 = OPTsolver.minTmDualByDelta(delta_arr[0][0], uPM, np.array(cost_a), budget_a, np.array(cost_d), None, wd)

    for i in range(len(upaths) + 1):
        _, tm = OPTsolver.minTmDualByDelta(delta_arr[0][i], uPM, np.array(cost_a), budget_a, np.array(cost_d), None, wd)
        f[0][i] = tm0 - tm

    for i in range(1, len(upaths)+1):
        for k in range(1, budget_d + 1):
            if cost_d[i-1]<= k:
                delta0 = delta_arr[k - int(cost_d[i-1])][i - 1].copy()
                _, tm0 =  OPTsolver.minTmDualByDelta(delta0, uPM, np.array(cost_a), budget_a, np.array(cost_d), None, wd)
                delta0[i-1] = 1
                _, tm = OPTsolver.minTmDualByDelta(delta0, uPM, np.array(cost_a), budget_a, np.array(cost_d), None, wd)
                diff = tm0-tm
                if diff <= 0:
                    diff = 0
                if f[k][i - 1] > f[k - int(cost_d[i-1])][i - 1] + diff:
                    f[k][i] =  f[k][i - 1]
                    delta_arr[k][i] = delta_arr[k][i - 1].copy()
                else:
                    f[k][i] = f[k - int(cost_d[i-1])][i - 1] + diff
                    delta_arr[k][i] = delta0.copy()
            else:
                f[k][i] = f[k][i - 1]
                delta_arr[k][i] = delta_arr[k][i-1].copy()
            #print(D)
    return delta_arr[budget_d][len(upaths)].copy()


def effDPTmdual(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    if budget_d == float('inf'):
        return np.full(len(upaths), 1)
    budget_d = int(budget_d)
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    f = [[ 0 for _ in range(len(upaths)+1)] for _ in range(budget_d + 1)]

    delta_arr = [[ None for _ in range(len(upaths)+1) ] for _ in range(budget_d + 1)]
    delta0 = np.zeros(len(upaths))
    for i in range(len(upaths)):
        if cost_d[i]==0:
            delta0[i] = 1
    for i in range(len(upaths) + 1):
        delta_arr[0][i] = delta0.copy()
    for k in range(budget_d + 1):
        delta_arr[k][0] = delta0.copy()

    _, tm0 = OPTsolver.minTmDualByDelta(delta0, uPM, np.array(cost_a), budget_a, np.array(cost_d), None, wd)

    v = [ 0 for _ in range(len(upaths))]
    for i in range(len(upaths)):
        if cost_d[i] == 0:
            continue
        delta0[i] = 1
        _, tm = OPTsolver.minTmDualByDelta(delta0, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
        diff = (tm0 - tm)
        if diff <= 0:
            diff = 0
        v[i] = diff/cost_d[i]
        delta0[i] = 0

    for i in range(1, len(upaths)+1):
        for k in range(1, budget_d + 1):
            if cost_d[i-1]<= k:
                if f[k][i - 1] >= f[k - int(cost_d[i-1])][i - 1] + v[i-1]:
                    f[k][i] =  f[k][i - 1]
                    delta_arr[k][i] = delta_arr[k][i - 1].copy()
                else:
                    f[k][i] = f[k - int(cost_d[i-1])][i - 1] + v[i-1]
                    delta_arr[k][i] = delta_arr[k - int(cost_d[i-1])][i - 1].copy()
                    delta_arr[k][i][i-1] = 1
            else:
                f[k][i] = f[k][i - 1]
                delta_arr[k][i] = delta_arr[k][i-1].copy()
            #print(D)
    return delta_arr[budget_d][len(upaths)].copy()


def effDPTmdual2(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    return


def greedyTmdual(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    if budget_d == float('inf'):
        return np.full(len(upaths), 1)
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    candidate = [1 for _ in range(len(upaths))] #0: cost is larger than budget_d or has been selected, 1: cost is smaller or equal to budget_d
    res_delta = np.zeros(len(upaths))
    for i in range(len(upaths)):
        if cost_d[i] == 0:
            candidate[i] = 0
            res_delta[i] = 1

    while True:
        tmp_delta = res_delta.copy()
        _, tm0 = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
        eff = [float('-inf') for _ in range(len(upaths))]
        for i in range(len(upaths)):
            if candidate[i] == 0:
                continue
            if cost_d[i] > budget_d:
                candidate[i] = 0
                continue
            tmp_delta[i] = 1
            _, tm = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
            diff = (tm0 - tm)
            if diff <= 0:
                diff = 0
            eff[i] = diff/cost_d[i]
            tmp_delta[i] = 0
        #print(eff)
        if max(candidate) == 0 or max(eff) == float('-inf'):
            break

        selected = eff.index(max(eff))
        print("selected: {} with eff:{}".format(selected, eff[selected]))
        res_delta[selected] = 1
        #print("delta paths:{}".format(sorted(printDelta(res_delta))))
        budget_d -= cost_d[selected]
        candidate[selected] = 0

    return res_delta


def effGreedyTmdual(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    if budget_d == float('inf'):
        return np.full(len(upaths), 1)
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    res_delta = np.zeros(len(upaths))
    for i in range(len(upaths)):
        if cost_d[i] == 0:
            res_delta[i] = 1

    tmp_delta = res_delta.copy()
    _, tm0 = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
    eff = [float('-inf') for _ in range(len(upaths))]
    for i in range(len(upaths)):
        if cost_d[i] == 0:
            continue
        if cost_d[i] > budget_d:
            continue
        tmp_delta[i] = 1
        _, tm = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
        diff = (tm0 - tm)
        if diff <= 0:
            diff = 0
        eff[i] = diff/cost_d[i]
        tmp_delta[i] = 0

    while budget_d>0:
        #remove the paths unaffordable from candidate paths
        for i in range(len(upaths)):
            if cost_d[i] > budget_d:
                eff[i] = float('-inf')
        if max(eff) == float('-inf'):
            break

        selected = eff.index(max(eff))
        print("selected: {} with eff:{}".format(selected, eff[selected]))
        res_delta[selected] = 1

        budget_d -= cost_d[selected]
        eff[selected] = float('-inf')

    return res_delta

def effGreedyTmdual2(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges, l):
    if budget_d == float('inf'):
        return np.full(len(upaths), 1)
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    candidate = [1 for _ in range(len(upaths))] #0: cost is larger than budget_d or has been selected, 1: cost is smaller or equal to budget_d
    res_delta = np.zeros(len(upaths))
    for i in range(len(upaths)):
        if cost_d[i] == 0:
            candidate[i] = 0
            res_delta[i] = 1

    for _ in range(l):
        tmp_delta = res_delta.copy()
        _, tm0 = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
        eff = [float('-inf') for _ in range(len(upaths))]
        for i in range(len(upaths)):
            if candidate[i] == 0:
                continue
            if cost_d[i] > budget_d:
                candidate[i] = 0
                continue
            tmp_delta[i] = 1
            _, tm = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
            diff = (tm0 - tm)
            if diff <= 0:
                diff = 0
            eff[i] = diff/cost_d[i]
            tmp_delta[i] = 0
        #print(eff)
        if max(candidate) == 0 or max(eff) == float('-inf'):
            break

        selected = eff.index(max(eff))
        print("selected: {} with eff:{}".format(selected, eff[selected]))
        res_delta[selected] = 1
        #print("delta paths:{}".format(sorted(printDelta(res_delta))))
        budget_d -= cost_d[selected]
        candidate[selected] = 0

    tmp_delta = res_delta.copy()
    _, tm0 = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
    eff = [float('-inf') for _ in range(len(upaths))]
    for i in range(len(upaths)):
        if candidate[i] == 0:
            continue
        if cost_d[i] > budget_d:
            continue
        tmp_delta[i] = 1
        _, tm = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
        diff = (tm0 - tm)
        if diff <= 0:
            diff = 0
        eff[i] = diff/cost_d[i]
        tmp_delta[i] = 0

    while budget_d>0:
        #remove the paths unaffordable from candidate paths
        for i in range(len(upaths)):
            if cost_d[i] > budget_d:
                eff[i] = float('-inf')
        if max(eff) == float('-inf'):
            break

        selected = eff.index(max(eff))
        print("selected: {} with eff:{}".format(selected, eff[selected]))
        res_delta[selected] = 1

        budget_d -= cost_d[selected]
        eff[selected] = float('-inf')

    return res_delta

"""

def greedyTmdual(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    candidate = [1 for _ in range(len(upaths))] #0: cost is larger than budget_d or has been selected, 1: cost is smaller or equal to budget_d
    res_delta = np.zeros(len(upaths))
    #for i in range(len(upaths)):
        #if cost_d[i] == 0:
            #cost_d[i] = 1e-10
            #candidate[i] = 0
            #res_delta[i] = 1
    while True:
        tmp_delta = res_delta.copy()
        _, tm0 = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
        eff = [float('-inf') for _ in range(len(upaths))]
        for i in range(len(upaths)):
            if candidate[i] == 0:
                continue
            if cost_d[i] > budget_d:
                candidate[i] = 0
                continue
            tmp_delta[i] = 1
            _, tm = OPTsolver.minTmDualByDelta(tmp_delta, uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
            diff = tm0 - tm
            if diff <= 1e-10:
                diff = 0
            if cost_d[i] == 0:
                if diff == 0:
                    diff = 1
                eff[i] = diff/1e-15
            else:
                eff[i] = diff/cost_d[i]
            tmp_delta[i] = 0
        #print(eff)
        if max(candidate) == 0 or max(eff) == float('-inf') or max(eff)<=0:
            break

        selected = eff.index(max(eff))
        #print("selected: {} with eff:{}".format(selected, eff[selected]))
        res_delta[selected] = 1
        #print("delta paths:{}".format(sorted(printDelta(res_delta))))
        budget_d -= cost_d[selected]
        candidate[selected] = 0

    return res_delta
"""

def SDPTmdual(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    return OPTsolver.SDPminTmDual(uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)

#single process, it might take too long
def minTmdual_(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    return OPTsolver.minTmDual(uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)

def minTmdual(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges, delta_list, mintmdual_list):
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    return_delta, return_mintmdual = OPTsolver.minTmDual(uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd)
    for i in range(len(return_delta)):
        delta_list.append(return_delta[i])
    mintmdual_list.append(return_mintmdual)

def minTmdualQP(upaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges, rounding):
    uPM = getArr(upaths, edges)
    wd = getVec(weightsByPd, edges)
    return OPTsolver.minTmDualQP(uPM, np.array(cost_a), budget_a, np.array(cost_d), budget_d, wd, rounding)

def getLinksTraversedByPaths(paths):
    links = []
    for p in paths:
        for link in p:
            if link not in links:
                links.append(link)
    return links

def MCPd(Upaths, datapaths, cost_a, budget_a, cost_d, budget_d, weightsByPd, edges):
    """
    L: limit of the sum of cost
    X=[e1, e2, ..., en]
    weights={e1 => w1, e2 => w2, ...}
    S=[s1, s2, ..., sm]
    sets={s1=>[e1,e2], s2=>[], }
    costs={s1 => c1, s2 => c2, ...}
    H = a subset of S
    U = a subset of S
    getWMCSet(S, X, weights, sets, costs, L):
    """
    X = getLinksTraversedByPaths(datapaths)
    weights = {}
    for x in X:
        weights[x] = 1
    S = [i for i in range(len(Upaths))]
    sets={}
    for i in range(len(S)):
        sets[i] = []
        for link in Upaths[i]:
            if link in X:
                sets[i].append(link)
    costs = {}
    for i in range(len(S)):
        costs[S[i]] = cost_d[i]
    L = budget_d
    selectedPaths = WMC.greedy([], X, S, weights, costs, sets, L)
    delta = [0 for _ in range(len(Upaths))]
    for i in selectedPaths:
        delta[i] = 1
    return delta

#GRB.BINARY, GRB.CONTINUOUS
def maxTm(paths, cost_a, budget_a, weightsByPd, edges, vType):
    PM = getArr(paths, edges)
    wd = getVec(weightsByPd, edges)
    return OPTsolver.maxTm(PM, np.array(cost_a), budget_a, wd, vType)

def OPTmax(costsM, paths, edges, budget, x0M, t, tmax):
    PM=getArr(paths, edges)
    return OPTsolver.attackMILP(PM, np.array(costsM), budget, t, tmax, np.array(x0M), edges)

def OPTmaxRR(costsM, paths, edges, budget, x0M, t, tmax):
    PM=getArr(paths, edges)
    return OPTsolver.attackMILPRR(PM, np.array(costsM), budget, t, tmax, np.array(x0M), edges)

def heuristic(costs, paths, edges, budget, x0, t, tmax):
    candidateEdges = edges.copy()
    lm = []
    maxdelay = 0
    while len(candidateEdges)>0 and budget>0:
        #maxdelay = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
        optEdge = None
        for edge in candidateEdges:
            if costs[edge]>budget:
                candidateEdges.pop(candidateEdges.index(edge))
                continue
            lm.append(edge)
            delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
            lm.pop()
            if delayDegrade>maxdelay:
                maxdelay = delayDegrade
                optEdge = edge
        if optEdge is None:
            break
        lm.append(optEdge)
        candidateEdges.pop(candidateEdges.index(optEdge))
        budget = budget - costs[optEdge]
    return lm

if __name__ == "__main__":
    programStart = datetime.datetime.now()
    log = False
    checkPdincluded = True
    caseid = "WMC-1"

    #maximum seconds for Tmdual IQP
    tQP = 2*3600
    nap = 10
    cycle = 6
    #set up initial value:t, tmax
    initrange = 15
    t = 15
    tmax = 200
    numberMonitor = 10#30#20#15#8#10#10#15#4
    numberMonitors = [numberMonitor]#[4,5,6,7,8]#[20,21,22,23,24,25]#[10,11,12,13,14,15]#[4,5,6,7,8]#[10,11,12,13,14,15]
    loops = 20 #how many times for each number of monitors

    #monitors, paths
    monitorsConst = "one"#"onetwo" for Bics, "one" for others #one=> place monitors at 1-degree nodes; any=> place monitors at any nodes
    degreeLimit = 1
    #numberPathsBtw2M = float('inf')

    #SP: shortest path, All: all possible path, SPECIAL: for the conterexample of Tm
    routing = "SP"


    lenPs = []#[20, 1000, 5000, 10000, 20000]#[240, 280, 320, 360, 400]#[100,120,140,160,180]#[50,60,70,80,90]
    #lenpds = [10]#[240]#[100][50]#[240, 280, 320, 360, 400]#[100,120,140,160,180]#[50,60,70,80,90]#[100,120,140,160,180]#[50,60,70,80,90]#[200]#[200]#[100]#[50]#[200, 240, 280, 320, 360, 400]#[50,60,70,80,90]#[50,60,70,80,90]#[100,120,140,160,180]
    #budget = float("inf")
    budgets_a = [1,3, 5,7,9, float('inf')]#[5,7,9,11,13, float('inf')]#[budget]#[21,22,23,24,25,-1]#[9,10,11,12,13,-1]#[2,3,4,5,6,-1]
    #budgets_d = [0, 5, 10, 15, 20, float("inf")]#[5, 10, 15, 20, float("inf")]#[5, 9, 13, 17, float('inf')]
    xaxis = budgets_a
    xlabel = "attack budget" #budget_a, pd, P
    
    costMax = 2


    #**** QP always first for maxtQP
    methods = ["MILP", "TmILP", "ALS greedy", "CALS greedy", "heuristic greedy", "MILP-RR", "top traversal", "random"]#["WMC_def", "random_def"]# ["Tmdual_def", "Tmdual_def_greedy", "WMC_def", "random_def"]#["Tmdual_def_SDP"]#["Tmdual_def", "Tmdual_def_greedy", "Tmdual_def_DP", "random_def"]#"Tmdual_def_DP",["Tmdual_def_RR", "Tmdual_def_HR"]#["Tmdual_def", "random_def"]
    
    #methods = ['Tmdual_def', "Tmdual_def_greedy_eff","Tmdual_def_greedy_eff2", 'Tmdual_def_greedy',"Tmdual_def_DP_eff", "Tmdual_def_DP_eff2", 'Tmdual_def_DP', 'random_def']
    ginfos = ["nodes", "paths", "edges", "costs", "datapaths"]
    """
    #get the graph
    parser = GraphMLParser()
    gname= "Colt"#"Colt"#"Bics"#"BeyondTheNetwork"#"Cogentco"#"bridge2"#"BeyondTheNetwork"#"Getnet"
    g = parser.parse("./archive/{}.graphml".format(gname))
    """
    #pmlengths = ["top 1 traversal", "avg"]
    gname="AS8717"#"AS8717"#"AS20965"#"AS8717"
    g = ipM.loadgraph("./MatlabData/CAIDA_{}.mat".format(gname))
    
    #f = open('{}-log'.format(gname), 'w')
    if log:
        f = open('{}-log'.format(gname), 'w')
        sys.stdout = f


 
    #cost_d_cand = paH.getCostsAtRandom(Upaths, costDEFmu, costDEFsigma, seed=0)
    #cost_d_cand = paH.getCostsAtRandomInt(Upaths, costInth, seed=0)

    expinfo = {}
    allres = [{me:[] for me in methods} for _ in range(len(xaxis))]
    alltms = [{me:[] for me in methods} for _ in range(len(xaxis))]
    alllms = [{me:[] for me in methods} for _ in range(len(xaxis))]
    allpros = [{me:[] for me in methods} for _ in range(len(xaxis))]
    alltimes = [{me:[] for me in methods} for _ in range(len(xaxis))]
    #allginfo = [{ginfo:[] for ginfo in ginfos} for _ in range(len_)]
    expinfo["tQP"] = tQP
    expinfo["costMax"] = costMax
    expinfo["t"]=t
    expinfo["tmax"]=tmax
    expinfo["numberMonitors"]=numberMonitors
    expinfo["loops"]=loops
    expinfo["monitorsConst"]=monitorsConst
    expinfo["gname"]=gname
    expinfo["budgets_a"] = budgets_a
    #expinfo["budgets_d"] = budgets_d
    #expinfo["lenpds"] = lenpds
    expinfo["lenPs"] = lenPs
    expinfo["# of nodes with {} degree".format(degreeLimit)]= paH.getnNodes(g, degreeLimit)
    expinfo["init"]="[0,{})".format(initrange)
    print(expinfo)

    linksPd = []
    seeds = []
    seed = 0
    detectedFraction = []
    while len(seeds)<loops:
        paths = paH.getUpath(g, numberMonitor, monitorsConst, routing, seed=seed)
        print("len(paths): {}".format(len(paths)))
        nodes = paH.getNodes(paths)
        edges = paH.getEdges(paths)
        print("# of nodes:{}".format(len(nodes)))
        print("# of edges:{}".format(len(edges)))            
        weights = paH.getWeight(paths, edges)
        
        #set up x0 (initial value)
        x0M = paH.getEdgesX0AtRandom(edges, initrange)
        costsM = paH.getEdgesX0AtRandom(edges, costMax)
        x0 = {}
        costs = {}
        for i in range(len(edges)):
            x0[edges[i]] = x0M[i]
            costs[edges[i]] = costsM[i]

        tmp_res = [{me:0 for me in methods} for _ in range(len(xaxis))]
        tmp_tms = [{me:0 for me in methods} for _ in range(len(xaxis))]
        tmp_lms = [{me:0 for me in methods} for _ in range(len(xaxis))]
        tmp_pros = [{me:0 for me in methods} for _ in range(len(xaxis))]
        tmp_times = [{me:0 for me in methods} for _ in range(len(xaxis))]
        print("=====seed: {}=====".format(seed))
        #for bi_a in range(len(budgets_a)):
        #    budget_a = budgets_a[bi_a]
        for bi in range(len(budgets_a)):
            budget = budgets_a[bi]
            print("---budget:{}----".format(budget))
            for method in methods:
                methodStart = datetime.datetime.now()
                print("-method: {}-".format(method))
                if method=="ALS greedy" or method=="greedy":
                    lm = gd.greedy(weights, costs, paths, paths, budget, isCALS=False)
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                    #PdM = getArr(paths, edges)
                    #print("obj: {}".format(delayDegrade))
                    #print("obj by calculation: {}".format(LA.norm(PdM.dot(xhat - x0M), 1)))
                elif method=="CALS greedy":
                    lm = gd.greedy(weights, costs, paths, paths, budget, isCALS=True)
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                    """
                    PM = getArr(paths, edges)
                    res_detected = OPTsolver.detectorMILP(PM, PM.dot(xhat), t, tmax)
                    #lmM = np.zeros(len(edges))
                    #for link in lm:
                    #    lmM[edges.index(link)] = 1
                    #print("lmM:{}".format(lmM))
                    #print("xhat:{}".format(xhat))
                    print("sum(res_detected): {}".format(sum(res_detected)))
                    print(res_detected)
                    detected = 0
                    for j in range(len(res_detected)):
                        if res_detected[j]==1 and edges[j] in lm:
                            detected +=1
                    detectedFraction.append(detected/len(lm))
                    print(detectedFraction)
                    """
                elif method=="heuristic greedy":
                    lm = heuristic(costs, paths, edges, budget, x0, t, tmax)
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                elif method=="CKR":
                    lm = CKR.getLm(weights, monitors)
                    CKRlm = len(lm)
                    #pos=plg.plot(g, weights, monitors, lm)
                elif method=="random":
                    lm = paH.randomLm(edges, budget, costs, paths)
                    if paH.isCut(lm, paths):
                        print("Lm forms a cut for paths")
                    else:
                        print("Lm NOT forms a cut for paths")
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                elif method=="top traversal":
                    lm = paH.topTraversal(budget, weights, costs, paths)
                    if paH.isCut(lm, paths):
                        print("Lm forms a cut for paths")
                    else:
                        print("Lm NOT forms a cut for paths")
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                    #pm diverse
                elif method=="modified CALS greedy":
                    pathsweights = paH.getPathsWeights(paths, datapaths)
                    edgesTopaths = paH.getPlbyPathIndex(paths)
                    pathset = [i for i in range(len(paths))]
                    lm = WMC.getWMCSet(edges, pathset, pathsweights, edgesTopaths ,costs, budget)
                    upbObjv = tmax*(np.exp(1)/(np.exp(1)-1))*(paH.getTm_(lm, paths))
                    #allres[xindexplot]["upper bound"].append(upbObjv)
                elif method=="TmLP-H":
                    lm = TmLP.getLm(costs, paths, datapaths, edges, budget, isILP=False, rounding="heuristic")
                elif method=="TmILP":
                    lm = TmLP.getLm(costs, paths, paths, edges, budget, isILP=True, rounding="")
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                elif method=="TmLP-RR":
                    lm = TmLP.getLm(costs, paths, paths, edges, budget, isILP=False, rounding="random")
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                elif method=="TmLP-greedy":
                    lm = TmLP.getLm(costs, paths, datapaths, edges, budget, isILP=False, rounding="greedy")
                elif method=="TmLP-greedy2":
                    lm = TmLP.getLm(costs, paths, datapaths, edges, budget, isILP=False, rounding="greedy2")
                elif method=="all":
                    lm = edges   
                elif method=="MILP":
                    delayDegrade, lm, xhat = OPTmax(costsM, paths, edges, budget, x0M, t, tmax)
                elif method=="MILP-RR":
                    _, lm = OPTmaxRR(costsM, paths, edges, budget, x0M, t, tmax)
                    delayDegrade, xhat = getMaxDelayBydictLmx0(paths, paths, lm, t, tmax, x0, edges)
                print("obj value:{}".format(delayDegrade))
                tmp_res[bi][method] = delayDegrade
                tmp_pros[bi][method] = len(paths)
                tmp_lms[bi][method] = len(lm)
                methodEnd = datetime.datetime.now()
                period = methodEnd - methodStart
                tmp_times[bi][method] = period.seconds
                print("{} takes {}".format(method, period))

        for i in range(len(xaxis)):
            for method in methods:
                allres[i][method].append(tmp_res[i][method])
                alltms[i][method].append(tmp_tms[i][method])
                alllms[i][method].append(tmp_lms[i][method])
                allpros[i][method].append(tmp_pros[i][method])
                alltimes[i][method].append(tmp_times[i][method])
        
        plb.plot(allres, methods, xaxis, xlabel, gname, None, False, "")
        savedfolder = "./attack_variedka/{}".format(gname)
        res_out = open("{}/res.pickle".format(savedfolder),"wb")
        pickle.dump(allres, res_out)
        res_out.close()

        tms_out = open("{}/tms.pickle".format(savedfolder),"wb")
        pickle.dump(alltms, tms_out)
        tms_out.close()

        lms_out = open("{}/lms.pickle".format(savedfolder),"wb")
        pickle.dump(alllms, lms_out)
        lms_out.close()

        pros_out = open("{}/pros.pickle".format(savedfolder),"wb")
        pickle.dump(allpros, pros_out)
        pros_out.close()

        times_out = open("{}/times.pickle".format(savedfolder),"wb")
        pickle.dump(alltimes, times_out)
        times_out.close()

        expinfo_out = open("{}/expinfo.pickle".format(savedfolder),"wb")
        pickle.dump(expinfo, expinfo_out)
        expinfo_out.close()
        
        seeds.append(seed) ###removed when seed_ is asigned
        expinfo["seeds"]=seeds
        print("seeds:{}".format(seeds))
        
        #print time
        programEnd = datetime.datetime.now()
        print ("total time:{}".format(programEnd - programStart))

        
        print(seeds)
        seed = seed + 1 ###removed when seed_ is asigned
    if log:
        f.close()

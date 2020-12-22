from operator import itemgetter, attrgetter
import pathHandler as paH
from gurobipy import *
import numpy
import operator
import random

def sumCost(lm, costs):
    sumCost = 0
    for edge in lm:
        sumCost += costs[edge]
    return sumCost
    
def getri(paths, edges):
    R = []
    for i in range(len(paths)):
        r = []
        for j in range(len(edges)):
            if edges[j] in paths[i]:
                r.append(1)
            else:
                r.append(0)
        R.append(r)
    return R

def LPsolver(costs, paths, datapaths, edges, budget, isILP):
    rd = getri(datapaths, edges)
    r = getri(paths, edges)
    model = Model("optimization")
    if isILP:
        x = model.addVars(len(edges), vtype=GRB.BINARY)#GRB.CONTINUOUS
        y = model.addVars(len(edges), vtype=GRB.BINARY)
        z = model.addVars(len(edges), vtype=GRB.BINARY)
    else:
        x = model.addVars(len(edges), vtype=GRB.CONTINUOUS)
        y = model.addVars(len(edges), vtype=GRB.CONTINUOUS)
        z = model.addVars(len(edges), vtype=GRB.CONTINUOUS)
    
    model.addConstrs(x[j] <=1 for j in range(len(edges)))
    model.addConstrs(y[j] <=1 for j in range(len(edges)))
    model.addConstrs(z[j] <=1 for j in range(len(edges)))    
    model.addConstrs(z[j] <= 1 - x[j] for j in range(len(edges)))
    model.addConstrs(z[j] <= y[j] for j in range(len(edges)))
    model.addConstrs(z[j] >= y[j] - x[j] for j in range(len(edges)))
    
    model.addConstr(quicksum(x[j]*costs[edges[j]] for j in range(len(edges)))<= budget)
    
    for i in range(len(r)):
        rx = model.addVar()
        model.addConstr(rx == quicksum(r[i][j]*x[j] for j in range(len(edges))))
        model.addConstrs(rx >= r[i][j]*y[j] for j in range(len(edges)))
        
    model.setObjective(quicksum(quicksum(rd[i][j]*z[j] for j in range(len(edges))) for i in range(len(rd))), GRB.MAXIMIZE)
    #model.Params.LogFile = "TmLP"
    model.Params.OutputFlag = 0
    model.optimize()
    if model.status != 2:
        print("Model not OPTIMAL - status: ", model.status)
        raise
    
    res = []
    for j in range(len(edges)):
        res.append(x[j].x)
    return res
def randomR(x):
    rounded = []
    for i in range(len(x)):
        rounded.append(1 if (random.random() <= x[i]) else 0)
    return rounded
def getLm(costs, paths, datapaths, edges, budget, isILP, rounding):
    Lm = []
    x = LPsolver(costs, paths, datapaths,edges, budget, isILP)
    #print(x)
    if isILP:
        for j in range(len(edges)):
            if x[j]==1:
                Lm.append(edges[j])
    else:
        if rounding=="heuristic":
            while (not paH.isCut(Lm, paths)) and (max(x)!=-1):
                i = x.index(max(x))
                if costs[edges[i]] > budget:
                    x[i]=-1
                else:
                    Lm.append(edges[i])
                    budget -= costs[edges[i]]
                    x[i]=-1
        elif rounding=="random":
            while sumCost(Lm, costs)>budget or Lm==[]:
                Lm=[]
                roundedx = randomR(x)
                #print("rounded x:{}".format(roundedx))
                for j in range(len(edges)):
                    if roundedx[j]==1:
                        Lm.append(edges[j])
        elif rounding=="greedy":
            xPerCost = []
            for j in range(len(edges)):
                xPerCost.append((edges[j], x[j]/costs[edges[j]], costs[edges[j]], x[j]))
            xPerCost = sorted(xPerCost, key=itemgetter(1), reverse=True)
            #print("xPerCost:{}".format(xPerCost))
            for j in range(len(edges)):
                cost = xPerCost[j][2]
                lj = xPerCost[j][0]
                if cost <= budget:
                    Lm.append(lj)
                    budget -= cost
                    if paH.isCut(Lm, paths) or budget <=0:
                        break
        elif rounding=="greedy2":
            xPerCost = {}
            for j in range(len(edges)):
                xPerCost[edges[j]] = x[j]/costs[edges[j]]
            Pathsleft=paths.copy()
            while Pathsleft:
                pl = paH.getPl(Pathsleft)
                maxC = float("-inf")
                for e, _ in pl.items():
                    if costs[e]<=budget:
                        c = len(pl[e])*xPerCost[e]
                        if c > maxC:
                            maxC = c
                            l = e
                #a termination condition: when no link can be compromised because of the budget limit
                if maxC == float("-inf"): 
                    break
                Lm.append(l)
                budget -= costs[l]
                for path in pl[l]:
                    #update the paths being cut
                    Pathsleft.remove(path)
    return Lm

from gurobipy import *
import numpy

def maxdam(rn, rm, lm, t, tmax, x0, edges):
    numberPathn = rn.shape[0]
    numberPathm = rm.shape[0]
    numberLinks = len(edges)
    performanceUpperBound = numpy.array([tmax]*numberLinks)
    
    for e in lm:
        performanceUpperBound[edges.index(e)] = t
    
    m = Model("optimization")
    xe = m.addVars(numberLinks, vtype=GRB.CONTINUOUS)
    
    #objective function
    m.setObjective(quicksum(quicksum(rm[i,j]*xe[j] for j in range(numberLinks))- rm.dot(x0)[i] for i in range(numberPathm)), GRB.MAXIMIZE)
    
    #rn remain the same
    m.addConstrs(quicksum(rn[i,j]*xe[j] for j in range(numberLinks)) == rn.dot(x0)[i]  for i in range(numberPathn))
    
    #upper bound for performance
    m.addConstrs(xe[i] <= performanceUpperBound[i] for i in range(numberLinks))
    m.Params.OutputFlag = 0
    m.optimize()
    if m.status != 2:
        print("Model not OPTIMAL - status: ", m.status)
        raise
        
    xee = numpy.zeros(numberLinks)
    
    for i in range(numberLinks):
        xee[i] = xe[i].x
    objv = numpy.sum(rm.dot(xee-x0))
    return (objv, m.ObjVal, lm, xee)
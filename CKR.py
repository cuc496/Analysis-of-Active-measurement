from gurobipy import *
import numpy
import operator

def edgebetw(u,v):
    for e in u.edges():
        if e in v.edges():
            return e
    return -1

def getNodes(weights):
    nodes = []
    for e, w in weights.items():
        if e.node1 not in nodes:
            nodes.append(e.node1)
        if e.node2 not in nodes:
            nodes.append(e.node2)
    return nodes

def getEdges(weights):
    edges = []
    for e, w in weights.items():
        if e not in edges:
            edges.append(e)
    return edges

def relaxation(nodes, edges, weights, monitors):
    numberMonitors = len(monitors)
    numberNodes = len(nodes)
    numberEdges = len(edges)
    nodeOrder = {}
    for i, node in enumerate(nodes):
        nodeOrder[node] = i
    monitorOrder = {}
    for j, m in enumerate(monitors):
        monitorOrder[m] = j
    edgeOrder = {}
    for i, e in enumerate(edges):
        edgeOrder[e] = i
    
    model = Model("optimization")
    
    #probs[nodeOrder][monitorOrder]
    probs = model.addVars(numberNodes, numberMonitors, vtype=GRB.CONTINUOUS) 
    
    dist = model.addVars(numberEdges, numberMonitors, vtype=GRB.CONTINUOUS)
    for e in edges:
        for m in monitors:
            d = model.addVar(lb=-GRB.INFINITY)
            model.addConstr(d==(probs[nodeOrder[e.node1], monitorOrder[m]] - probs[nodeOrder[e.node2], monitorOrder[m]]))
            model.addGenConstrAbs(dist[edgeOrder[e], monitorOrder[m]], d, "absconstr")
    model.setObjective((1/2)*quicksum(w*quicksum(dist[edgeOrder[e], monitorOrder[m]] for m in monitors) for e, w in weights.items()), GRB.MINIMIZE)
    
    #the sum of probability is equal to 1
    model.addConstrs(quicksum(probs[i, j] for j in range(numberMonitors))== 1 for i in range(numberNodes))
    
    #the probability must be equal to 1, if it is a monitor
    #print("monitorOrder:{}".format(monitorOrder))
    ###
    #for n, o in nodeOrder.items():
    #    print("id:{}, order:{}".format(n.id, o))
    #print("nodeOrder:{}".format(nodeOrder))
    ###
    
    model.addConstrs(probs[nodeOrder[m], monitorOrder[m]]==1 for m in monitors)
    model.Params.OutputFlag = 0
    model.optimize()
    if model.status != 2:
        print("Model not OPTIMAL - status: ", model.status)
        raise

    ###
    #for m in monitors:
    #    print("monitor:")
    #    for i in range(numberMonitors):
    #        print(probs[nodeOrder[m], i].x)
    ###
    
    
    res = {}
    for n in nodes:
        res[n] = numpy.zeros(numberMonitors)
        for i in range(numberMonitors):
            res[n][i]=probs[nodeOrder[n], i].x
    return res
def getCut(probs, weights, numberMonitors, edges):
    ###
    #edgeOrder = {}
    #for i, e in enumerate(edges):
    #    edgeOrder[e] = i
    ###
    Lm = []
    for e, w in weights.items():
        #print("e:{}".format(edgeOrder[e]))
        #print("e.node1:{}, {}".format(e.node1.id, probs[e.node1]))
        #print("e.node2:{}, {}".format(e.node2.id, probs[e.node2]))
        for i in range(numberMonitors):
            
            if probs[e.node1][i] != probs[e.node2][i]:
                if e not in Lm:
                    Lm.append(e)
    return Lm
    
def getLm(weights, monitors):
    nodes = getNodes(weights)
    edges = getEdges(weights)
    probs = relaxation(nodes, edges, weights, monitors)
    print(probs)
    numberMonitors = len(monitors)
    monitorOrder = {}
    for j, m in enumerate(monitors):
        monitorOrder[m] = j
    
    #sumW: monitor=>sum(weight(e)*|probs(e.node1)-probs(e.node2)|)
    sumW={}
    for m in monitors:
        sumW[m]=0
        for e, w in weights.items():
            sumW[m]= sumW[m] + w*(numpy.abs(probs[e.node1][monitorOrder[m]] - probs[e.node2][monitorOrder[m]]))
    listW=sorted(sumW.items(), key=operator.itemgetter(1))
    #print(listW)
    
    q = 0.0
    while q==0.0:
        q = numpy.random.random()
    #print("q:{}".format(q))
    p = numpy.random.randint(0,2)
    #print("p:{}".format(p))
    if p==0:
        rang = [i for i in range(numberMonitors-1)]            
    else:
        rang = [i for i in range(numberMonitors-2,-1, -1)]
    #print("probs:{}".format(probs))
    
    rang.append(numberMonitors-1)
    #print("rang:{}".format(rang))
    for i in rang:
        #setting up mi: the position of the monitor in the vector monitors
        mm = listW[i][0]
        mi = monitorOrder[mm]
        rmNode = []
        #print("mi:{}".format(mi))         
        for n in nodes:
            #print("n:{}".format(n))
            #print("before probs[n]:{}".format(probs[n]))
            if i == (numberMonitors-1) or probs[n][mi]>=q:
                probs[n].fill(0)
                probs[n][mi] = 1.0
                rmNode.append(n)
            #print("after probs[n]:{}".format(probs[n]))
        for n in rmNode:
            nodes.remove(n)
        #print("probs:{}".format(probs))
    #print("probs:{}".format(probs))
    return getCut(probs, weights, numberMonitors, edges)
        
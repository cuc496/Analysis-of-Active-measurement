import numpy as np
import gurobipy as gp
import pathHandler as paH
from gurobipy import GRB
import random
import sys
#import mosek
#import mosek.fusion as mf

def randomR(x):
    rounded = []
    for i in range(len(x)):
        rounded.append(1 if (random.random() <= x[i]) else 0)
    return rounded

def getAi(len_L, i):
    arr = np.zeros((len_L, len_L))
    arr[i,i] = 1
    return arr

def isTooMuch(delta, cost, k):
    s = 0
    for i in range(len(delta)):
        if delta[i]==1:
            s += cost[i]
    return s>k

def constructA(R, c_a):
    len_P = R.shape[0]
    len_L = R.shape[1]
    arr = np.concatenate((-R, 
                          np.dot(R, getAi(len_L, 0)), 
                          np.zeros((len_P, len_L))), axis=1)
    for j in range(1, len_L):
        arr_ = np.concatenate((-R, 
                            np.dot(R, getAi(len_L, j)), 
                            np.zeros((len_P, len_L))), axis=1)
        arr = np.concatenate((arr, arr_), axis = 0)
    tmp = np.concatenate((c_a.reshape(1, c_a.shape[0]), np.zeros((1, 2*len_L))), axis = 1)
    arr = np.concatenate((arr, tmp), axis = 0)
    arr = np.concatenate((arr, np.identity(3*len_L)), axis = 0)
    tmp = np.concatenate((np.identity(len_L), np.zeros((len_L,len_L)), np.identity(len_L)), axis = 1)
    arr = np.concatenate((arr, tmp), axis = 0)
    tmp = np.concatenate((np.zeros((len_L, len_L)), -np.identity(len_L), np.identity(len_L)), axis = 1)
    arr = np.concatenate((arr, tmp), axis = 0)
    tmp = np.concatenate((-np.identity(len_L), np.identity(len_L), -np.identity(len_L)), axis = 1)
    arr = np.concatenate((arr, tmp), axis = 0)
    return arr

def islandmatrix(dim, i):
    res = np.zeros((dim, dim))
    res[i,i] = 1
    return res

def islandvector(dim, i):
    res = np.zeros(dim)
    res[i] = 1
    return res
    
def constructSDPPara(len_P, len_L, R, c_a, c_d, wd, budget_a):
    i_row = np.zeros((len_P, 6*len_L+1))
    for _ in range(len_L):
        i_row = np.concatenate((-np.identity(len_P), i_row), axis = 1)
    leftHalf = np.concatenate((np.zeros((len_L*(len_P + 6) + 1, len_L*(len_P + 6) + 1)), i_row), axis=0)
    rightHalf = np.concatenate((np.transpose(i_row), np.zeros((len_P, len_P))), axis=0)
    q0 = np.full((len_P + 6)*len_L + 1 + len_P, 1.0)
    q0[len_L*len_P] = budget_a
    for i in range(len_L*len_P + 1 + 4*len_L, len(q0)):
        q0[i] = 0.0
    A0 = np.concatenate((leftHalf, rightHalf), axis=1)
    A1 = np.transpose(np.concatenate((constructA(R, c_a), np.zeros((len_P, 3*len_L))), axis = 0))
    r1 = np.concatenate((np.zeros(2*len_L), wd), axis = 0)
    A2lefthalf = np.zeros((len_L*(len_P + 6) + 1 + len_P, len_L*(len_P + 6) + 1))
    A2righthalf = np.concatenate((np.zeros((len_L*(len_P + 6) + 1, len_P)), np.identity(len_P)), axis=0)
    A2 = np.concatenate((A2lefthalf, A2righthalf), axis = 1)
    r2 = np.concatenate((np.zeros(len_L*(len_P + 6) + 1), np.full(len_P, 1)), axis=0)
    q3 = np.concatenate((np.zeros(len_L*(len_P + 6) + 1), c_d), axis=0)
    #q0 = q0.reshape((q0.size, 1))
    r1 = r1.reshape((r1.size, 1))
    r2 = r2.reshape((r2.size, 1))
    #q3
    return 0.5*A0, q0, A1, r1, A2, r2, q3
    
def SDPminTmDual(uPM, c_a, k_a, c_d, k_d, wd):
    delta = []
    len_P = uPM.shape[0]
    len_L = uPM.shape[1]
    A0, q0, A1, r1, A2, r2, q3 = constructSDPPara(len_P, len_L, uPM, c_a, c_d, wd, k_a)
    with mf.Model("SDP relaxation") as M:
        y_dim = len_L*(len_P + 6) + 1 + len_P
        y = M.variable(y_dim , mf.Domain.greaterThan(0.0))
        Y = M.variable([y_dim, y_dim], mf.Domain.unbounded())
        # Create constraints
        M.constraint(mf.Expr.sub(mf.Expr.mul(A1, y), r1), mf.Domain.greaterThan(0.0))
        M.constraint(mf.Expr.sub(mf.Expr.mul(A2, y), r2), mf.Domain.lessThan(0.0))
        
        M.constraint(mf.Expr.dot(q3, y), mf.Domain.lessThan(k_d))

        for i in range(len_L*(len_P + 6) + 1, len_L*(len_P + 6) + 1 + len_P):
            M.constraint(mf.Expr.sub(mf.Expr.dot(islandmatrix(y_dim, i), Y), mf.Expr.dot(islandvector(y_dim, i),y)), mf.Domain.greaterThan(0.0))
        
        M.constraint(mf.Expr.hstack(mf.Expr.vstack(Y, mf.Expr.transpose(y)), mf.Expr.vstack(y, 1.0)), mf.Domain.inPSDCone())
        #print("q0.shape:{}".format(q0.shape))
        #print("y_dim:{}".format(y_dim))
        #print("A0.shape:{}".format(A0.shape))
        #print("y_dim:{}".format(y_dim))
        #M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(q0,y),3))
        M.objective(mf.ObjectiveSense.Minimize, mf.Expr.add(mf.Expr.dot(A0,Y), mf.Expr.dot(q0,y)))
        #M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(C, X), x.index(0)))
        # Solve the problem
        M.solve()

        # Get the solution values
        sol = y.level()
        for i in range(len_L*(len_P + 6) + 1, len_L*(len_P + 6) + 1 + len_P):
            delta.append(sol[i])
        print(delta)
    
    return delta
    


def minTmDualByDelta(delta, uPM, c_a, k_a, c_d, k_d, wd):
    len_P = uPM.shape[0]
    len_L = uPM.shape[1]
    A = constructA(uPM, c_a)
    try:
        # Create a new model
        m = gp.Model("defense")

        # Create variables
        y = m.addMVar(shape=len_L*(len_P+6) + 1, lb=0.0, ub=GRB.INFINITY)
        
        # Set objective
        b = np.concatenate((np.full(len_L*(len_P+4) + 1,1), np.zeros(2*len_L)), axis = 0)
        b[len_L*len_P] = k_a
        m.setObjective(b @ y - sum([delta @ y[i*len_P:(i+1)*len_P] for i in range(len_L)]), GRB.MINIMIZE)
    
        # Add constraints
        c = np.concatenate((np.zeros(2*len_L), wd), axis = 0)
        m.addConstr(A.T @ y >= c)
    
        # Optimize model
        m.Params.OutputFlag = 0
        m.optimize()
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    return  y.X, m.objVal

def minTmDualByY(y, uPM, c_a, k_a, c_d, k_d, wd):
    len_P = uPM.shape[0]
    len_L = uPM.shape[1]
    A = constructA(uPM, c_a)
    try:
        # Create a new model
        m = gp.Model("defense")
        #m.Params.NonConvex=0
        
        # Create variables
        delta = m.addMVar(shape=len_P, vtype=GRB.BINARY)
        
        # Set objective
        b = np.concatenate((np.full(len_L*(len_P+4) + 1,1), np.zeros(2*len_L)), axis = 0)
        b[len_L*len_P] = k_a
        m.setObjective(b @ y - sum([delta @ y[i*len_P:(i+1)*len_P] for i in range(len_L)]), GRB.MINIMIZE)
    
        # Add constraints
        m.addConstr(c_d @ delta <= k_d)
    
        # Optimize model
        m.Params.OutputFlag = 0
        m.optimize()

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    return delta.X, m.objVal


#PM: paths P matrix (R)
#the info of datapathM is captured by wd
def maxTm(PM, c_a, k_a, wd, vType):
    len_P = PM.shape[0]
    len_L = PM.shape[1]
    A = constructA(PM, c_a)
    lmM = np.zeros(len_L)
    try:
        #create a new model
        m = gp.Model("attack")
        
        #creat variables
        x = m.addMVar(shape=3*len_L, vtype=vType)
        
        #set objective
        c = np.concatenate((np.zeros(2*len_L), wd), axis = 0)
        m.setObjective(c @ x, GRB.MAXIMIZE)
        
        #add constraints
        b = np.concatenate((np.zeros(len_L*len_P + 1), np.full(4*len_L,1), np.zeros(2*len_L)), axis = 0)
        b[len_L*len_P] = k_a
        m.addConstr(A @ x <= b)
        
        #optimize model
        m.Params.OutputFlag = 0
        m.optimize()

        for j in range(len_L):
            lmM[j] = x.X[j]
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    return lmM, m.objVal

"""
R.shape = (|P|, |L|) 
c_a.shape, wd.shape = (|L|)
c_d.shape = (|P|)
k_a, k_d float
the info of datapathM is captured by wd
"""        
def minTmDual(uPM, c_a, k_a, c_d, k_d, wd):
    len_P = uPM.shape[0]
    len_L = uPM.shape[1]
    A = constructA(uPM, c_a)
    try:
        # Create a new model
        m = gp.Model("defense")
        
        # Create variables
        delta = m.addMVar(shape=len_P, vtype=GRB.BINARY)
        y = m.addMVar(shape=len_L*(len_P+6) + 1, lb=0.0, ub=GRB.INFINITY)
        
        # Set objective
        b = np.concatenate((np.full(len_L*(len_P+4) + 1,1), np.zeros(2*len_L)), axis = 0)
        b[len_L*len_P] = k_a
        m.setObjective(b @ y - sum([delta @ y[i*len_P:(i+1)*len_P] for i in range(len_L)]), GRB.MINIMIZE)
    
        # Add constraints
        c = np.concatenate((np.zeros(2*len_L), wd), axis = 0)
        m.addConstr(A.T @ y >= c)
        m.addConstr(c_d @ delta <= k_d)
    
        # Optimize model
        m.Params.OutputFlag = 0
        m.optimize()
        y_ = []
        for i in range(len_L*len_P):
            y_.append(y.X[i])
        #print("y_:{}".format(y_))
        print("max y_: {}".format(max(y_)))
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    return delta.X, m.objVal


def minTmDualQP(uPM, c_a, k_a, c_d, k_d, wd, rounding):
    len_P = uPM.shape[0]
    len_L = uPM.shape[1]
    A = constructA(uPM, c_a)
    try:
        # Create a new model
        m = gp.Model("defense")
        m.Params.NonConvex=2
        
        # Create variables
        delta = m.addMVar(shape=len_P, lb=0.0, ub=1.0)
        y = m.addMVar(shape=len_L*(len_P+6) + 1, lb=0.0, ub=GRB.INFINITY)
        
        # Set objective
        b = np.concatenate((np.full(len_L*(len_P+4) + 1,1), np.zeros(2*len_L)), axis = 0)
        b[len_L*len_P] = k_a
        m.setObjective(b @ y - sum([delta @ y[i*len_P:(i+1)*len_P] for i in range(len_L)]), GRB.MINIMIZE)
    
        # Add constraints
        c = np.concatenate((np.zeros(2*len_L), wd), axis = 0)
        m.addConstr(A.T @ y >= c)
        m.addConstr(c_d @ delta <= k_d)
        
        # Optimize model
        m.Params.OutputFlag = 0
        m.optimize()
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')

    tmp_delta = []
    for i in range(len_P):
        tmp_delta.append(delta[i].x)
    
    if rounding=="heuristic":
        res_delta = []
        for i in range(len_P):
            res_delta.append(0)
        
        while (max(tmp_delta)!=-1):
            i = tmp_delta.index(max(tmp_delta))
            if c_d[i] <= k_d:
                res_delta[i]=1
                k_d -= c_d[i]
            tmp_delta[i]=-1
            
    elif rounding=="random":
        res_delta = randomR(tmp_delta)
        while isTooMuch(res_delta, c_d, k_d):
            res_delta = randomR(tmp_delta)
            
    return res_delta, m.objVal

def detectorMILP(PM, y, t, tmax):
    numberLinks = PM.shape[1]
    print("numberLinks:{}".format(numberLinks))
    try:
        m = gp.Model("minimum bad links")
        #decision variables
        delta = m.addMVar(shape=numberLinks,lb=0, vtype=GRB.BINARY)
        x = m.addMVar(shape=numberLinks, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        #objective
        m.setObjective(sum(delta), GRB.MINIMIZE)
        #constraints
        m.addConstr(PM @ x == y)
        m.addConstrs(x[j] <= (tmax*delta[j] + t*(1-delta[j])) for j in range(numberLinks))
        #optimize model
        m.Params.OutputFlag = 0
        m.optimize()
        if m.status != 2:
            print("Model not OPTIMAL - status: ", m.status)
            raise
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    delta_ = []
    for j in range(numberLinks):
        delta_.append(delta.X[j])
    return np.array(delta_)

def attackMILP(PM, costs, budget, t, tmax, x0, edges):
    numberPaths = PM.shape[0]
    numberLinks = PM.shape[1]
    #print("numberPaths: {}".format(numberPaths))
    #print("numberLinks: {}".format(numberLinks))
    m = gp.Model("maximize attack")
    #decision variables
    alpha = m.addMVar(shape=numberLinks, lb=0, vtype=GRB.BINARY)
    beta = m.addMVar(shape=numberPaths, lb=0, vtype=GRB.BINARY)
    xe = m.addMVar(shape=numberLinks, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    #objective
    m.setObjective(sum(sum(PM[i,j]*xe[j] for j in range(numberLinks))- PM.dot(x0)[i] for i in range(numberPaths)), GRB.MAXIMIZE)
    #constraints
    m.addConstrs(sum(PM[i,j]*xe[j] for j in range(numberLinks))- PM.dot(x0)[i] <= tmax*beta[i]*sum(PM[i,k] for k in range(numberLinks)) for i in range(numberPaths) )
    m.addConstrs(sum(PM[i,j]*xe[j] for j in range(numberLinks)) >= PM.dot(x0)[i] for i in range(numberPaths) )
    m.addConstr(beta <= PM @ alpha)
    m.addConstrs(xe[j] <= t*alpha[j] + tmax*(1-alpha[j]) for j in range(numberLinks))
    m.addConstr(alpha @ costs <= budget)
    #optimize model
    m.Params.OutputFlag = 0
    m.optimize()
    if m.status != 2:
        print("Model not OPTIMAL - status: ", m.status)
        raise
    lm=[]
    xhat = []
    for j in range(numberLinks):
        xhat.append(xe.X[j])
        if alpha.X[j]==1:
            lm.append(edges[j])
    return  m.objVal, lm, np.array(xhat)

def attackMILPRR(PM, costs, budget, t, tmax, x0, edges):
    numberPaths = PM.shape[0]
    numberLinks = PM.shape[1]
    m = gp.Model("maximize attack")
    #decision variables
    alpha = m.addMVar(shape=numberLinks, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    beta = m.addMVar(shape=numberPaths, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    xe = m.addMVar(shape=numberLinks, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    #objective
    m.setObjective(sum(sum(PM[i,j]*xe[j] for j in range(numberLinks))- PM.dot(x0)[i] for i in range(numberPaths)), GRB.MAXIMIZE)
    #constraints
    m.addConstrs(sum(PM[i,j]*xe[j] for j in range(numberLinks))- PM.dot(x0)[i] <= tmax*beta[i]*sum(PM[i,k] for k in range(numberLinks)) for i in range(numberPaths) )
    m.addConstrs(sum(PM[i,j]*xe[j] for j in range(numberLinks)) >= PM.dot(x0)[i] for i in range(numberPaths) )
    m.addConstr(beta <= PM @ alpha)
    m.addConstrs(xe[j] <= t*alpha[j] + tmax*(1-alpha[j]) for j in range(numberLinks))
    m.addConstr(alpha @ costs <= budget)
    #optimize model
    m.Params.OutputFlag = 0
    m.optimize()
    if m.status != 2:
        print("Model not OPTIMAL - status: ", m.status)
        raise
    alpha_ = []
    for j in range(numberLinks):
        alpha_.append(alpha.X[j])
    roundedAlpha = randomR(alpha_)
    lm=[]
    for j in range(numberLinks):
        if roundedAlpha[j]==1:
            lm.append(edges[j])
    return  m.objVal, lm
def maxdam(PdM, PnM, lmM, t, tmax, x0):
    numberPathn = PnM.shape[0]
    numberPathm = PdM.shape[0]
    numberLinks = PdM.shape[1]
    performanceUpperBound = np.array([tmax]*numberLinks)
    
    for j in range(numberLinks):
        if lmM[j] == 1: #link_j is compromised 
            performanceUpperBound[j] = t
    try:
        m = gp.Model("max delay degradation")
        xe = m.addVars(numberLinks, vtype=GRB.CONTINUOUS)
        
        #objective function
        m.setObjective(gp.quicksum(gp.quicksum(PdM[i,j]*xe[j] for j in range(numberLinks))- PdM.dot(x0)[i] for i in range(numberPathm)), GRB.MAXIMIZE)
        
        #rn remain the same
        m.addConstrs(gp.quicksum(PnM[i,j]*xe[j] for j in range(numberLinks)) == PnM.dot(x0)[i]  for i in range(numberPathn))
        
        #upper bound for performance
        m.addConstrs(xe[i] <= performanceUpperBound[i] for i in range(numberLinks))
        m.Params.OutputFlag = 0
        m.optimize()
        if m.status != 2:
            print("Model not OPTIMAL - status: ", m.status)
            raise
            
        xee = np.zeros(numberLinks)
        
        for i in range(numberLinks):
            xee[i] = xe[i].x
            
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    except AttributeError:
        print('Encountered an attribute error')
    return m.ObjVal, xee

"""
if __name__ == "__main__":
    R = np.array([[1,1,1,1], [0,1,0,0], [1,0,0,0], [0,0,1,1]])
    c_a = np.array([1,1,1,1])
    k_a = 1
    c_d = np.array([0,0,1,1])
    k_d = 2
    wd = np.array([1,2,1,1])
    res_delta = minDegradeSolver(R, c_a, k_a, c_d, k_d, wd)
    
    tmp = []
    for i in range(len(res_delta)): 
        if res_delta[i]==1:
            tmp.append(R[i])
    R_ = np.array(tmp)
    
    maxTm(R_, c_a, k_a, wd)
"""
    
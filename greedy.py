import pygraphml
import pathHandler

def greedy(weights, costs, paths, datapaths, budget, isCALS):
    #initialize paths and links
    Pathsleft=paths.copy()
    Lm = []
    budgetcost = 0
    while Pathsleft:
        #update pl
        pl = pathHandler.getPl(Pathsleft)
        
        #CALS
        if isCALS:
            maxeffDiffTm = float("-inf")
            l = 0
            currentTm = pathHandler.getTm(Lm, datapaths)
            for e, _ in pl.items():
                if (budgetcost+costs[e])<=budget:
                    diffTm = pathHandler.getTm(Lm + [e], datapaths) - currentTm
                    if (diffTm/costs[e]) > maxeffDiffTm:
                        maxeffDiffTm = (diffTm/costs[e])
                        l=e
            #a termination condition: when no link can be compromised because of the budget limit
            if maxeffDiffTm == float("-inf"): 
                break
                    
        #ALS
        else:
            #set the initial value for greedy
            minC = float("inf")
            l = 0
            for e, _ in pl.items():
                if (budgetcost+costs[e])<=budget:
                    c = (weights[e]/len(pl[e]))
                    if c < minC:
                        minC = c
                        l = e
                        
            #a termination condition: when no link can be compromised because of the budget limit
            if minC == float("inf"): 
                break
        #update Lm by the best selection
        Lm.append(l)
        budgetcost += costs[l]
        
        for path in pl[l]:
            #update the paths being cut
            Pathsleft.remove(path)
            
        #printing process
        progress = (1-(len(Pathsleft)/(len(paths))))*100
        print("progress:{}".format(progress))
    print("budget-cost:{}".format(budgetcost))
    return Lm
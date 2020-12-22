import pickle
import plotbargraph as plb
import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
"""
#ALS(journal image)
allres = [[pickle.load(open("./experiment2/case2/res.pickle", "rb")), pickle.load(open("./experiment2/case1/res.pickle", "rb"))], 
           [pickle.load(open("./experiment2/case5/res.pickle", "rb")), pickle.load(open("./experiment2/case3/res.pickle", "rb"))], 
           [pickle.load(open("./experiment2/case7/res.pickle", "rb")), pickle.load(open("./experiment2/case8/res.pickle", "rb"))]]
for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["greedy ALS"] = allres[i][j][k]["greedy"]
numberMonitors = [[[4,5,6,7,8], [4,5,6,7,8]], [[10,11,12,13,14,15], [10,11,12,13,14,15]], [[20,21,22,23,24,25], [20,21,22,23,24,25]]]
methods = ["greedy ALS", "CKR", "top traversal", "random", "all"]
allginfo = [[pickle.load(open("./experiment2/case2/graph.pickle", "rb")), pickle.load(open("./experiment2/case1/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment2/case5/graph.pickle", "rb")), pickle.load(open("./experiment2/case3/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment2/case7/graph.pickle", "rb")), pickle.load(open("./experiment2/case8/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Colt", "Cogent"], ["AS 20965", "AS 8717"]]
#plb.plot(allres, methods, numberMonitors, gname, None, multiple=True)
plb.plot(allres, methods, numberMonitors, gname, allginfo, multiple=True)

#CALS
allres = [[pickle.load(open("./experiment-CALS/case 4/res.pickle", "rb")), pickle.load(open("./experiment-CALS/case 3/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-CALS/case 2/res.pickle", "rb")), pickle.load(open("./experiment-CALS/case 1/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-CALS/case 5/res.pickle", "rb")), pickle.load(open("./experiment-CALS/case 6/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["greedy ALS"] = allres[i][j][k]["ALS greedy"]
            allres[i][j][k]["greedy CALS"] = allres[i][j][k]["CALS greedy"]

numberMonitors = [[[2, 3, 4, 5, 6, "$\infty$"], [2, 3, 4, 5, 6, "$\infty$"]], [[9, 10, 11, 12, 13, "$\infty$"], [9, 10, 11, 12, 13, "$\infty$"]], [[21,22,23,24,25,"$\infty$"], [21,22,23,24,25,"$\infty$"]]]
methods = ["greedy ALS", "greedy CALS", "top traversal", "random"]
allginfo = [[pickle.load(open("./experiment-CALS/case 4/graph.pickle", "rb")), pickle.load(open("./experiment-CALS/case 3/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-CALS/case 2/graph.pickle", "rb")), pickle.load(open("./experiment-CALS/case 1/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-CALS/case 5/graph.pickle", "rb")), pickle.load(open("./experiment-CALS/case 6/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Colt", "Cogent"], ["AS 20965", "AS 8717"]]
#plb.plot(allres, methods, numberMonitors, gname, None, multiple=True)
plb.plot(allres, methods, numberMonitors, gname, allginfo, multiple=True)


#mCALS
allres = [[pickle.load(open("./experiment-WMC/Bics/res.pickle", "rb")), pickle.load(open("./experiment-WMC/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/Cogentco/res.pickle", "rb")), pickle.load(open("./experiment-WMC/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/AS8717/res.pickle", "rb")), pickle.load(open("./experiment-WMC/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["greedy ALS"] = allres[i][j][k]["ALS greedy"]
            allres[i][j][k]["greedy CALS"] = allres[i][j][k]["CALS greedy"]
            allres[i][j][k]["modified greedy CALS"] = allres[i][j][k]["modified CALS greedy"]
numberMonitors = [[[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13,"$\infty$"], [5,7,9,11,13,"$\infty$"]]]
methods = ["modified greedy CALS", "greedy CALS", "greedy ALS", "top traversal", "random"]
allginfo = [[pickle.load(open("./experiment-WMC/Bics/graph.pickle", "rb")), pickle.load(open("./experiment-WMC/BeyondTheNetwork/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/Cogentco/graph.pickle", "rb")), pickle.load(open("./experiment-WMC/Colt/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/AS8717/graph.pickle", "rb")), pickle.load(open("./experiment-WMC/AS20965/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
#plb.plot(allres, methods, numberMonitors, gname, None, multiple=True)
plb.plot(allres, methods, numberMonitors, gname, allginfo, multiple=True)


#mCALS with upper-bound(journal image)
allres = [[pickle.load(open("./experiment-WMC/Bics/res.pickle", "rb")), pickle.load(open("./experiment-WMC/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/Cogentco/res.pickle", "rb")), pickle.load(open("./experiment-WMC/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/AS8717/res.pickle", "rb")), pickle.load(open("./experiment-WMC/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["greedy ALS"] = allres[i][j][k]["ALS greedy"]
            allres[i][j][k]["greedy CALS"] = allres[i][j][k]["CALS greedy"]
            allres[i][j][k]["modified greedy CALS"] = allres[i][j][k]["modified CALS greedy"]
numberMonitors = [[[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13,"$\infty$"], [5,7,9,11,13,"$\infty$"]]]
methods = ["upper bound", "modified greedy CALS", "greedy CALS", "greedy ALS", "top traversal", "random"]
allginfo = [[pickle.load(open("./experiment-WMC/Bics/graph.pickle", "rb")), pickle.load(open("./experiment-WMC/BeyondTheNetwork/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/Cogentco/graph.pickle", "rb")), pickle.load(open("./experiment-WMC/Colt/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-WMC/AS8717/graph.pickle", "rb")), pickle.load(open("./experiment-WMC/AS20965/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
#plb.plot(allres, methods, numberMonitors, gname, None, multiple=True)
plb.plotwithUpb(allres, methods, numberMonitors, gname, allginfo, multiple=True)





#experiment-Tmlps
allres = [[pickle.load(open("./experiment-TmLPs/Bics/res.pickle", "rb")), pickle.load(open("./experiment-TmLPs/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-TmLPs/Cogentco/res.pickle", "rb")), pickle.load(open("./experiment-TmLPs/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-TmLPs/AS8717/res.pickle", "rb")), pickle.load(open("./experiment-TmLPs/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["greedy ALS"] = allres[i][j][k]["ALS greedy"]
            allres[i][j][k]["greedy CALS"] = allres[i][j][k]["CALS greedy"]
            allres[i][j][k]["modified greedy CALS"] = allres[i][j][k]["modified CALS greedy"]
numberMonitors = [[[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13,"$\infty$"], [5,7,9,11,13,"$\infty$"]]]
methods = ["TmLP-H", "TmLP-RR", "TmILP", "modified greedy CALS", "CALS greedy", "ALS greedy"]
allginfo = [[pickle.load(open("./experiment-TmLPs/Bics/graph.pickle", "rb")), pickle.load(open("./experiment-TmLPs/BeyondTheNetwork/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-TmLPs/Cogentco/graph.pickle", "rb")), pickle.load(open("./experiment-TmLPs/Colt/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-TmLPs/AS8717/graph.pickle", "rb")), pickle.load(open("./experiment-TmLPs/AS20965/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
plb.plot(allres, methods, numberMonitors,"budget k", gname, allginfo, multiple=True)
"""
"""
#experiment-Pd-pd
allres = [[pickle.load(open("./experiment-Pd-pd/Bics/res.pickle", "rb")), pickle.load(open("./experiment-Pd-pd/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd/Cogentco/res.pickle", "rb")), pickle.load(open("./experiment-Pd-pd/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd/AS8717/res.pickle", "rb")), pickle.load(open("./experiment-Pd-pd/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["greedy ALS"] = allres[i][j][k]["ALS greedy"]
            allres[i][j][k]["greedy CALS"] = allres[i][j][k]["CALS greedy"]
            #allres[i][j][k]["modified greedy CALS"] = allres[i][j][k]["modified CALS greedy"]
pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[200, 240, 280, 320, 360, 400, 435], [200, 240, 280, 320, 360, 400, 435]]]
methods =  ["TmLP-H", "TmLP-RR", "TmILP", "CALS greedy", "ALS greedy"]
allginfo = [[pickle.load(open("./experiment-Pd-pd/Bics/graph.pickle", "rb")), pickle.load(open("./experiment-Pd-pd/BeyondTheNetwork/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd/Cogentco/graph.pickle", "rb")), pickle.load(open("./experiment-Pd-pd/Colt/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd/AS8717/graph.pickle", "rb")), pickle.load(open("./experiment-Pd-pd/AS20965/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
plb.plot(allres, methods, pds, "|Pd|", gname, allginfo, multiple=True)

#new paper |pd|
allres = [[pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/Bics/res.pickle", "rb")), pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/Cogentco/res.pickle", "rb")), pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/AS8717/res.pickle", "rb")), pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["TmLP-R"] = allres[i][j][k]["TmLP-greedy2"]
            allres[i][j][k]["greedy CALS"] = allres[i][j][k]["CALS greedy"]
            allres[i][j][k]["greedy ALS"] = allres[i][j][k]["ALS greedy"]
            #allres[i][j][k]["TmLP-Random Rounding"] = allres[i][j][k]["TmLP-RR"]
            #allres[i][j][k]["upper bound"] = allres[i][j][k]["TmILP"]
#numberMonitors = [[[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13,"$\infty$"], [5,7,9,11,13,"$\infty$"]]]
pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods =["TmILP", "TmLP-R", "TmLP-RR", "greedy CALS", "greedy ALS", "top traversal", "random"]
allginfo = [[pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/Bics/graph.pickle", "rb")), pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/BeyondTheNetwork/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/Cogentco/graph.pickle", "rb")), pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/Colt/graph.pickle", "rb"))], 
           [pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/AS8717/graph.pickle", "rb")), pickle.load(open("./experiment-Pd-pd-LPgreedy2-paper/AS20965/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
plb.plotwithUpb(allres, methods, pds,"|Pd|", gname, allginfo, multiple=True)
#plb.plot(allres, methods, numberMonitors, "|Pd|", gname, allginfo, multiple=True)
"""
"""
#globecom2020-P
allres = [[pickle.load(open("./globecom2020-P/Bics/res.pickle", "rb")), pickle.load(open("./globecom2020-P/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-P/Cogentco/res.pickle", "rb")), pickle.load(open("./globecom2020-P/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-P/AS8717/res.pickle", "rb")), pickle.load(open("./globecom2020-P/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["LP-R"] = allres[i][j][k]["TmLP-greedy2"]
            allres[i][j][k]["ILP"] = allres[i][j][k]["TmILP"]
            allres[i][j][k]["LP-RR"] = allres[i][j][k]["TmLP-RR"]
            allres[i][j][k]["greedy GALS"] = allres[i][j][k]["ALS greedy"]
#numberMonitors = [[[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13,"$\infty$"], [5,7,9,11,13,"$\infty$"]]]
pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods =["ILP", "LP-R", "LP-RR", "greedy GALS", "top traversal", "random"]
allginfo = [[pickle.load(open("./globecom2020-P/Bics/graph.pickle", "rb")), pickle.load(open("./globecom2020-P/BeyondTheNetwork/graph.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-P/Cogentco/graph.pickle", "rb")), pickle.load(open("./globecom2020-P/Colt/graph.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-P/AS8717/graph.pickle", "rb")), pickle.load(open("./globecom2020-P/AS20965/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
plb.plotwithUpb(allres, methods, pds,"|P|", gname, allginfo, multiple=True)
#plb.plot(allres, methods, numberMonitors, "|Pd|", gname, allginfo, multiple=True)



#globecom2020-budget
allres = [[pickle.load(open("./globecom2020-budget/Bics/res.pickle", "rb")), pickle.load(open("./globecom2020-budget/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-budget/Cogentco/res.pickle", "rb")), pickle.load(open("./globecom2020-budget/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-budget/AS8717/res.pickle", "rb")), pickle.load(open("./globecom2020-budget/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["LP-R"] = allres[i][j][k]["TmLP-greedy2"]
            allres[i][j][k]["ILP"] = allres[i][j][k]["TmILP"]
            allres[i][j][k]["LP-RR"] = allres[i][j][k]["TmLP-RR"]
            allres[i][j][k]["greedy GALS"] = allres[i][j][k]["ALS greedy"]
            #allres[i][j][k]["TmLP-Random Rounding"] = allres[i][j][k]["TmLP-RR"]
            #allres[i][j][k]["upper bound"] = allres[i][j][k]["TmILP"]
yaxis = numberMonitors = [[[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13, "$\infty$"], [5,7,9,11,13, "$\infty$"]], [[5,7,9,11,13,"$\infty$"], [5,7,9,11,13,"$\infty$"]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods =["ILP", "LP-R", "LP-RR", "greedy GALS", "top traversal", "random"]
allginfo = [[pickle.load(open("./globecom2020-budget/Bics/graph.pickle", "rb")), pickle.load(open("./globecom2020-budget/BeyondTheNetwork/graph.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-budget/Cogentco/graph.pickle", "rb")), pickle.load(open("./globecom2020-budget/Colt/graph.pickle", "rb"))], 
           [pickle.load(open("./globecom2020-budget/AS8717/graph.pickle", "rb")), pickle.load(open("./globecom2020-budget/AS20965/graph.pickle", "rb"))]]
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
plb.plotwithUpb(allres, methods, yaxis,"budget", gname, allginfo, multiple=True)



#attack2-P
allres = [[pickle.load(open("./attack2/Bics/res.pickle", "rb")), pickle.load(open("./attack2/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./attack2/Cogentco/res.pickle", "rb")), pickle.load(open("./attack2/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./attack2/AS8717/res.pickle", "rb")), pickle.load(open("./attack2/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["LP-R"] = allres[i][j][k]["TmLP-greedy2"]
            allres[i][j][k]["ILP"] = allres[i][j][k]["TmILP"]
            allres[i][j][k]["LP-RR"] = allres[i][j][k]["TmLP-RR"]
            allres[i][j][k]["greedy GALS"] = allres[i][j][k]["ALS greedy"]
            #allres[i][j][k]["TmLP-Random Rounding"] = allres[i][j][k]["TmLP-RR"]
            #allres[i][j][k]["upper bound"] = allres[i][j][k]["TmILP"]
yaxis = numberMonitors = [[[10,20,30,40,50,105], [10,20,30,40,50,105]], [[10,20,30,40,50,105], [10,30,50,70,90,190]], [[10,30,50,70,90,190], [10,30,50,70,90,190]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods =["ILP", "LP-R", "LP-RR", "greedy GALS", "top traversal", "random"]
allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
plb.plotwithUpb(allres, methods, yaxis,"|P|", gname, allginfo, True, "", "ILP")


#attack2-budget
allres = [[pickle.load(open("./attack1/Bics/res.pickle", "rb")), pickle.load(open("./attack1/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./attack1/Cogentco/res.pickle", "rb")), pickle.load(open("./attack1/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./attack1/AS8717/res.pickle", "rb")), pickle.load(open("./attack1/AS20965/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["LP-R"] = allres[i][j][k]["TmLP-greedy2"]
            allres[i][j][k]["ILP"] = allres[i][j][k]["TmILP"]
            allres[i][j][k]["LP-RR"] = allres[i][j][k]["TmLP-RR"]
            allres[i][j][k]["greedy GALS"] = allres[i][j][k]["ALS greedy"]
            #allres[i][j][k]["TmLP-Random Rounding"] = allres[i][j][k]["TmLP-RR"]
            #allres[i][j][k]["upper bound"] = allres[i][j][k]["TmILP"]
yaxis = numberMonitors = [[[1,2,3,4,5,"$\infty$"], [1,2,3,4,5,"$\infty$"]], [[1,2,3,4,5,"$\infty$"], [1,2,3,4,5,"$\infty$"]], [[1,2,3,4,5,"$\infty$"], [1,2,3,4,5,"$\infty$"]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods =["ILP", "LP-R", "LP-RR", "greedy GALS", "top traversal", "random"]
allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 8717", "AS 20965"]]
plb.plotwithUpb(allres, methods, yaxis,"attack budget", gname, allginfo, True, "", "ILP")


#defense-budget
allres = [[pickle.load(open("./TNSE image/defense strategies kd varies/Bics/res.pickle", "rb")), pickle.load(open("./TNSE image/defense strategies kd varies/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./TNSE image/defense strategies kd varies/Cogentco/res.pickle", "rb")), pickle.load(open("./TNSE image/defense strategies kd varies/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./TNSE image/defense strategies kd varies/AS20965/res.pickle", "rb")), pickle.load(open("./TNSE image/defense strategies kd varies/AS8717/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            if "Tmdual_def" in allres[i][j][k]:
                allres[i][j][k]["MIIQP"] = allres[i][j][k]["Tmdual_def"]
            allres[i][j][k]["Greedy Defense"] = allres[i][j][k]["Tmdual_def_greedy"]
            allres[i][j][k]["random"] = allres[i][j][k]["random_def"]
            allres[i][j][k]["max cover"] = allres[i][j][k]["WMC_def"]


yaxis = numberMonitors = [[[0, 10,15,20,40,"$\infty$"], [0, 10,15,20,40,"$\infty$"]], [[0, 5,10,15,20,"$\infty$"], [0,5,10,15,20,"$\infty$"]], [[0,5,10,15,20,"$\infty$"], [0,5,10,15,20,"$\infty$"]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["MIIQP", "Greedy Defense", "max cover", "random"]
allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plotwithUpb(allres, methods, yaxis,"defense budget", gname, allginfo, True, "", "MIIQP")


#varied attack-budget
allres = [[pickle.load(open("./TNSE image/defense strategies ka varies/Bics/res.pickle", "rb")), pickle.load(open("./TNSE image/defense strategies ka varies/BeyondTheNetwork/res.pickle", "rb"))], 
           [pickle.load(open("./TNSE image/defense strategies ka varies/Cogentco/res.pickle", "rb")), pickle.load(open("./TNSE image/defense strategies ka varies/Colt/res.pickle", "rb"))], 
           [pickle.load(open("./TNSE image/defense strategies ka varies/AS20965/res.pickle", "rb")), pickle.load(open("./TNSE image/defense strategies ka varies/AS8717/res.pickle", "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            if "Tmdual_def" in allres[i][j][k]:
                allres[i][j][k]["MIIQP"] = allres[i][j][k]["Tmdual_def"]
            allres[i][j][k]["Greedy Defense"] = allres[i][j][k]["Tmdual_def_greedy"]
            allres[i][j][k]["random"] = allres[i][j][k]["random_def"]
            allres[i][j][k]["max cover"] = allres[i][j][k]["WMC_def"]
            
yaxis = [[[1,2,3,4,5,"$\infty$"], [1,2,3,4,5,"$\infty$"]], [[1,2,3,4,5,"$\infty$"], [1,2,3,4,5,"$\infty$"]], [[1,2,3,4,5,"$\infty$"], [1,2,3,4,5,"$\infty$"]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["MIIQP", "Greedy Defense", "max cover", "random"]
allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plotwithUpb(allres, methods, yaxis,"attack budget", gname, allginfo, True, "", "MIIQP")

#varied attack-budget
address = "./attack_variedka"
allres = [[pickle.load(open("{}/Bics/res.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/res.pickle".format(address), "rb")), pickle.load(open("{}/Colt/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/res.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/res.pickle".format(address), "rb"))]]
allpros = [[pickle.load(open("{}/Bics/pros.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/pros.pickle".format(address), "rb")), pickle.load(open("{}/Colt/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/pros.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/pros.pickle".format(address), "rb"))]]
    
xaxis = [[[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')],[1,3, 5,7,9, float('inf')]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["MILP", "TmILP", "ALS greedy", "CALS greedy", "heuristic greedy", "MILP-RR", "top traversal", "random"]
#allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
#plb.plot(allres, methods, xaxis, "attack budget", gname, allginfo, True, "avg degradation/path (ms)")
plb.plotwithUpb(allres, methods, xaxis,"attack budget", gname, allpros, True, "", "MILP")


#time
address = "./attack_variedka"
allres = [[pickle.load(open("{}/Bics/times.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/times.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/times.pickle".format(address), "rb")), pickle.load(open("{}/Colt/times.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/times.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/times.pickle".format(address), "rb"))]]

xaxis = [[[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')],[1,3, 5,7,9, float('inf')]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["MILP", "TmILP", "ALS greedy", "CALS greedy", "heuristic greedy", "MILP-RR", "top traversal", "random"]
#allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plot(allres, methods, xaxis, "attack budget", gname, True, True, "seconds")

#attack_CALS_overestimated
address = "./attack_CALS_overestimated"
allres = [[pickle.load(open("{}/Bics/res.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/res.pickle".format(address), "rb")), pickle.load(open("{}/Colt/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/res.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/res.pickle".format(address), "rb"))]]
allpros = [[pickle.load(open("{}/Bics/pros.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/pros.pickle".format(address), "rb")), pickle.load(open("{}/Colt/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/pros.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/pros.pickle".format(address), "rb"))]]
    
xaxis = [[[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')],[1,3, 5,7,9, float('inf')]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["CALS greedy", "CALS greedy overestimated"]
#allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plot(allres, methods, xaxis, "attack budget", gname, allpros, True, "avg degradation/path (ms)")


#attack_MILP_overestimated
address = "./attack_MILP_overestimated"
allres = [[pickle.load(open("{}/Bics/res.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/res.pickle".format(address), "rb")), pickle.load(open("{}/Colt/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/res.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/res.pickle".format(address), "rb"))]]
allpros = [[pickle.load(open("{}/Bics/pros.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/pros.pickle".format(address), "rb")), pickle.load(open("{}/Colt/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/pros.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/pros.pickle".format(address), "rb"))]]
    
xaxis = [[[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')],[1,3, 5,7,9, float('inf')]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["MILP", "MILP overestimated"]
#allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plot(allres, methods, xaxis, "attack budget", gname, allpros, True, "avg degradation/path (ms)")


#attack
address = "./attack_new"
allres = [[pickle.load(open("{}/Bics/res.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/res.pickle".format(address), "rb")), pickle.load(open("{}/Colt/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/res.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/res.pickle".format(address), "rb"))]]
allpros = [[pickle.load(open("{}/Bics/pros.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/pros.pickle".format(address), "rb")), pickle.load(open("{}/Colt/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/pros.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/pros.pickle".format(address), "rb"))]]
    
xaxis = [[[4,5,6,7,8],[4,5,6,7,8]], [[10,11,12,13,14,15],[10,11,12,13,14,15]], [ [20,21,22,23,24,25], [20,21,22,23,24,25]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["ALS greedy", "CKR", "top traversal", "random", "all"]
#allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plot(allres, methods, xaxis, "# terminals", gname, allpros, True, "avg degradation/path (ms)")

"""

#attack_overestimated
address = "./attack_MILP_overestimated"
allres = [[pickle.load(open("{}/Bics/res.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/res.pickle".format(address), "rb")), pickle.load(open("{}/Colt/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/res.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/res.pickle".format(address), "rb"))]]
allpros = [[pickle.load(open("{}/Bics/pros.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/pros.pickle".format(address), "rb")), pickle.load(open("{}/Colt/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/pros.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/pros.pickle".format(address), "rb"))]]

address = "./attack_CALS_overestimated"
allres_ = [[pickle.load(open("{}/Bics/res.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/res.pickle".format(address), "rb")), pickle.load(open("{}/Colt/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/res.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/res.pickle".format(address), "rb"))]]
allpros_ = [[pickle.load(open("{}/Bics/pros.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/pros.pickle".format(address), "rb")), pickle.load(open("{}/Colt/pros.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/pros.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/pros.pickle".format(address), "rb"))]]


for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["CALS greedy"] = allres_[i][j][k]["CALS greedy"]
            allres[i][j][k]["CALS greedy overestimated"] = allres_[i][j][k]["CALS greedy overestimated"]
            allpros[i][j][k]["CALS greedy"] = allpros_[i][j][k]["CALS greedy"]
            allpros[i][j][k]["CALS greedy overestimated"] = allpros_[i][j][k]["CALS greedy overestimated"]


xaxis = [[[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')],[1,3, 5,7,9, float('inf')]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
methods = ["MILP", "MILP overestimated", "CALS greedy", "CALS greedy overestimated"]
#allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plot(allres, methods, xaxis, "attack budget", gname, allpros, True, "avg degradation/path (ms)")

"""
#attack_detector
address = "./attack_detector"
allres = [[pickle.load(open("{}/Bics/res.pickle".format(address), "rb")), pickle.load(open("{}/BeyondTheNetwork/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/Cogentco/res.pickle".format(address), "rb")), pickle.load(open("{}/Colt/res.pickle".format(address), "rb"))], 
           [pickle.load(open("{}/AS20965/res.pickle".format(address), "rb")), pickle.load(open("{}/AS8717/res.pickle".format(address), "rb"))]]

for i in range(len(allres)):
    for j in range(len(allres[i])):
        for k in range(len(allres[i][j])):
            allres[i][j][k]["detection rate"] = allres[i][j][k]["detected"]
            allres[i][j][k]["false alarm rate"] = allres[i][j][k]["false alarm"]
            allres[i][j][k]["compromised"] = allres[i][j][k]["Lm"]
            allres[i][j][k]["detected"] = allres[i][j][k]["detected Lm"]
            
            
xaxis = [[[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')], [1,3, 5,7,9, float('inf')]], [[1,3, 5,7,9, float('inf')],[1,3, 5,7,9, float('inf')]]]
#pds = [[[50,60,70,80,90,105], [50,60,70,80,90,105]], [[100, 120, 140, 160, 180, 190], [100, 120, 140, 160, 180, 190]], [[240, 280, 320, 360, 400, 435], [240, 280, 320, 360, 400, 435]]]
axisGroup = ["detection rate", "false alarm rate", "compromised", "detected"]
#allginfo = True
gname = [["Bics", "BTN"], ["Cogent", "Colt"], ["AS 20965","AS 8717"]]
plb.plot(allres, axisGroup, xaxis, "attack budget", gname, True, True, "rate")


#characteristics
address = "./attack_characteristics"
method = "ALS greedy"
allidentifiabilities = []
alldensities = []
alldelays = []
graphs = ["Bics", "BeyondTheNetwork", "Colt", "Cogentco", "AS20965", "AS8717"]# ["Bics", "BeyondTheNetwork"] 
graphlegend = ["Bics", "BeyondTheNetwork", "Colt", "Cogentco", "AS20965", "AS8717"]
for g in graphs:
    identifiabilities = []
    densities = []
    delays = []
    res = pickle.load(open("{}/{}/res.pickle".format(address, g), "rb"))
    pros = pickle.load(open("{}/{}/pros.pickle".format(address, g), "rb"))
    ide = pickle.load(open("{}/{}/identifiability.pickle".format(address, g), "rb"))
    den = pickle.load(open("{}/{}/density.pickle".format(address, g), "rb"))
    lenP = pickle.load(open("{}/{}/lenP.pickle".format(address, g), "rb"))
    for i in range(len(res)):
        for j in range(20):
            print(i,j)
            delay = res[i][method][j]/(pros[i][method][j]*lenP[i][method][j])
            #delay = res[i][method][j]/(pros[i][method][j])
            delays.append(delay)
            alldelays.append(delay)
            identifiabilities.append(ide[i][method][j])
            allidentifiabilities.append(ide[i][method][j])
            densities.append(den[i][method][j])
            alldensities.append(den[i][method][j])
    plt.legend(graphlegend)
    plt.scatter(densities, delays)
from scipy.stats import spearmanr
cov, p = spearmanr(allidentifiabilities, alldelays)
print(cov, p)
#    plt.scatter(densities, delays)

"""
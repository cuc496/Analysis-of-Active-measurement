import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')



def plotwithUpb(allres, xgroups, xlabels, xunit, gname, ginfo ,multiple, ylabel_, uppername):
    if multiple == False:
        allres = [[allres]]
        xlabels = [[xlabels]]
        gname = [[gname]]
    #men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
    #women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)
    
    
    width = 0.8  # the width of the bars
    matplotlib.rcParams.update({'font.size': 15})
    if multiple:
        fig, ax = plt.subplots(nrows=3, ncols=2)
    else:
        fig, ax = plt.subplots()
        ax = [[ax]]
    

    handles = []
    for j in range(len(ax)):
        for k in range(len(ax[j])):
            title = gname[j][k]
            ind = np.arange(len(xlabels[j][k]))  # the x locations for the groups
            means = {}
            stds = {}
            for method in xgroups:
                means[method] = []
                stds[method] = []
            
            for x in range(len(allres[j][k])):
                for method in xgroups:
                    if method not in allres[j][k][x]:
                        continue
                    if ginfo:
                        delayperpath = np.array(allres[j][k][x][method])/np.array(ginfo[j][k][x][method])
                        means[method].append(np.average(delayperpath))
                        stds[method].append(np.std(delayperpath))
                        ylabel = 'avg degradation/path (ms)'
                    else:
                        means[method].append(np.average(allres[j][k][x][method]))
                        stds[method].append(np.std(allres[j][k][x][method]))
                        ylabel = 'avg degradation (ms)'
            xgroups.remove(uppername)
            w = [ind-width/2+width/(2*len(xgroups))]
            for i in range(len(xgroups)-1):
                ww = w[i]+width/len(xgroups)
                w.append(ww)        
            
            if uppername in allres[j][k][0]:
                rects = ax[j][k].errorbar([0,1,2,3,4,5], means[uppername], markersize=40, markeredgewidth=3, label=uppername, linestyle='None', marker='_', color="r")
                handles.append(rects)
                ax[j][k].errorbar([0,1,2,3,4,5], means[uppername], stds[uppername], markersize=0,markeredgewidth=3, label=uppername, linestyle='None', marker='_', color="black")

            patterns = ["*", "//", "O", "\\\\", "...", "//", "o"]
            colors = [ "orange", "green", "blue","red", "purple", "black", "yellow"]
            for i in range(len(xgroups)):
                rects = ax[j][k].bar(w[i], means[xgroups[i]], 
                          width/len(xgroups), yerr=stds[xgroups[i]], 
                          label=xgroups[i], hatch=patterns[i], color=colors[i], edgecolor="black", alpha=0.6)
                handles.append(rects)
                #bars.append(rects)
                #autolabel(ax, rects, "center")
            
            
            #for bar, pattern in zip(bars, patterns):
            #    bar.set_hatch(pattern)
            
            if (ylabel_ != ""):
                ylabel = ylabel_
            #ax[j][k].set_ylabel(ylabel)
            ax[j][k].set_xlabel(xunit)
            ax[j][k].set_ylabel(ylabel)
            #plt.ylim([0, 80])
            ax[j][k].set_title(title)
            ax[j][k].set_xticks(ind)
            ax[j][k].set_xticklabels(xlabels[j][k])
            ax[j][k].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            xgroups.insert(0, uppername)
            fig.legend(handles=handles, labels=xgroups, loc="lower center" ,bbox_transform=fig.transFigure, mode="expand", ncol=8)
    #xgroups.remove("upper bound")
    fig.legend(handles=handles, labels=xgroups, loc="lower center" ,bbox_transform=fig.transFigure, mode="expand", ncol=8)
    fig.tight_layout()
    #ax.yaxis.grid(linestyle='-', linewidth=2)
    
    plt.show()



def plot(allres, xgroups, xlabels, xunit, gname, ginfo ,multiple, ylabel_):
    if multiple == False:
        allres = [[allres]]
        xlabels = [[xlabels]]
        gname = [[gname]]
    #men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
    #women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)
    
    
    width = 0.8  # the width of the bars
    matplotlib.rcParams.update({'font.size': 15})
    if multiple:
        fig, ax = plt.subplots(nrows=3, ncols=2)
    else:
        fig, ax = plt.subplots()
        ax = [[ax]]
    
    
    #rects1 = ax.bar(ind - width/2, men_means, width, yerr=men_std,
    #                label='Men')
    #rects2 = ax.bar(ind + width/2, women_means, width, yerr=women_std,
    #                label='Women')
    
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    
    
    #autolabel(rects1, "left")
    #autolabel(rects2, "right")
    handles = []
    for j in range(len(ax)):
        for k in range(len(ax[j])):
            title = gname[j][k]
            ind = np.arange(len(xlabels[j][k]))  # the x locations for the groups
            means = {}
            stds = {}
            for method in xgroups:
                means[method] = []
                stds[method] = []
            
            for x in range(len(allres[j][k])):
                for method in xgroups:
                    if ginfo:
                        delayperpath = np.array(allres[j][k][x][method])#/np.array(ginfo[j][k][x][method])
                        means[method].append(np.average(delayperpath))
                        stds[method].append(np.std(delayperpath))
                        ylabel = 'avg degradation/path (ms)'
                    else:
                        means[method].append(np.average(allres[j][k][x][method]))
                        stds[method].append(np.std(allres[j][k][x][method]))
                        ylabel = 'avg degradation (ms)'
            w = [ind-width/2+width/(2*len(xgroups))]
            for i in range(len(xgroups)-1):
                ww = w[i]+width/len(xgroups)
                w.append(ww)        
            #w = [ind - width/2, ind + width/2]
            #bars = []
            patterns = ["*", "//", "O", "\\\\", "...", "//", "o", "***"]
            colors = [ "orange", "green", "blue","red", "purple", "black", "yellow", "gray"]
            for i in range(len(xgroups)):
                rects = ax[j][k].bar(w[i], means[xgroups[i]], 
                          width/len(xgroups), yerr=stds[xgroups[i]], 
                          label=xgroups[i], hatch=patterns[i], color=colors[i], edgecolor="black", alpha=0.6)
                handles.append(rects)
                #bars.append(rects)
                #autolabel(ax, rects, "center")
            
            
            #for bar, pattern in zip(bars, patterns):
            #    bar.set_hatch(pattern)
            
            if (ylabel_ != ""):
                ylabel = ylabel_
            #ax[j][k].set_ylabel(ylabel)
            #ax[j][k].set_xlabel('budget k')
            ax[j][k].set_xlabel(xunit)
            ax[j][k].set_ylabel(ylabel)
            #plt.ylim([0, 80])
            ax[j][k].set_title(title)
            ax[j][k].set_xticks(ind)
            ax[j][k].set_xticklabels(xlabels[j][k])
            ax[j][k].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.legend(handles=handles, labels=xgroups, loc="lower center" ,bbox_transform=fig.transFigure, mode="expand", ncol=8)
    fig.tight_layout()
    #ax.yaxis.grid(linestyle='-', linewidth=2)
    
    plt.show()
    
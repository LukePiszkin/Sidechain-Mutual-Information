import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from tqdm import tqdm
from copy import copy

def adaptive_hist(x,y,plot): 
    ## takes in two random variables and returns their joint probability distribution 
    ## using the adaptive histogram algorithm described in Darbellay and Vajda, IEEE 1999
    ## the algorithm is also outlined and simplified by Marek and Tichavsky, JCMF 2009
    
    final_cells = []
    final_rects = []
    current_cells = [[[np.min(x),np.max(x)],[np.min(y),np.max(y)]]] 
    
    if plot:
        fig,ax = plt.subplots()
        ax.scatter(x,y)
        ax.add_patch(Rectangle((np.min(x),np.min(y)),(np.max(x)-np.min(x)),(np.max(y)-np.min(y)),fill=False,edgecolor='r',linewidth=1))
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.2)
        plt.close()
    
    breakout = True
    while breakout:
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(x,y)
            
        remove_list = []
        
        for c in range(len(current_cells)):
        ## check if there are at least 2 data points in the cell
            count = 0
            x_mother = []
            y_mother = []

            for i in range(len(x)):
                if x[i] >= current_cells[c][0][0] and x[i] <= current_cells[c][0][1]:
                    if y[i] >= current_cells[c][1][0] and y[i] <= current_cells[c][1][1]:
                            count += 1
                            x_mother.append(x[i])
                            y_mother.append(y[i])
            
            if count >= 2:
                ##  attempt divide into 4 equiprobable cells
                new_xbins = [current_cells[c][0][0],np.median(x_mother),current_cells[c][0][1]]
                new_ybins = [current_cells[c][1][0],np.median(y_mother),current_cells[c][1][1]]
                
                N_s = [0,0,0,0]
                for i in range(len(x_mother)):
                    if x_mother[i] >= new_xbins[0] and x_mother[i] <= new_xbins[1]:
                        if y_mother[i] >= new_ybins[0] and y_mother[i] <= new_ybins[1]:
                            N_s[0] = N_s[0] + 1

                        else:
                            N_s[1] = N_s[1] + 1
                            
                    elif x_mother[i] >= new_xbins[1] and x_mother[i] <= new_xbins[2]:
                        if y_mother[i] >= new_ybins[0] and y_mother[i] <= new_ybins[1]:
                            N_s[2] = N_s[2] + 1
                            
                        else:
                            N_s[3] = N_s[3] + 1        
                
                if goodness_of_fit_T(np.sum(N_s),N_s) > 7.81:  ## check acceptance by local independance of marginals
                    ## accpect move and make new cells
                    remove_list.append((current_cells[c]))
                    current_cells.append([[new_xbins[0],new_xbins[1]],[new_ybins[0],new_ybins[1]]])
                    current_cells.append([[new_xbins[1],new_xbins[2]],[new_ybins[0],new_ybins[1]]])
                    current_cells.append([[new_xbins[0],new_xbins[1]],[new_ybins[1],new_ybins[2]]])
                    current_cells.append([[new_xbins[1],new_xbins[2]],[new_ybins[1],new_ybins[2]]])
                    if plot:
                        ax.add_patch(Rectangle((new_xbins[0],new_ybins[0]),(new_xbins[1]-new_xbins[0]),(new_ybins[1]-new_ybins[0]),fill=False,edgecolor='r',linewidth=1))
                        ax.add_patch(Rectangle((new_xbins[0],new_ybins[1]),(new_xbins[1]-new_xbins[0]),(new_ybins[2]-new_ybins[1]),fill=False,edgecolor='r',linewidth=1))
                        ax.add_patch(Rectangle((new_xbins[1],new_ybins[0]),(new_xbins[2]-new_xbins[1]),(new_ybins[1]-new_ybins[0]),fill=False,edgecolor='r',linewidth=1))
                        ax.add_patch(Rectangle((new_xbins[1],new_ybins[1]),(new_xbins[2]-new_xbins[1]),(new_ybins[2]-new_ybins[1]),fill=False,edgecolor='r',linewidth=1))
                        
                ## if reject, solidify cell for final computation
                else:
                    final_cells.append(current_cells[c])
                    final_rects.append(Rectangle((current_cells[c][0][0],current_cells[c][1][0]),(current_cells[c][0][1]-current_cells[c][0][0]),(current_cells[c][1][1]-current_cells[c][1][0]),fill=False,edgecolor='g',linewidth=1))
                    remove_list.append((current_cells[c]))
            
            else:
                final_cells.append(current_cells[c])
                final_rects.append(Rectangle((current_cells[c][0][0],current_cells[c][1][0]),(current_cells[c][0][1]-current_cells[c][0][0]),(current_cells[c][1][1]-current_cells[c][1][0]),fill=False,edgecolor='g',linewidth=1))
                remove_list.append((current_cells[c]))
        
        if len(current_cells) == 0:
            breakout = False
                
        else:
            if plot:
                for rect in final_rects:
                    new_rect = copy(rect)
                    ax.add_patch(new_rect)

                plt.tight_layout()
                plt.xlabel('variable x')
                plt.ylabel('variable y')
                plt.show(block=False)
                plt.pause(0.2)
                plt.close()
            
            for cell in remove_list:
                current_cells.remove(cell)        
    
    plt.close()
    
    pdf = []
    x_marg = []
    y_marg = []
    for cell in final_cells:
        count = 0
        x_count = 0
        y_count = 0
        for i in range(len(x)):
            if x[i] >= cell[0][0] and x[i] <= cell[0][1]:
                x_count += 1
                if y[i] >= cell[1][0] and y[i] <= cell[1][1]:
                    count += 1
            
            if y[i] >= cell[1][0] and y[i] <= cell[1][1]:
                y_count += 1
        
        pdf.append(count)
        x_marg.append(x_count)
        y_marg.append(y_count)
    
    return pdf,x_marg,y_marg
    
def goodness_of_fit_T(N,N_s):
    stat_T = (4/N)*np.sum([(N_s[i] - N/4)**2 for i in range(len(N_s))])
    return stat_T

def mutual_info_adapt(pdf,x_marg,y_marg):
    mi = 0
    big_N = np.sum(pdf)

    for n in range(len(pdf)):
        try:
            mi = mi + (pdf[n]/big_N)*np.log((pdf[n]*big_N)/((x_marg[n])*(y_marg[n])))
        except:
            pass

    return mi 
    
def generate_polynomial(degree,input_x):
    """chooses coefficients for a polynomial of the given degree, such that f(a) == b"""

    #to fit only one data point, we can choose arbitrary values for every coefficient except one, which we initially set to zero.
    coefficients = [random.randint(-1000, 1000) for _ in range(degree-1)]
    print(coefficients)
    
    
    output_y = []
    for ele in input_x:
        output_y.append(np.sum([coefficients[n]*ele**n for n in range(len(coefficients))]))
    
    return output_y

if __name__ == "__main__":
    
    ## style stuff for matplotlib
    font_names = [f.name for f in fm.fontManager.ttflist]
    mpl.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    size = int(10e2)
    x = [random.uniform(0,1) for ele in range(size)]
    s = 1000
    noise = [s*random.gauss(0,1) for ele in range(size)]
    y = generate_polynomial(5,x)
    y = [ele+noise[i] for i, ele in enumerate(y)]
    y = [random.uniform(0,1) for ele in range(size)]

    # joint_pdf, xedges, yedges = np.histogram2d(x,y,bins=100)
    # norm = np.sum(joint_pdf)
    print('Correlation coefficient: ',np.corrcoef(x,y)[0][1])
    # joint_pdf = [[joint_pdf[i][j]/norm for j in range(len(joint_pdf[i]))] for i in range(len(joint_pdf))]

    pdf,x_marg,y_marg = adaptive_hist(x,y,True)
    mi = mutual_info_adapt(pdf,x_marg,y_marg)
    print('Mutual Information: ',mi)
    print('Corrected Mutual Information: ',np.sqrt(1-np.exp(-2*mi)))

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from sklearn.metrics import mean_squared_error,r2_score
from Training_pkg.Statistic_Func import regress2, linear_regression


def plot_loss_accuracy_with_epoch(loss, accuracy, outfile):
    COLOR_ACCURACY = "#69b3a2"
    COLOR_LOSS = "#3399e6"
    
    epoch_x = np.array(range(len(accuracy)))
    batchsize = np.around(len(loss)/len(accuracy))

    accuracy_x = epoch_x * batchsize
    loss_x = np.array(range(len(loss))) 
    
    fig = plt.figure(figsize=(24, 8))
    # 修改了ax的位置，将legend挤出图窗外，重现类似题主的问题
    ax1 = fig.add_axes([0.1, 0.2, 0.9, 0.9])
    #fig, ax1 = plt.subplots(figsize=(24, 8))
    ax2 = ax1.twinx()

    ax1.plot(loss_x, loss, color=COLOR_LOSS, lw=1)
    ax2.plot(accuracy_x, accuracy, color=COLOR_ACCURACY, lw=3)

    x_labels = [str(i) for i in epoch_x]
    ax1.set_xlabel("Epoch",fontsize=24)
    ax1.set_xticks(accuracy_x, x_labels, fontsize=20)
    ax1.set_ylabel("Loss", color=COLOR_LOSS, fontsize=24)
    ax1.tick_params(axis="y", labelcolor=COLOR_LOSS)
    ax1.tick_params(axis='y',labelsize=20)

    ax2.set_ylabel("R2", color=COLOR_ACCURACY, fontsize=24)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACCURACY)
    ax2.tick_params(axis='y',labelsize=20)


    fig.suptitle("Loss and R2 vs Epoch", fontsize=32)

    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.close()
    return                                                                                                                                                                                                                                                                                                                


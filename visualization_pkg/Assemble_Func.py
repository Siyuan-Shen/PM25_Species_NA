
from visualization_pkg.iostream import save_loss_accuracy_figure
from visualization_pkg.Training_plot import plot_loss_accuracy_with_epoch



def plot_save_loss_accuracy_figure(loss, accuracy, valid_loss, valid_accuracy, typeName, species, version, nchannel, width, height, special_name):
    training_fig_outfile, valid_fig_outfile = save_loss_accuracy_figure(typeName=typeName,species=species,version=version,nchannel=nchannel,width=width,height=height,special_name=special_name)
    plot_loss_accuracy_with_epoch(loss=loss,accuracy=accuracy,outfile=training_fig_outfile)
    plot_loss_accuracy_with_epoch(loss=valid_loss, accuracy=valid_accuracy, outfile=valid_fig_outfile)
    return
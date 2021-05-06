from matplotlib import pyplot as plt

def complete_plot(dimensionality, title, ylabel, label='number of pixels'):
    plt.axvline(dimensionality, linestyle='--', color='green', alpha=0.4, label=label)
    plt.legend(frameon=False)
    plt.xlabel('Number of training examples N')
    plt.ylabel(ylabel)
    plt.title(title)
    
def train_test_loss_plot(train_test_losses, labels, title, dimension=784, ylogscale=False, ylabel='Log loss'):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    plt.suptitle(title)
    
    for train_test_loss, label in zip(train_test_losses, labels):
        axes[0].plot(train_test_loss[0], train_test_loss[1], '.-', label=label)
        axes[1].plot(train_test_loss[0], train_test_loss[2], '.-', label=label)
    plt.sca(axes[0])   
    complete_plot(dimension, 'Train', ylabel)
    
    plt.sca(axes[1])
    complete_plot(dimension, 'Test', '')
    
    if ylogscale:
        axes[0].set_yscale('log')        
        axes[1].set_yscale('log')
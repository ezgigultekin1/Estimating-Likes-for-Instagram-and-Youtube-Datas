import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(train_sizes, train_scores, test_scores, title):
    plt.figure()
    plt.plot(train_sizes, train_scores, label='Training error')
    plt.plot(train_sizes, test_scores, label='Cross-validation error')
    plt.title(title)
    plt.xlabel('Training size')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

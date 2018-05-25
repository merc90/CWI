from utils.dataset         import Dataset
from utils.cwi             import CWI
from utils.baseline        import Baseline
from utils.scorer          import report_score
import numpy
import matplotlib.pyplot as plt

def execute(language):
    data = Dataset(language)
    instance = CWI(language)
    # baseline = Baseline(language)
    print("{}: {} training - {} dev - {} test".format(language, len(data.trainset), len(data.devset), len(data.testset)))

    instance.train(data.trainset)
    predictions = instance.test(data.testset)
    # baseline.train(data.trainset)
    # predBaseline = baseline.test(data.testset)
    gold_labels = [sent['gold_label'] for sent in data.testset]

    accuracy = numpy.cumsum([ prediction == sent for sent, prediction in zip(gold_labels, predictions) ]) / range(1,len(data.testset)+1)
    # accuracy = numpy.cumsum([ prediction == sent for sent, prediction in zip(gold_labels, predBaseline) ]) / range(1,len(data.testset)+1)

    print("For", language, "language:")
    report_score(gold_labels, predictions)
    # report_score(gold_labels, predBaseline)
    
    plt.figure("Learning graphs")     #Creates the plot and set the title to Learning Graphs
    graph = plt.subplot2grid((1,1),(0,0))                  #Create subplot in 0 coordinate
    title = "Learning Rate for Complex Words Identification of "+language #Define title
    graph.set_title(title)      #Set the title of plot 
    graph.plot(100.*accuracy[10:], 'g-', label="Accuracy")       #Plot line for accuracy
    graph.set_yscale('linear')                  #set y scale linear
    graph.set_ylabel('Accuracy')             #Set y label Accuracy
    graph.set_xscale('linear')                  #Set x scale linear
    graph.set_xlabel('Iterations')              #Set x label Iterations
    legend = plt.legend(loc='upper right')      #Declare the position of legends
    for label in legend.get_texts():        
        label.set_fontsize('small')             #setting font size of label to small
    for label in legend.get_lines():
        label.set_linewidth(1)                  #Setting line width of legend to 1
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    execute('spanish')
    execute('english')
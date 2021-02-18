"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 3: Logistic Regression

   Miikka Silfverberg
"""

import os
import numpy as np
import nltk

from data import read_semeval_datasets, evaluate, write_semeval, get_class

from random import seed, shuffle
seed(0)

# defining the paths for input and output data
assignment_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(assignment_path,'data')
results_dir = os.path.join(assignment_path,'results')

# Bias token added to every example. This is equivalent to having a
# separate bias weight.
BIAS="BIAS"

# Set this to 1 to classify the test set after you have tuned all
# hyper-parameters. Don't classify the test set before you have
# finished tuning all hyper-parameters.
CLASSIFY_TEST_SET=0

def extract_features(data):
    """
        This is a bag-of-words feature extractor for document
        classification. The features of a document are simply its
        tokens. A BIAS token is added to every example.

        This function modifies @a data. For every ex in data, it adds
        a binary np.array ex["FEATURES"].

        No need to edit this function.
    """
    # Make encoder available globally (this is needed for analyzing
    # feature weights).
    global encoder

    all_tokens = sorted(list({wf for ex in data["training"] 
                              for wf in ex["BODY"]+[BIAS]}))
    encoder = {wf:i for i,wf in enumerate(all_tokens)}
    
    for data_set in data.values():
        for i,ex in enumerate(data_set):
            feature_ids = list({encoder[tok] for tok in ex["BODY"]+[BIAS] 
                                if tok in encoder})
            ex["FEATURES"] = np.zeros(len(all_tokens))
            ex["FEATURES"][feature_ids] = 1

def custom_extract_features(data):
    """ 
        If you have found good improvements with your perceptron feature
        extraction strategy, you can copy your code here. Changing this
        function is not required however.
    """
    for data_set in data.values():
        for ex in data_set:
            ex['BODY'] = nltk.word_tokenize(ex['BODY'])

    extract_features(data)

def softmax(scores):
    """
        This function implements softmax for the real numbers in the
        np.array scores. Provide a proper implementation for the
        function.
    """
    exp_scores = np.exp(scores)

    return exp_scores/sum(exp_scores)

class LogisticRegression:
    """ 
        This is a simple logistic regression model.
    
        It is initialized with a training set. You can train the
        classifier for n epochs by calling train(). You can classify
        a data set using classify().

        The fields of this classifier are
        
        W      - The collection of model parameters.
                 W[klass] is the parameter vector for klass.
        Y      - A list of sentiment classes.
        N      - The number of updates that have been performed 
                 during training.
        Lambda - The learning rate.
    """
    def __init__(self,training_data):
        self.Y = list({ex["SENTIMENT"] for ex in training_data})        
        # Initialize all parameters to 0.
        self.W = {klass:np.zeros(training_data[0]["FEATURES"].shape)
                  for klass in self.Y}
        self.N = 0
        # This is the lambda learning rate. You can tune it
        # using the development set.
        self.Lambda = 0.1

    def classify_ex(self,ex):
        """
            This function classifies a single example. 

            It returns a dictionary where the keys are classes
            (positive, neutral, negative) and the values are
            probabilities, for example 
                  p("positive"|ex["FEATURES"];self.W).

            Implement your own classification. You will need the
            weights in self.W and the feature vector
            ex["FEATURES"]. You should also make use of the function
            softmax().
        """ 
        score = np.array([ex["FEATURES"]@self.W[klass] for klass in self.Y])
        soft_score = softmax(score)
        return dict(zip(self.Y, soft_score))

    def classify(self,data):
        """
            This function classifies a data set. 

            No need to change this function.
        """
        return [get_class(self.classify_ex(ex)) for ex in data]

    def estimate_ex(self,ex):
        """
            This function trains on a single example.

            You should edit it to implement parameter estimation
            properly. You will need to call self.classify_ex() and you
            need to use the feature vector ex["FEATURES"].

            You will also need the learning rate self.Lambda and the
            parameters self.W.
        """
        gold_class = ex["SENTIMENT"]
        sys_class_distribution = self.classify_ex(ex) 

        for klass in self.Y:
            if klass == gold_class:
                self.W[klass] = self.W[klass] + self.Lambda * (1-sys_class_distribution[klass])*ex["FEATURES"]
            else:
                self.W[klass] = self.W[klass] - self.Lambda * sys_class_distribution[klass] * ex["FEATURES"] 


    def train(self,train_data,dev_data,epochs):
        """
            This function trains the model. 

            No need to change this function.
        """
        for n in range(epochs):
            shuffle(train_data)
            for ex in train_data:
                self.N += 1
                self.estimate_ex(ex)
            sys_classes = self.classify(dev_data)
            acc, _ = evaluate(sys_classes, dev_data)
            print("Epoch %u: Dev accuracy %.2f%%" % (n+1, acc))
            
if __name__=="__main__":
    # Read training, development and test sets and open the output for
    # test data.
    print("Reading data (this may take a while).")
    data = read_semeval_datasets(data_dir)

    output_file = open(os.path.join(results_dir,"test.output.txt"),
                       "w", encoding="utf-8")

    print("Extracting features.")
    custom_extract_features(data)

    # Use the development set to tune the number of training epochs.
    epochs = 13

    print("Training model.")
    model = LogisticRegression(data["training"])
    model.train(data["training"], data["development.gold"], epochs)

    if CLASSIFY_TEST_SET:
        print("Labeling test set.")
        test_output = model.classify(data["test.input"])
        acc, fscores = evaluate(test_output, data["test.gold"])
        print("Final test accuracy: %.2f%%" % acc)
        print("Per class F1-score:")
        for c in fscores:
            print(" %s %.2f" % (c,fscores[c]))
        write_semeval(data["test.input"], test_output, output_file)
    print()


    """
        Write your code for analyzing model weights here.

        You can use the Python dict encoder which will be defined.
    """
    print('Weight for feature "happy" in the positive class is: %.3f'
          % model.W["positive"][encoder["happy"]])

import random
import math
from collections import Counter
from DecisionTree import DecisionTree

class RandomForest(object):
    def __init__(self, tree_num):
        self.tree_num = tree_num
        self.forest = []

    def bootstrap(self, records):
        """
        This function bootstrap will return a set of records, which has the same
        size with the original records but with replacement.
        """
        # You code here
        bootstrap_records = []
        size = len(records)
        
        # Build bootstrap records from size of records, with random selection of records
        for i in range(0, size):
            bootstrap_records.append(records[random.randrange(0, (size - 1), 1)])
        
        return bootstrap_records
        


    def train(self, records, attributes):
        """
        This function will train the random forest, the basic idea of training a
        Random Forest is as follows:
        1. Draw n bootstrap samples using bootstrap() function
        2. For each of the bootstrap samples, grow a tree, with the following
            modification: at each node, randomly sample m of the predictors and
            choose the best split from among those variables
        """
        # Your code here
        # Create the trees, and add them to the forrest
        for i in range(0, self.tree_num):
            tree = DecisionTree()
            tree.train(self.bootstrap(records), attributes)
            self.forest.append(tree)
        
        pass
        
    def predict(self, sample):
        """
        The predict function predicts the label for new data by aggregating the
        predictions of each tree

        This function should return the predicted label
        """
        # Your code here
        predictions = {}

        # Create a dictionary of the class label predictions
        for tree in self.forest:
            r = tree.predict(sample)
            if r not in predictions:
                predictions[r] = 0
                predictions[r] += 1
    
        # Find the key with the maximum value, and return it as the prediction
        return max(predictions, key=predictions.get)
        
    pass
        


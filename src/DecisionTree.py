from collections import Counter
import unittest
import math


class TreeNode(object):
    def __init__(self, isLeaf=False):
        # Your code here
        self.attribute = -1
        self.value = None
        self.results = None
        self.lb = None
        self.rb = None
        self.isLeaf = isLeaf
        self.classification = None

    def predict(self, sample):
        """
        This function predicts the label of given sample
        """
        # Your code here
        branch = self
        
        while branch.isLeaf == False:
            if sample['attributes'][branch.attribute] == branch.value:
                # True Branch
                branch = branch.lb
            else:
                # False Branch
                branch = branch.rb
        
        # Once you reach the leaf, return the classification label
        return branch.classification



class DecisionTree(object):
    """
        Class of the Decision Tree
    """
    def __init__(self):
        self.root = None

    def stopping_cond(self, records, attributes):
        """
        The stopping_cond() function is used to terminate the tree-growing
        process by testing whether all the records have either the same class
        label or the same attribute values.

        This function should return True/False to indicate whether the stopping
        criterion is met.
        """
        # Your code here
        labels = {}
        
        # Check if all the class labels are the same
        for record in records:
            r = record["label"]
            if r not in labels:
                labels[r] = 0
            labels[r] += 1
        
        # If one of the class labels is equal to the record count, stop
        for key in labels.keys():
            if labels[key] == len(records):
                return True

    def classify(self, records):
        """
        This function determines the class label to be assigned to a leaf node.
        For each node t, let p(i|t) denote the fraction of training records from
        class i associated with the node t. In most cases, the leaf node is
        assigned to the class that has the majority number of training records

        This function should return a label that is assigned to the node
        """

        # Count the labels and return the majority label
        return max(records, key=records.get)

    def entropy(self, records):
        """
        This function calculates the entropy of given set of records
        """
        results = self.class_label_count(records)
        e = 0.0
        for r in results.keys():
            p = float(results[r]) / len(records)
            e = e - p * math.log(p, 2)
        return e

    def split_set(self, records, attribute, key):
        """
        This function splits the set into two subsets based on a given key value
        from an attribute. It will only perform a binary split, not multi-way.
        """
        left_subset = []
        right_subset = []

        # If the key value exists in the record, add it to the left subset
        # otherwise, place it in the right subset
        for record in records:
            if record['attributes'][attribute] == key:
                left_subset.append(record)
            else:
                right_subset.append(record)

        return left_subset, right_subset    

    def find_best_split(self, records, attributes):
        """
        The find_best_split() function determines which attribute should be
        selected as the test condition for splitting the training records.
        The test condition should be measured by Gain Ratio.

        This function should return multiple information:
        attributed selected for splitting,
        threshold value for splitting,
        best_gain_ratio,
        left subset,
        right subset
        """

        # Your code here
        # Hint-1: loop through all available attributes
        # Hint-2: for each attribute, loop through all possible values
        # Hint-3: calculate gain ratio and pick the best attribute

            # Split the records into two parts based on the value of the select
            # attribute

                # calculate the information gain based on the new split

                # calculate the gain ratio

                # if the gain_ratio is better the best split we have tested
                # set this split as the best split

        max_gain = 0.0
        split_attr = None
        split_sets = None
    
        cur_score = self.entropy(records)
    
        # Find all of each attribute's values from sub_attributes
        for attr in range(0, len(attributes)):
            attr_values = {}
            
            # Go through each record and mark possible values
            for record in records:
                attr_values[record['attributes'][attributes[attr]]] = 1
    
        # Test splits on all of these values
        for value in attr_values.keys():
            left_subset, right_subset = self.split_set(records, attributes[attr], value)
            
            # Find Information Gain
            p = float(len(left_subset)) / len(records)
            gain = cur_score - p * self.entropy(left_subset) - (1 - p) * self.entropy(right_subset)
                
            # If there is an information gain and neither set is completely empty,
            # update the values.
            if gain > max_gain and len(left_subset) > 0 and len(right_subset) > 0:
                max_gain = gain
                split_attr = (attributes[attr], value)
                split_sets = (left_subset, right_subset)
        
        return split_sets, split_attr, max_gain
            
    def train(self, records, attributes):
        """
            This function trains the model with training records "records" and
            attribute set "attributes", the format of the data is as follows:
                records: training records, each record contains following fields:
                    label - the lable of this record
                    attributes - a list of attribute values
                attributes: a list of attribute indices that you can use for
                            building the tree
            Typical data will look like:
                records: [
                            {
                                "label":"p",
                                "attributes":['p','x','y',...]
                            },
                            {
                                "label":"e",
                                "attributes":['b','y','y',...]
                            },
                            ...]
                attributes: [0, 2, 5, 7,...]
        """
        # Randomly choose the attribute subset
        rand_attributes = []
        for attr in attributes:
            if random.random() > .5:
                rand_attributes.append(attr)
        
        # Create the tree with the training data
        self.tree_growth(records, rand_attributes)

    def tree_growth(self, records, attributes):
        """
        This function grows the Decision Tree recursively until the stopping
        criterion is met. Please see textbook p164 for more details

        This function should return a TreeNode
        """
        # Your code here
        # Hint-1: Test whether the stopping criterion has been met by calling function stopping_cond()
        # Hint-2: If the stopping criterion is met, you may need to create a leaf node
        # Hint-3: If the stopping criterion is not met, you may need to create a
        #         TreeNode, then split the records into two parts and build a
        #         child node for each part of the subset

        split_sets, split_attr, max_gain = self.find_best_split(records, attributes)
        # If gain is greater than 0 and stopping condition is not met, we want to
        # split again, other wise it is a leaf node
        if max_gain > 0 and not self.stopping_cond(records, attributes):
            left_branch = self.tree_growth(split_sets[0], attributes)
            right_branch = self.tree_growth(split_sets[1], attributes)
            
            # Create a Tree Node, and set the root of the decision tree to it. (The
            # final self.root will be the tree root)
            self.root = TreeNode()
            self.root.attribute = split_attr[0]
            self.root.value = split_attr[1]
            self.root.lb = left_branch
            self.root.rb = right_branch
            return self.root
        else:
            class_label_counts = self.class_label_count(records)
            node = TreeNode(isLeaf=True)
            node.results = class_label_counts
            node.classification = self.classify(class_label_counts)
            return node

    def predict(self, sample):
        """
        This function predict the label for new sample by calling the predict
        function of the root node
        """
        return self.root.predict(sample)



if __name__ == "__main__":
    unittest.main()


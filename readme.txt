Random Forest in Python

by Raveena Mewani

The code performs Random Forest Aggregation to predict class labels using bootstrap sampling in the production of Decision Trees. 

Description of the Code

Random Forest Class

Creating a Random Forest

The Random Forest Class first produces the bootstrap samples by gathering the same number of samples as the size of the provided data set:

    bootstrap_records = []
    size = len(records)


    # Build bootstrap records from size of records, with random selection of records
    for i in range(0, size):
        bootstrap_records.append(records[random.randrange(0, (size - 1), 1)])
Using this bootstrap sample set, the Random Forest class then creates a DecisionTree() with it and adds the tree to the forest:

     tree = DecisionTree()
   tree.train(self.bootstrap(records), attributes)
   self.forest.append(tree)
Steps 1 & 2 are repeated for the number of requested trees (self.tree_num) in the forest.

Prediction with a Random Forest

The predict(self, sample) function is called with a sample record to predict on. A predictions dictionary is created to store the predicted label counts. The forest is then looped through, calling the DecisionTree.predict(self, sample) function on each tree in the forest, updating the prediction label key in the dictionary. Finally, the maximum value key is returned as the majority vote for the predicted label of the sample.

    predictions = {}
    # Create a dictionary of the class label predictions
    for tree in self.forest:
        r = tree.predict(sample)
        if r not in predictions:
            predictions[r] = 0
        predictions[r] += 1
  
# Find the key with the maximum value, and return it as the prediction
    return max(predictions, key=predictions.get)

    
Decision Tree Class

Creating a Decision Tree

The train(self) function is the entry point to the creation of a decision tree. It handles the selection of random attributes to build the tree on and the calling of tree_growth() function.

    # Randomly choose the attribute subset
    rand_attributes = []
    for attr in attributes:
        if random.random() > .5:
            rand_attributes.append(attr)


    # Create the tree with the training data
    self.tree_growth(records, rand_attributes)
The tree_growth(self, records, attributes) function recursively builds the decision tree by splitting on select attributes by way of entropy.

split_sets, split_attr, max_gain = self.find_best_split(records, attributes)
find_best_split(self, records, attributes) loops through all of the attributes and their values, testing the splits on each by means of the split_set(self, records, attribute, key) function. The Information Gain is calculated using entropy. The attribute and respective value with the largest Information Gain is chosen to split on. The split sets, attribute information and information gain is returned.

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
            gain = cur_score - p * self.entropy(left_subset) - (1 - p) *
self.entropy(right_subset)

            # If there is an information gain and neither set is completely empty,
            # update the values.
            if gain > max_gain and len(left_subset) > 0 and len(right_subset) > 0:
                max_gain = gain
                split_attr = (attributes[attr], value)
                split_sets = (left_subset, right_subset)


    return split_sets, split_attr, max_gain
```
Back in tree_growth(self, records, attributes), if the Information Gain is larger than 0 and stopping_cond(self, records, attributes) returns false, then tree_growth(self, records, attributes) will be called again on the left and right subsets, and a non-leaf TreeNode() will be created, initializing the class variables respectively.

    # If gain is greater than 0 and stopping condition is not met, we want to
    # split again, other wise it is a leaf node
    if max_gain > 0 and not self.stopping_cond(records, attributes):
        true_branch = self.tree_growth(split_sets[0], attributes)
        false_branch = self.tree_growth(split_sets[1], attributes)


        # Create a Tree Node, and set the root of the decision tree to it. (The
        # final self.root will be the tree root)
        self.root = TreeNode()
        self.root.attribute = split_attr[0]
        self.root.value = split_attr[1]
        self.root.lb = true_branch
        self.root.rb = false_branch
        return self.root
If the gain is < 0 or the stopping condition is met, then a leaf TreeNode() will be created, initializing the class variables respectively.

      	else:
            class_label_counts = self.class_label_count(records)
            node = TreeNode(isLeaf=True)
            node.results = class_label_counts
            node.classification = self.classify(class_label_counts)
            return node
Once the recursion completes, the root of the tree will be returned and stored in the self.root of the DecisionTree().

Prediction with a Decision Tree

The DecisionTree.predict(self, sample) function is called, which calls the predict(self, sample) of the TreeNode() root.

return self.root.predict(sample)
The TreeNode.predict(self, sample) function iterates through the branches of the DecisionTree based on the split attribute of the node it is on and the sample’s attribute value. Once a leaf node of the tree is reached, the branch’s classification label is returned as the predicted class label.

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
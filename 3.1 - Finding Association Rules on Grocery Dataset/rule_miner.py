import itertools

class RuleMiner(object):

    def __init__(self, support_t, confidence_t):
        """Class constructor for RuleMiner
        Arguments:
            support_t {int} -- support threshold for the dataset
            confidence_t {int} -- confidence threshold for the dataset
        """
        self.support_t = support_t
        self.confidence_t = confidence_t

    def get_support(self, data, itemset):
        """Returns the support for an itemset. The support of an itemset
        refers to the number of baskets wherein the itemset is present.
        Arguments:
            data {pd.DataFrame} -- DataFrame containing the dataset represented
            as a matrix
            itemset {list} -- list of items to check in each observation
            in the dataset
        Returns:
            int -- support for itemset in data
        """
        # TODO: Implement this function based on the documentation.
        # Hint: Use the pandas.DataFrame.all() and the pandas.DataFrame.sum()
        # function.
        # Get baskets (rows) that are true
        count = data[itemset].all(axis=1).sum()
        return count
        

    def merge_itemsets(self, itemsets):
        """Returns a list of merged itemsets. If one itemset of size 2
        from itemsets contains one item in another itemset of size 2 from
        itemsets, the function merges these itemsets to produce an itemset of
        size 3.
        Arguments:
            itemsets {list} -- list which contains itemsets to merge.
        Returns:
            list -- list of merged itemsets

        Example:
            If itemsets is equal to [[1, 2], [1, 3], [1, 5], [2, 6]], then the
            function should return [[1, 2, 3], [1, 2, 5], [1, 2, 6], [1, 3, 5]]
        """

        new_itemsets = []

        cur_num_items = len(itemsets[0])

        if cur_num_items == 1:
            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    new_itemsets.append(list(set(itemsets[i]) | set(itemsets[j])))

        else:
            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    combined_list = list(set(itemsets[i]) | set(itemsets[j]))
                    if len(combined_list) == cur_num_items + 1 and combined_list not in new_itemsets:
                        new_itemsets.append(combined_list)

        return new_itemsets

    def get_rules(self, itemset):
        """Returns a list of rules produced from an itemset.
        Arguments:
            itemset {list} -- list which contains items.
        Returns:
            list -- list of rules produced from an itemset.

        Example:
            If itemset is equal to [[1, 2, 3], then the function should
            return [[[1], [2, 3]], [[2, 3], [1]], [[2], [1, 3]], [[1, 3], [2]],
            [[3], [1, 2]], [[1, 2], [3]]]
            The list represents the following rules, respectively:
            {1} -> {2, 3}
            {2, 3} -> {1}
            {2} -> {1, 3}
            {1, 3} -> {2}
            {3} -> {1, 2}
            {1, 2} -> {3}
        """

        combinations = itertools.combinations(itemset, len(itemset) - 1)
        combinations = [list(combination) for combination in combinations]

        rules = []
        for combination in combinations:
            diff = set(itemset) - set(combination)
            rules.append([combination, list(diff)])
            rules.append([list(diff), combination])

        return rules

    def get_frequent_itemsets(self, data):
        """Returns a list frequent itemsets in the dataset. The support of each
        frequent itemset should be greater than or equal to the support
        threshold.
        Arguments:
            data {pd.DataFrame} -- DataFrame containing the dataset represented
            as a matrix
        Returns:
            list -- list of frequent itemsets in the dataset.
        """

        # TODO: Complete this function.

        itemsets = [[i] for i in data.columns]
        old_itemsets = []
        flag = True

        while flag:
            new_itemsets = []
            for itemset in itemsets:
                # TODO: Get the support for each itemset and add the itemset to
                # the list new_itemsets if the support for the itemset is
                # greater than or equal to the support threshold support_t
                # Hint: Use the get_support() function that we have defined in
                # this class.
                if self.get_support(data,itemset) >= self.support_t:
                    new_itemsets.append(itemset)

            if len(new_itemsets) != 0:
                old_itemsets = new_itemsets
                itemsets = self.merge_itemsets(new_itemsets)
            else:
                flag = False
                itemsets = old_itemsets

        return itemsets

    def get_confidence(self, data, rule):
        """Returns the confidence for a rule. Suppose the rule is X -> y, then
        the confidence for the rule is the support of the concatenated list of
        X and y divided by the support of X.
        Arguments:
            data {pd.DataFrame} -- DataFrame containing the dataset represented
            as a matrix
            rule {list} -- list which contains two values. If the rule is
            X -> y, then a rule is a list containing [X, y].
        Returns:
            float -- confidence for rule in data
        """

        # TODO: Implement this function based on the documentation.
        # Hint: Use the get_support() function that we have defined in this
        # class.
        cf = self.get_support(data,rule[0] + rule[1]) / self.get_support(data,rule[0])
        return cf

    def get_association_rules(self, data):
        """Returns a list of association rules with support greater than or
        equal to the support threshold support_t and confidence greater than or
        equal to the confidence threshold confidence_t.
        Arguments:
            data {pd.DataFrame} -- DataFrame containing the dataset represented
            as a matrix
        Returns:
            list -- list of association rules. If the rule is X -> y, then each
            rule is a list containing [X, y].
        """
        # TODO: Complete this function.

        # TODO: Call the get_frequent_itemsets() function that we have defined
        # in this class, and assign the result to the variable itemsets.
        itemsets = self.get_frequent_itemsets(data)
        
          # TODO: Get the rules for each frequent itemset and add to the
        # list rules
        # Hint: Use the get_rules() function that we have defined in this
        # class.
        rules = []
        for itemset in itemsets:
            rules = rules + self.get_rules(itemset)
           
        association_rules = []
        # TODO: Get the confidence for each rule and add the rule to
        # the list association_rules if the confidence for the rule is
        # greater than or equal to the confidence threshold confidence_t
        # Hint: Use the get_confidence() function that we have defined in
        # this class.
        for rule in rules:
            if self.get_confidence(data, rule) >= self.confidence_t:
                association_rules.append(rule)
          
        return association_rules

        return association_rules

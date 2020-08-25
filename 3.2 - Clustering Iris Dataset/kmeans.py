import numpy as np
import pandas as pd

class KMeans(object):

    def __init__(self, k, start_var, end_var, num_observations, data):
        """Class constructor for KMeans
        Arguments:
            k {int} -- number of clusters to create from the data
            start_var {int} -- starting index of the variables (columns) to
            consider in creating clusters from the dataset. This is
            useful for excluding some columns for clustering.
            end_var {int} -- ending index of the variables (columns) to
            consider in creating clusters from the dataset. This is
            useful for excluding some columns for clustering.
            num_observations {int} -- total number of observations (rows) in
            the dataset.
            data {DataFrame} -- the dataset to cluster
        """
        np.random.seed(1)
        self.k = k
        self.start_var = start_var
        self.end_var = end_var
        self.num_observations = num_observations
        self.columns = [i for i in data.columns[start_var:end_var]]
        self.centroids = pd.DataFrame(columns=self.columns)


    def initialize_centroids(self, data):
        """Returns initial centroids. This function picks a random point from
        the dataset as the first centroid, then iteratively picks points that
        are farthest from the current set of centroids.

        The algorithm for this initialization is as follows:
        1. Randomly select the first centroid from the data points in the
        dataset.
        2. For each data point, compute its distance from each centroid in the
        current set of centroids. For each distance computed from each
        centroid, retain only the shortest distance for each data point. In
        other words, we are computing the distance of each data point from
        the nearest centroid.
        3. Select the data point with the maximum distance from the nearest
        centroid as the next centroid.
        4. Repeat steps 2 and 3 until we have k number of centroids.

        Arguments:
            data {DataFrame} -- dataset to cluster
        Returns:
            DataFrame -- contains the values of the initial location of the
            centroids.
        """

        # TODO: Complete this function.

        # Step 1: Randomly select a data point from the dataset as the first
        # centroid.
        index = np.random.randint(low=0, high=self.num_observations)
        point = data.iloc[index, self.start_var:self.end_var]
        self.centroids = self.centroids.append(point, ignore_index=True)
        sliced_data = data.iloc[:, self.start_var:self.end_var]

        # Step 2: Select the remaining centroids.
        for i in range(1, self.k):
             
            # The variable distance is a DataFrame that will store the
            # distances of each data point from each centroid. Each column
            # represents the distance of the data point from a specific
            # centroid. Example, the value in row 3 column 0 of the DataFrame
            # distances represents the distance of data point 3 from
            # centroid 0.
            distances = pd.DataFrame()

            # TODO: Get the Euclidean distance of each data point in the
            # dataset from each centroid in the current set of centroids.
            # Then store it to a column in the DataFrame distances
            # Hint: Use the get_euclidean_distance() function that we have
            # defined in this class.
            for j in range(len(self.centroids)):
                # Get distance of all data pts from the centroid
                dist = self.get_euclidean_distance(sliced_data,self.centroids.loc[j])
                # Store it in a column in distances
                distances = pd.concat([distances, pd.DataFrame(dist)], axis=1)
                # Step 3: Select the data point with the maximum distance from the
                # nearest centroid as the next centroid.
                # TODO: Get the minimum distance of each data point from centroid.
                # Then, get the index of the data point with the maximum distance
                # from the nearest centroid and store it to variable index.
                # Hint: Use pandas.DataFrame.min() and pandas.Series.idxmax()
                # functions.
                
                # Get the minimum distance of each data point and make it the next index 
                index = distances.min(axis=1).idxmax(axis=1)
                
            # Append the selected data point to the set of centroids.
            point = data.iloc[index, self.start_var:self.end_var]
            self.centroids = self.centroids.append(point, ignore_index=True)
            
        return self.centroids

    def get_euclidean_distance(self, point1, point2):
        """Returns the Euclidean distance between two data points. These
        data points can be represented as 2 Series objects. This function can
        also compute the Euclidean distance between a list of data points
        (represented as a DataFrame) and a single data point (represented as
        a Series), using broadcasting.

        The Euclidean distance can be computed by getting the square root of
        the sum of the squared difference between each variable of each data
        point.

        For the arguments point1 and point2, you can only pass these
        combinations of data types:
        - Series and Series -- returns np.float64
        - DataFrame and Series -- returns pd.Series

        For a DataFrame and a Series, if the shape of the DataFrame is
        (3, 2), the shape of the Series should be (2,) to enable broadcasting.
        This operation will result to a Series of shape (3,)

        Arguments:
            point1 {Series or DataFrame} - data point
            point2 {Series or DataFrame} - data point
        Returns:
            np.float64 or pd.Series -- contains the Euclidean distance
            between the data points.
        """

        # TODO: Implement this function based on the documentation.
        # Hint: Use the pandas.Series.sum() and the numpy.sqrt() functions.
       
        if isinstance(point1, pd.DataFrame):
            dist = []
            for idx,point in point1.iterrows():
                dist.append(np.sqrt(np.sum([(point-point2)**2])))
                
            return pd.Series(dist)
        else:
            dist = np.sqrt(np.sum([(point1-point2)**2])) 
            return dist

    def group_observations(self, data):
        """Returns the clusters of each data point in the dataset given
        the current set of centroids. Suppose this function is given 100 data
        points to cluster into 3 groups, the function returns a Series of
        shape (100,), where each value is between 0 to 2.

        Arguments:
            data {DataFrame} -- dataset to cluster
        Returns:
            Series -- represents the cluster of each data point in the dataset.
        """
        # TODO: Complete this function.

        # The variable distance is a DataFrame that will store the distances
        # of each data point from each centroid. Each column represents the
        # distance of the data point from a specific centroid. Example, the
        # value in row 3 column 0 of the DataFrame distances represents the
        # distance of data point 3 from centroid 0.
        distances = pd.DataFrame()
        sliced_data = data.iloc[:, self.start_var:self.end_var]
        for i in range(self.k):
            # TODO: Get the Euclidean distance of the data from each centroid
            # then store it to a column in the DataFrame distances
            # Hint: Use the get_euclidean_distance() function that we have
            # defined in this class.
            # Get distance of all data pts from the centroid
            dist = self.get_euclidean_distance(sliced_data,self.centroids.loc[i])
            
            # Store it in a column in distances
            distances = pd.concat([distances, pd.DataFrame({i:dist})], axis=1)
        
        # TODO: get the index of the lowest distance for each data point and
        # assign it to a Series named groups
        # Hint: Use pandas.DataFrame.idxmin() function.
        groups = distances.idxmin(axis=1)
        return groups.astype('int32')

    def adjust_centroids(self, data, groups):
        """Returns the new values for each centroid. This function adjusts
        the location of centroids based on the average of the values of the
        data points in their corresponding clusters.

        Arguments:
            data {DataFrame} -- dataset to cluster
            groups {Series} -- represents the cluster of each data point in the
            dataset.
        Returns:
            DataFrame -- contains the values of the adjusted location of the
            centroids.
        """

        # TODO: Complete this function.

        grouped_data = pd.concat([data, groups.rename('group')], axis=1)

        # TODO: Group the data points together using the group column, then
        # get their mean and store to variable centroids.
        # Hint: use pandas.DataFrame.groupby and
        # pandas.core.groupby.GroupBy.mean functions.
        centroids = grouped_data.groupby(['group']).mean()

        return centroids

    def train(self, data, iters):
        """Returns a Series which represents the final clusters of each data
        point in the dataset. This function stops clustering if one of the
        following is met:
        - The values of the centroids do not change.
        - The clusters of each data point do not change.
        - The maximum number of iterations is met.

        Arguments:
            data {DataFrame} -- dataset to cluster
            iters {int} -- maximum number of iterations before the clustering
            stops
        Returns:
            Series -- represents the final clusters of each data point in the
            dataset.
        """

        # TODO: Complete this function.

        cur_groups = pd.Series(-1, index=[i for i in range(self.num_observations)])
        i = 0
        flag_groups = False
        flag_centroids = False
       
        # While no stopping criterion has been met, do the following
        while i < iters and not flag_groups and not flag_centroids:

            # TODO: Get the clusters of the data points in the dataset and
            # store it in variable groups.
            # Hint: Use the group_observation() function that we have defined
            # in this class.
            groups = self.group_observations(data)
           
            # TODO: Adjust the centroids based on the current clusters and
            # store it in variable centroids.
            # Hint: Use the adjust_centroids() function that we have defined
            # in this class.
            centroids = self.adjust_centroids(data,groups).iloc[:, self.start_var:self.end_var]
            # TODO: Check if there are changes with the clustering of the
            # data points
            if groups.equals(cur_groups):
                flag_groups = True
            # TODO: Check if there are changes with the values of the centroids
            if centroids.equals(self.centroids):
                flag_centroids = True
            
            cur_groups = groups
            self.centroids = centroids

            i+=1
            print('Iteration', i)

        print('Done clustering!')
        return cur_groups

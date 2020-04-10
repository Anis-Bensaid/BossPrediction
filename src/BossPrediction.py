import pandas as pd
import pickle
import numpy as np
import time
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


class Model:
    def __init__(self):
        self.data = None
        self.groupby_col = None
        self.link_rate_table = None
        self.proxy_calculator = None

        # Private attributes and constants
        self.__req_columns = ['Company ID', 'Mega Industry', 'Exec ID', 'Long title', 'Short title', 'Level', 'Boss ID']
        self.__stop_words = [stop_word for stop_word in stopwords.words('english') if
                             stop_word not in ['it', 'ma', 'she', 'has', 'do', 'be']]

    def fit(self, data=None, data_path=None, groupby_col='Mega Industry', proxies_size=25000, verbose=True):
        """
        Fits the model to the training set i.e. format the input and calculate the like.rate.
        :param data: DataFrame: training set (provide either this or data_path)
        :param data_path: str: path for the training set (provide either this or data)
        :param groupby_col: str: (default='Mega Industry) name of the column by which we aggregate the link.rates
        :param proxies_size: int: number of potential pairs to use as proxies
        :param verbose: bool: (default=True) print the steps
        :return: None
        """
        if verbose:
            print('Fitting the model...')
        start = time.time()
        self.groupby_col = groupby_col.lower().replace(' ', '.')
        self.data = self.__read_data(data, data_path)
        self.data = self.__check_data_validity(self.data, verbose=verbose)
        self.data = self.__format_data(self.data, verbose=verbose)
        self.data = self.__generate_link_rate(self.data, verbose=True)
        self.link_rate_table = self.data[[self.groupby_col, 'short.title.stem.n2', 'short.title.stem.n1',
                                          'link.rate']].drop_duplicates()
        print('Time elapsed: ', time.time() - start)
        if verbose:
            print('Fitting the proxy calculator...')
        start = time.time()
        self.proxy_calculator = ProxyCalculator(self.groupby_col)
        self.proxy_calculator.fit(self.data, proxies_size)
        if verbose:
            print('Time elapsed: ', time.time() - start)

    def predict(self, data_path=None, data=None, nb_suggestions=1, use_proxies=True, alpha=0.7, nb_proxies=5,
                verbose=True):
        """
        Given a DataFrame with a list of N-1 and N-2, predicts the boss IDs of N-2.
        :param data: DataFrame: training set (provide either this or data_path)
        :param data_path: str: path for the training set (provide either this or data)
        :param nb_suggestions: int: number of outputted predictions per executive id
        :param use_proxies: Bool: whether or not to use proxies
        :param alpha: the weight given to the short title similarity
        :param nb_proxies: number of proxies to average on for the link.rate.proxy
        :param verbose: bool: (default=True) print the steps
        :return: DataFrame: the same DataFrame data, with two extra columns: Boss ID and Model Score
        """
        start = time.time()
        input_data = self.__read_data(data, data_path)
        prediction = self.__check_data_validity(input_data, verbose=verbose)
        prediction = self.__format_data(prediction, verbose=verbose)
        # Append the link.rates from the training set
        prediction = prediction.merge(self.link_rate_table, how='left')
        # Use proxies
        if use_proxies:
            prediction_to_proxies = prediction[prediction['link.rate'].isna()]
            if not prediction_to_proxies.empty:
                link_rate_table_proxies = self.proxy_calculator.calculate_proxies(prediction_to_proxies, alpha,
                                                                                  nb_proxies)
                prediction = prediction.merge(link_rate_table_proxies, how='left', on=['exec.id.n1', 'exec.id.n2'])
                prediction['Proxy'] = ~prediction['link.rate'].isna()
                prediction['link.rate'].fillna(prediction['link.rate.proxy'], inplace=True)
            else:
                prediction['Proxy'] = False
        else:
            prediction['Proxy'] = False
        # Select the top nb_suggestion suggestions
        prediction = prediction.groupby('exec.id.n2').apply(
            lambda x: x.nlargest(nb_suggestions, 'link.rate')).reset_index(drop=True)
        # Finally format the output.
        prediction = input_data.merge(prediction[['exec.id.n2', 'exec.id.n1', 'link.rate', 'Proxy']], how='left',
                                      right_on='exec.id.n2', left_on='Exec ID')
        prediction['Boss ID'].fillna(prediction['exec.id.n1'], inplace=True)
        prediction['Model Score'] = prediction['link.rate']
        prediction.drop(['exec.id.n2', 'exec.id.n1', 'link.rate'], axis=1, inplace=True)
        if verbose:
            print('Time elapsed: ', time.time() - start)
        return prediction

    @staticmethod
    def __read_data(data=None, data_path=None, verbose=True):
        """
        Reads the data using either the path for the data-set or the actual DataFrame. (It's a little extra...)
        :param data: DataFrame: the DataFrame to read
        :param data_path: str: path to the DataFrame (.csv file)
        :param verbose: bool: allows prints
        :return: DataFrame
        """
        if (data is None) and (data_path is None):
            raise TypeError("fit() missing 1 required argument: 'data' or 'data_path'")
        if data is not None:
            return data
        else:
            if verbose:
                print('Reading data...')
            return pd.read_csv(data_path, encoding='utf8')

    def __check_data_validity(self, data, verbose=True):
        """
        Checks that we have the right columns, drops duplicates and NAs etc...
        :param data: DataFrame:
        :param verbose: bool:
        :return: DataFrame
        """
        if verbose:
            print("Checking data validity...")
        # In case we apply it to the test set ...
        if 'Boss ID' not in data.columns:
            data['Boss ID'] = None
        # Check the column names
        for col in self.__req_columns:
            if col not in data.columns:
                raise KeyError("column " + col + " missing from training data.")
        # Keep the important columns
        data = data.loc[:, self.__req_columns].copy()
        data.columns = [col.lower().replace(' ', '.') for col in data.columns]
        # Drop rows with missing values (except in boss.id)
        data.dropna(subset=data.columns.difference(['boss.id']), inplace=True)
        # Drop duplicated executive IDs
        data.drop_duplicates(['exec.id'], inplace=True)
        # Filter out bad formatting
        data = data.loc[(data['company.id'].astype(str) != '0') & (data['exec.id'].astype(str) != '0'), :]
        return data

    @staticmethod
    def __clean_stem_titles(data, input_colname, stop_words, verbose=True):
        """
        Cleans and stems the titles in a the column input_colname of the data DataFrame.
        :param data: DataFrame: containing the titles to clean and stem
        :param input_colname: str: name of the column containing the titles to clean and stem
        :param stop_words: list: list of stopwords to delete
        :param verbose: bool: allows printing
        :return: DataFrame: data with two extra columns, one for cleaned titles, and one for the stemmed titles.
        """
        if verbose:
            print('Cleaning and stemming ' + input_colname.replace('.', ' ') + 's...')
        # Clean the titles
        data[input_colname + '.clean'] = data[input_colname].str.lower().str.replace('[^0-9a-z ]+', '').str.replace(
            ' +', ' ')
        porter_stemmer = PorterStemmer()
        # Stem the titles
        # TODO : optimize this part.
        data[input_colname + '.stem'] = data[input_colname + '.clean'].str.split().apply(
            lambda x: [porter_stemmer.stem(word) for word in x if word not in stop_words]).apply(
            lambda x: ' '.join(x))
        data.drop([input_colname, input_colname + '.clean'], axis=1, inplace=True)
        return data

    def __create_pairs(self, data, verbose=True):
        """
        Creates all the possible combination of (N-2, N-1) per company. Each row will have all the columns relative to
        the N-2 executive and one of the possible N-1 executives.
        :param data: training DataFrame
        :param verbose: bool: allows printing
        :return: DataFrame: training DataFrame reformatted
        """
        if verbose:
            print('Creating all possible (N-1, N-2) pairs by company...')
        data = data.loc[data['level'] == 'N-2', :].merge(
            data.loc[data['level'] == 'N-1', data.columns.difference([self.groupby_col, 'boss.id'])], how='left',
            on='company.id', suffixes=['.n2', '.n1'])
        data.drop(['level.n2', 'level.n1'], axis=1, inplace=True)
        data.dropna(subset=data.columns.difference(['boss.id']), inplace=True)
        data.drop_duplicates(inplace=True)
        return data

    def __generate_link_rate(self, data, verbose=True):
        """
        Generates the link rates.
        :param data: DataFrame: training DataFrame
        :param verbose: bool: allows printing
        :return: None
        """
        if verbose:
            print('Generating link rates...')
        data['link'] = (data['exec.id.n1'] == data['boss.id']).astype(int)
        groupby_cols = ['short.title.stem.n1', 'short.title.stem.n2']

        # groupby_col: name of the column on which we are aggregating the link rate
        if self.groupby_col != '':
            groupby_cols = [self.groupby_col] + ['short.title.stem.n1', 'short.title.stem.n2']
        group = data.groupby(groupby_cols)
        count_rate = group.size()
        link = group['link'].sum()
        link_rate = link / count_rate
        link_table = count_rate.to_frame('match.count').merge(
            link_rate.to_frame('link.rate'), left_index=True, right_index=True)
        data = data.merge(link_table, left_on=groupby_cols, right_index=True)
        return data[
            [self.groupby_col, 'long.title.stem.n2', 'short.title.stem.n2', 'long.title.stem.n1', 'short.title.stem.n1',
             'match.count', 'link.rate']]

    def __format_data(self, data, verbose=True):
        """
        Formats the data by cleaning the titles, and creating all possible (N-1,N-2) combinations
        :param data: DataFrame: DataFrame to format
        :param verbose: bool: allows printing
        :return:
        """
        data = self.__clean_stem_titles(data, 'long.title', verbose=verbose, stop_words=self.__stop_words)
        data = self.__clean_stem_titles(data, 'short.title', verbose=verbose, stop_words=self.__stop_words)
        data = self.__create_pairs(data, verbose=True)
        return data

    def save(self, filename, verbose=True):
        """
        Saves the model in a pickle file.
        :param filename: str: path_to_file
        :param verbose: bool: allows printing
        :return: None
        """
        if verbose:
            print('Saving model...')
        if not filename.endswith('.pkl'):
            raise TypeError("The file name should end with .pkl")
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        if verbose:
            print('Done.')


class ProxyCalculator:
    def __init__(self, groupby_col='mega.industry'):
        """
        :param groupby_col: str: (default='mega.industry') name of the column by which we aggregate the link.rates
        """
        self.groupby_col = groupby_col.lower().replace(' ', '.')
        self.data = None
        self.proxies = None

    def fit(self, data, proxies_size):
        """
        Fit the calculator to the data by selecting the subset of proxy pairs that occur the most in the training set.
        :param data: DataFrame : training set
        :param proxies_size: int : (Best value ~25000) size of the subset of proxy pairs that occur the most in the
        training set
        """
        self.proxies = data.groupby([self.groupby_col]).apply(
            lambda x: x.nlargest(proxies_size, 'match.count')).reset_index(drop=True)

    def calculate_proxies(self, data, alpha=0.7, nb_proxies=5):
        """

        :param data: DataFrame : contains the inputs, i.e. that data point that need predictions
        :param alpha: float : (between 0 and 1) the weight of short titles compared to long titles for the proxy
        calculation
        :param nb_proxies: number of proxies amongst which we take the weighted average of link.rate
        :return: DataFrame : input data plus the column link.rate.proxy
        """
        data = data.merge(self.proxies, how='left', on=[self.groupby_col], suffixes=['', '.proxy'])
        data = self.__avg_proxy_link(data, alpha, nb_proxies)
        return data

    @staticmethod
    def __cosine_sim(data, col1, col2, colname):
        """
        Caluculate the cosine similarity between columns col1 and col2 of data
        :param data: DataFrame :
        :param col1: first column to compute the cosine similarity
        :param col2: second column to compute the cosine similarity
        :param colname: name of the outputted column (where the result is saved)
        :return: DataFrame : input data plus the column containing the cosine similarity
        """
        data[col1 + '.split'] = data[col1].str.split()
        data[col2 + '.split'] = data[col2].str.split()
        data[colname + '.overlap'] = [len(set(a).intersection(b)) for a, b in
                                      zip(data[col1 + '.split'], data[col2 + '.split'])]
        data[colname + '.sim'] = data[colname + '.overlap'] / (
                data[col1 + '.split'].str.len() * data[col2 + '.split'].str.len())
        data.drop([col1 + '.split', col2 + '.split', colname + '.overlap'], axis=1, inplace=True)
        return data

    def __title_sim(self, data, level, alpha):
        """
        Calculates the title similarity defined as a weighted average (alpha) of the long.title.sim and short.title.sim
        :param data: DataFrame:
        :param level: (n1 or n2)
        :param alpha: weight given to the short.title.sim as opposed to long.title.sim
        :return: DataFrame: input data plus the column containing the title similarity
        """
        data = self.__cosine_sim(data, 'long.title.stem.' + level, 'long.title.stem.' + level + '.proxy',
                                 'long.titles.' + level)
        data = self.__cosine_sim(data, 'short.title.stem.' + level, 'short.title.stem.' + level + '.proxy',
                                 'short.titles.' + level)
        data['titles.sim.' + level] = ((1 - alpha) * data['long.titles.' + level + '.sim'] + alpha * data[
            'short.titles.' + level + '.sim'])
        return data

    def __pair_sim(self, data, alpha):
        """
        Calculates pair similarity which is the product of n1 title similarity and n2 title similarity
        :param data: DataFrame:
        :param alpha: weight given to the short.title.sim as opposed to long.title.sim
        :return:
        """
        data = self.__title_sim(data, 'n1', alpha)
        data = self.__title_sim(data, 'n2', alpha)
        data['pair.sim'] = data['titles.sim.n1'] * data['titles.sim.n2']
        return data

    def __avg_proxy_link(self, data, alpha, nb_proxies):
        """
        Calculates the weighted average of link.rates amongst the most similar proxies
        :param data: DataFrame: 
        :param alpha: weight given to the short.title.sim as opposed to long.title.sim
        :param nb_proxies: number of proxies to average on
        :return: DataFrame:
        """
        data = self.__pair_sim(data, alpha)
        data = data.groupby(['exec.id.n2', 'exec.id.n1']).apply(
            lambda x: x.nlargest(nb_proxies, 'pair.sim')).reset_index(drop=True)
        data = data.groupby(['exec.id.n2', 'exec.id.n1']).apply(
            lambda x: np.average(x['link.rate.proxy'], weights=x['pair.sim'])).to_frame('link.rate.proxy')
        return data.reset_index()

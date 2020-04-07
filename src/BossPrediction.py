import pandas as pd
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


class Model:
    def __init__(self):
        self.data = None
        self.groupby_col = None

        # Private attributes and constants
        self.__req_columns = ['Company ID', 'Mega Industry', 'Exec ID', 'Long title', 'Short title', 'Level', 'Boss ID']
        self.__stop_words = [stop_word for stop_word in stopwords.words('english') if
                             stop_word not in ['it', 'ma', 'she', 'has', 'do', 'be']]

    def fit(self, data=None, data_path=None, groupby_col='Mega Industry', verbose=True):
        self.groupby_col = groupby_col.lower().replace(' ', '.')
        self.data = self.__read_data(data, data_path)
        self.data = self.__check_data_validity(self.data, verbose=verbose)
        self.data = self.__format_data(self.data, verbose=verbose)
        self.data = self.__generate_link_rate(self.data, groupby_col=self.groupby_col, verbose=True)

    def predict(self, data=None, data_path=None, nb_suggestions=1, verbose=True):
        output_columns = ['exec.id.n1']
        input_data = self.__read_data(data, data_path)
        prediction = self.__check_data_validity(input_data, verbose=verbose)
        prediction = self.__format_data(prediction, verbose=verbose)
        prediction = prediction.merge(self.data[[self.groupby_col, 'short.title.stem.n2', 'short.title.stem.n1',
                                                 'link.rate.' + self.groupby_col]].drop_duplicates(), how='left')
        prediction = prediction.groupby('exec.id.n2').apply(
            lambda x: x.nlargest(nb_suggestions, 'link.rate.' + self.groupby_col)).reset_index(drop=True)
        prediction['boss.id'] = prediction['exec.id.n1']
        prediction['level'] = "N-2"
        prediction = prediction[
            ['company.id', self.groupby_col, 'exec.id.n2', 'long.title.n2', 'short.title.n2', 'level', 'boss.id',
             'link.rate.' + self.groupby_col]]

        input_data['Model Score'] = np.nan
        prediction.columns = input_data.columns
        return pd.concat([input_data[input_data['Level'] != 'N-2'], prediction])

    def __read_data(self, data=None, data_path=None, verbose=True):
        if (data is None) and (data_path is None):
            raise TypeError("fit() missing 1 required argument: 'data' or 'data_path'")
        if data is not None:
            return data
        else:
            if verbose:
                print('Reading data...')
            return pd.read_csv(data_path, encoding='utf8')

    def __check_data_validity(self, data, verbose=True):
        if verbose:
            print("Checking data validity...")
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

    def __clean_stem_titles(self, data, input_colname, stop_words, verbose=True):
        if verbose:
            print('Cleaning and stemming ' + input_colname.replace('.', ' ') + 's...')
        porter_stemmer = PorterStemmer()
        data[input_colname + '.clean'] = data[input_colname].str.lower().str.replace('[^0-9a-z ]+', '').str.replace(
            ' +', ' ')
        data[input_colname + '.stem'] = data[input_colname + '.clean'].str.split().apply(
            lambda x: [porter_stemmer.stem(word) for word in x if word not in self.__stop_words]).apply(
            lambda x: ' '.join(x))
        return data

    def __create_pairs(self, data, verbose=True):
        if verbose:
            print('Creating all possible (N-1, N-2) pairs by company...')
        data = data.loc[data['level'] == 'N-2', :].merge(
            data.loc[data['level'] == 'N-1', data.columns.difference([self.groupby_col, 'boss.id'])], how='left',
            on='company.id', suffixes=['.n2', '.n1'])
        data.drop(['level.n2', 'level.n1'], axis=1, inplace=True)
        data.dropna(subset=data.columns.difference(['boss.id']), inplace=True)
        data.drop_duplicates(inplace=True)
        data['link'] = (data['exec.id.n1'] == data['boss.id']).astype(int)
        return data

    def __generate_link_rate(self, data, groupby_col, verbose=True):
        groupby_cols = ['short.title.stem.n1', 'short.title.stem.n2']
        if groupby_col != '':
            groupby_cols = [groupby_col] + ['short.title.stem.n1', 'short.title.stem.n2']
            groupby_col = '.' + groupby_col
        group = data.groupby(groupby_cols)
        count_rate = group.size()
        link = group['link'].sum()
        link_rate = link / count_rate
        link_table = count_rate.to_frame('match.count' + groupby_col).merge(
            link_rate.to_frame('link.rate' + groupby_col), left_index=True, right_index=True)
        return data.merge(link_table, left_on=groupby_cols, right_index=True)

    def __format_data(self, data, verbose=True):
        data = self.__clean_stem_titles(data, 'long.title', verbose=verbose, stop_words=self.__stop_words)
        data = self.__clean_stem_titles(data, 'short.title', verbose=verbose, stop_words=self.__stop_words)
        data = self.__create_pairs(data, verbose=True)
        return data

    def save(self, filename):
        if not filename.endswith('.pkl'):
            raise TypeError("The file name should end with .pkl")
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
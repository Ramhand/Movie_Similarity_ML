import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import json


class DataBaseConstructor:
    def __init__(self):
        self.movies = pd.read_csv(
            'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv')
        self.credits = pd.read_csv(
            'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv')
        self.db = sqlite3.connect('db.db')
        self.movies.to_sql('movies', self.db, if_exists='replace')
        self.credits.to_sql('credits', self.db, if_exists='replace')
        self.query = """
            SELECT *
            FROM movies
            FULL OUTER JOIN credits ON movies.title = credits.title
        """
        self.combo_data = pd.read_sql_query(self.query, self.db)
        self.combo_data = self.combo_data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


class EDA_Transformer:
    def __init__(self):
        self.db = DataBaseConstructor()
        self.json_columns = ['genres', 'keywords', 'cast', 'crew']
        self.json_deconstructor(self.json_columns)
        self.column_cleanup()

    def json_deconstructor(self, lisp):
        for i in lisp:
            mega_results = []
            for j in self.db.combo_data[i]:
                results = []
                data = json.loads(j)
                if not i == 'crew':
                    for k in data:
                        results.append(k['name'])
                else:
                    for k in data:
                        if k['job'] == 'Director':
                            results.append(k['name'])
                if i == 'cast' and len(results) > 3:
                    results = results[:3]
                st = ''
                for _ in results:
                    _ = _.strip().replace(' ', '')
                    st += _
                    st += ','
                st = st[:-1]
                mega_results.append(st)
            self.db.combo_data[i] = mega_results

    def column_cleanup(self):
        for i in self.json_columns:
            self.db.combo_data[i] = self.db.combo_data[i].apply(lambda x: x.split(',') if x is not None else x)
        self.db.combo_data['tags'] = self.db.combo_data[
            ['overview', 'genres', 'keywords', 'cast', 'crew']].values.tolist()
        self.db.combo_data['tags'] = self.db.combo_data['tags'].apply(self.tag_cleanup)
        self.db.combo_data.drop(columns=['genres', 'keywords', 'cast', 'crew', 'overview'], inplace=True)
        self.data = self.db.combo_data
        columns = self.data.columns.to_list()
        columns[columns.index('title')] = 'title2'
        self.data.columns = columns
        self.data.drop(columns='title2', inplace=True)
        self.tobevecced = self.data['tags']
        vector = CountVectorizer()
        self.vecced = vector.fit_transform(self.tobevecced).toarray()

    def tag_cleanup(self, x):
        st = ''
        for i in x:
            if i is not None:
                if not type(i) == str:
                    for j in i:
                        st += j
                        st += ' '
                else:
                    st += i + ' '
        st = st[:-1].lower()
        return st


class KNearestNestedDoll:
    def __init__(self):
        self.eda = EDA_Transformer()
        self.vex = self.eda.vecced
        self.data = self.eda.data
        self.similarity = cosine_similarity(self.vex)
        self.loop()

    def loop(self):
        self.recommend(input('Enter a movie, get recommendations'))

    def recommend(self, movie):
        m_ind = self.data.index[self.data['title'] == movie]
        relatives = self.similarity[m_ind][0]
        results = sorted(list(enumerate(relatives)), reverse=True, key=lambda x: x[1])[1:6]
        for i in results:
            print(self.data.loc[i[0], 'title'])
        self.loop()


if __name__ == '__main__':
    KNearestNestedDoll()

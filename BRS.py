import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


class Books():
    
    def __init__(self):
        self.books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, warn_bad_lines=False,encoding="latin-1")
        self.users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, warn_bad_lines=False,encoding="latin-1")
        self.ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, warn_bad_lines=False,encoding="latin-1")
        
        # Splitting Explicit and Implicit user ratings
        self.ratings_explicit = self.ratings[self.ratings['Book-Rating'] != 0]
        self.ratings_implicit = self.ratings[self.ratings['Book-Rating'] == 0]
        
        # Each Books Mean ratings and Total Rating Count
        self.average_rating = pd.DataFrame(self.ratings_explicit.groupby('ISBN')['Book-Rating'].mean())
        self.average_rating['ratingCount'] = pd.DataFrame(self.ratings_explicit.groupby('ISBN')['Book-Rating'].count())
        self.average_rating = self.average_rating.rename(columns = {'Book-Rating':'MeanRating'})
        
        # To get a stronger similarities
        counts1 = self.ratings_explicit['User-ID'].value_counts()
        self.ratings_explicit = self.ratings_explicit[self.ratings_explicit['User-ID'].isin(counts1[counts1 >= 50].index)]
        
        # Explicit Books and ISBN
        self.explicit_ISBN = self.ratings_explicit.ISBN.unique()
        self.explicit_books = self.books.loc[self.books['ISBN'].isin(self.explicit_ISBN)]
        
        # Look up dict for Book and BookID
        self.Book_lookup = dict(zip(self.explicit_books["ISBN"], self.explicit_books["Book-Title"]))
        self.ID_lookup = dict(zip(self.explicit_books["Book-Title"],self.explicit_books["ISBN"]))


        
    def Top_Books(self, n=10, RatingCount = 100, MeanRating = 3):
        
        BOOKS = self.books.merge(self.average_rating, how = 'right', on = 'ISBN')
        
        M_Rating = BOOKS.loc[BOOKS.ratingCount >= RatingCount].sort_values('MeanRating', ascending = False).head(n)

        H_Rating = BOOKS.loc[BOOKS.MeanRating >= MeanRating].sort_values('ratingCount', ascending = False).head(n)

            
        return M_Rating, H_Rating

    
class SVD(Books):
    
    def __init__(self, n_latent_factor = 50):
        super().__init__()
        self.n_latent_factor = n_latent_factor
        self.ratings_mat = self.ratings_explicit.pivot(index="User-ID", columns="ISBN", values="Book-Rating").fillna(0)
        
        self.uti_mat = self.ratings_mat.values
        # normalize by each users mean
        self.user_ratings_mean = np.mean(self.uti_mat, axis = 1)
        self.mat = self.uti_mat - self.user_ratings_mean.reshape(-1, 1)
        
        self.explicit_users = np.sort(self.ratings_explicit['User-ID'].unique())
        self.User_lookup = dict(zip(range(1,len(self.explicit_users)),self.explicit_users))
        
        self.predictions = None

    def scipy_SVD(self):
        
        # singular value decomposition
        U, S, Vt = svds(self.mat, k = self.n_latent_factor)
        
        S_diag_matrix=np.diag(S)
        
        # Reconstructing Original Prediction Matrix
        X_pred = np.dot(np.dot(U, S_diag_matrix), Vt) + self.user_ratings_mean.reshape(-1, 1)
        
        self.predictions = pd.DataFrame(X_pred, columns = self.ratings_mat.columns, index = self.ratings_mat.index)
    
        return

    def Recommend_Books(self, userID, num_recommendations = 10):
        
        # Get and sort the user's predictions
        user_row_number = self.User_lookup[userID] # User ID starts at 1, not 0

        sorted_user_predictions = self.predictions.loc[user_row_number].sort_values(ascending=False) 
        
        # Get the user's data and merge in the books information.
        user_data = self.ratings_explicit[self.ratings_explicit['User-ID'] == (self.User_lookup[userID])]
        user_full = (user_data.merge(self.books, how = 'left', left_on = 'ISBN', right_on = 'ISBN').
                         sort_values(['Book-Rating'], ascending=False)
                     )
    
        # Recommend the highest predicted rating books that the user hasn't seen yet.
        recom = (self.books[~self.books['ISBN'].isin(user_full['ISBN'])].
                            merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                                  left_on = 'ISBN',
                                  right_on = 'ISBN'))
        recom = recom.rename(columns = {user_row_number: 'Predictions'})
        recommend = recom.sort_values(by=['Predictions'], ascending = False)
        recommendations = recommend.iloc[:num_recommendations, :-1]
        
        return user_full, recommendations
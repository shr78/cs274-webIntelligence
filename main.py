import BRS
import argparse
import sys
import pandas as pd
import time
import numpy as np

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='SVD collaborative filtering')

    parser.add_argument(
        "--user",
        action="store_true",
        help="User based collaborative filtering"
    )
    parser.add_argument(
        "--item",
        action="store_true",
        help="Item collaborative filtering"
    )
    
    return parser.parse_args()

def YN():
    reply = str(input('\n\nContinue (y/n):\t')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        sys.exit();
    else:
        return False


def main():
    
    args = parse_arguments()
    
    cont = True
    
    if not args.user and not args.item:
        print("\n\nChoose user or item for testing user/item based collaborative filtering\n")
        
        Top_B = BRS.Books()
        
        High_Mean_Rating, High_Rating_Count = Top_B.Top_Books()
        
        pd.set_option('display.max_colwidth', -1)
        
        print("\n\nBooks with high ratings :\n")
        print(High_Mean_Rating[['Book-Title','MeanRating','ratingCount','Book-Author']])
        
        print("\n\nBooks with high rating count :\n")
        print(High_Rating_Count[['Book-Title','MeanRating','ratingCount','Book-Author']])
        
        sys.exit()

    if args.user:
        start_time = time.time()
        UCF = BRS.SVD()
        
        UCF.scipy_SVD()
        
        while cont:
            
            try:
                User_ID = int(input('Enter User ID in the range {0}-{1}: '.format(1,len(UCF.explicit_users))))
            except:
                print('Enter a number')
                sys.exit()
            
            if User_ID in range(1,len(UCF.explicit_users)):
                pass
            else:
                print("Choose between {0}-{1}".format(1,len(UCF.explicit_users)))
                sys.exit()
            
            
            Rated_Books , SVD_Recommended_Books = UCF.Recommend_Books(userID=User_ID)
             
            pd.set_option('display.max_colwidth', -1)
            
            print("\nThe Books already  rated by the user\n")
            print(Rated_Books[['Book-Title','Book-Rating']].to_string(index=False))
            
            print("\nRecommended Books for the user\n")
            SVD_Recommended_Books = SVD_Recommended_Books.merge(UCF.average_rating, how='left', on='ISBN')
            SVD_Recommended_Books = SVD_Recommended_Books.rename(columns = {'Book-Rating':'MeanRating'})
            print(SVD_Recommended_Books[['Book-Title','MeanRating']].to_string(index=False))
            print("--- {} minutes ---"  .format((time.time() - start_time)/60))
            cont = YN()
            
    if args.item:
        
        while cont:
            
            book_name = input('\n\nEnter the Book Title:\t')
            item(book_name)
            cont = YN()


def item(book_name):
    start_time = time.time()
    
    books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, warn_bad_lines=False,encoding="latin-1")
    books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-Url-S', 'Image-Url-M', 'Image-Url-L']
    users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1",warn_bad_lines=False)
    users.columns = ['User-ID', 'Location', 'Age']
    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, warn_bad_lines=False, encoding="latin-1")
    ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
    
    
    
    '''
    print(ratings.shape)
    print(list(ratings.columns))
    
    plt.rc("font", size=15)
    ratings.Book-Rating.value_counts(sort=False).plot(kind='bar')
    plt.title('Rating Distribution\n')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('system1.png', bbox_inches='tight')
    plt.show()
    
    print(books.shape)
    print(list(books.columns))
    
    print(users.shape)
    print(list(users.columns))
    
    users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
    plt.title('Age Distribution\n')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('system2.png', bbox_inches='tight')
    plt.show()
    
    rating_count = pd.DataFrame(ratings.groupby('ISBN')['Book-Rating'].count())
    rating_count.sort_values('Book-Rating', ascending=False).head()
    
    most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209', '0060928336', '0312195516'], index=np.arange(5), columns = ['ISBN'])
    most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
    most_rated_books_summary
    
    average_rating = pd.DataFrame(ratings.groupby('ISBN')['Book-Rating'].mean())
    average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['Book-Rating'].count())
    average_rating.sort_values('ratingCount', ascending=False).head()
    
    counts1 = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(counts1[counts1 >= 200].index)]
    counts = ratings['Book-Rating'].value_counts()
    ratings = ratings[ratings['Book-Rating'].isin(counts[counts >= 100].index)]
    
    ratings_pivot = ratings.pivot(index='User-ID', columns='ISBN').Book-Rating
    User-ID = ratings_pivot.index
    ISBN = ratings_pivot.columns
    print(ratings_pivot.shape)
    ratings_pivot.head()
    
    bones_ratings = ratings_pivot['0316666343']
    similar_to_bones = ratings_pivot.corrwith(bones_ratings)
    corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
    corr_bones.dropna(inplace=True)
    corr_summary = corr_bones.join(average_rating['ratingCount'])
    corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR', ascending=False).head(10)
    
    books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', '0345342968', '0060930535', '0375707972', '0684872153'], 
                                      index=np.arange(9), columns=['ISBN'])
    corr_books = pd.merge(books_corr_to_bones, books, on='ISBN')
    corr_books
    '''
    
    
    
    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    columns = [ 'Year-Of-Publication', 'Publisher', 'Image-Url-S', 'Image-Url-M', 'Image-Url-L']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating.head()
    
    combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['Book-Title'])
    
    book_ratingCount = (combine_book_rating.
         groupby(by = ['Book-Title'])['Book-Rating'].
         count().
         reset_index().
         rename(columns = {'Book-Rating': 'totalRatingCount'})
         [['Book-Title', 'totalRatingCount']]
        )
    #book_ratingCount.head()
    
    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
    rating_with_totalRatingCount.head()
    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    #print(book_ratingCount['totalRatingCount'].describe())
    
    #print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))
    
    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    
    #booktitle = rating_popular_book['Book-Title']
    #print(book_name in rating_popular_book['Book-Title'])
    '''
    if not(book_name in rating_popular_book['Book-Title']):
        print('\'', book_name, '\' is not in the book list')
        sys.exit();
    '''
    
    combined = rating_popular_book.merge(users, left_on = 'User-ID', right_on = 'User-ID', how = 'left')
    
    #print(combined[~combined['Location'].str.contains("usa|canada")].Location)
    
    us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada|united kingdom")]
    us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
    us_canada_user_rating.head()
    
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['User-ID', 'Book-Title'])
    
    '''
    
    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
    us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)
    
    
    from sklearn.neighbors import NearestNeighbors
    
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(us_canada_user_rating_matrix)
    
    #print(us_canada_user_rating_pivot['Pride and Prejudice'].index)
    query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
    print('Query Index ' , query_index)
    query_index = 156
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
    
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
    
    
    '''
    
    us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'User-ID', columns = 'Book-Title', values = 'Book-Rating').fillna(0)
    us_canada_user_rating_pivot2.head()
    
    us_canada_user_rating_pivot2.shape
    
    X = us_canada_user_rating_pivot2.values.T
    X.shape
    
    import sklearn
    from sklearn.decomposition import TruncatedSVD
    
    SVD = TruncatedSVD(n_components=50)
    matrix = SVD.fit_transform(X)
    matrix.shape
    
    import warnings
    warnings.filterwarnings("ignore",category =RuntimeWarning)
    corr = np.corrcoef(matrix)
    corr.shape
    
    us_canada_book_title = us_canada_user_rating_pivot2.columns
    us_canada_book_list = list(us_canada_book_title)
    
    #book_name = input('\n\nEnter the Book Title:\t')
    '''
    book_name = 'A Walk to Remember'
    book_name = 'Alice in Wonderland'
    book_name = 'Catch 22'
    book_name = '1984'
    
    12: 20, 21,22
    20: 19
    50 : 24
    100 : 18,23
    200 : 25
    
    '''
    book_index = us_canada_book_list.index(book_name)
    
    corr_book = corr[book_index];
    
    df=pd.DataFrame()
    df['Title'] = us_canada_book_list
    df['Score'] = corr_book
    
    df = df.sort_values(by='Score', ascending=False)
    
    print('Recommendations for :',book_name)
    
    print(df[1:11].to_string(index=False))
    
    #raw_recommends = list(us_canada_book_title[corr_book >=0.9])
    
    
    
    '''
    for i in range(len(raw_recommends)):
        if i >= 10: 
            break
        print( i+1,'.',raw_recommends[i])
    '''    
    
    print("--- {} minutes ---"  .format((time.time() - start_time)/60))
    
        
    '''
        add user recommendations
        fix recommendations
        implement baseline for association analysis
        complete ppt
        contains search / UI deevlopement
        
        change dataset
        report output for values of k = 20,250,500, 100
        
        understand latent factorss
        what matrixi s used for recommendation prediction in svd
    
    
    '''
if __name__ == '__main__':
    main()

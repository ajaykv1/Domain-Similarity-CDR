from utils import *

print("* Loading GloVe File *\n")
glove_embedding = KeyedVectors.load_word2vec_format('./GloVe_File/glove.6B.300d.txt', binary=False)

#######################
# Loading Amazon Data #
#######################

print("* Loading Amazon Dataframes *")

music_instruments_df = pd.read_pickle('./dataframes/music_instruments_df.pkl') 
software_df = pd.read_pickle('./dataframes/software_df.pkl') 
video_games_df = pd.read_pickle('./dataframes/video_games_df.pkl') 

print("* Pre-processing Amazon Dataframes *\n")

software_df = software_df.drop(['overall', 'reviewerID'], axis = 1)
video_games_df = video_games_df.drop(['overall', 'reviewerID'], axis = 1)
music_instruments_df = music_instruments_df.drop(['overall', 'reviewerID'], axis = 1)

software_df['asin'] = software_df['asin'].astype(str)
video_games_df['asin'] = video_games_df['asin'].astype(str)
music_instruments_df['asin'] = music_instruments_df['asin'].astype(str)

# Dropping Duplicates
video_games_df = video_games_df.drop_duplicates(subset='asin', keep="first")
music_instruments_df = music_instruments_df.drop_duplicates(subset='asin', keep="first")
software_df = software_df.drop_duplicates(subset='asin', keep="first")

######################
# Loading Books Data #
######################

print("* Loading Books Dataframes *")

romance_df = pd.read_pickle('./dataframes/romance_df.pkl') 
nonfiction_df = pd.read_pickle('./dataframes/nonfiction_df.pkl') 
historical_df = pd.read_pickle('./dataframes/historical_df.pkl') 

print("* Pre-processing Books Dataframes *\n")

romance_df = romance_df.drop(['user_id', 'rating', 'genre'], axis = 1)
historical_df = historical_df.drop(['user_id', 'rating', 'genre'], axis = 1)
nonfiction_df = nonfiction_df.drop(['user_id', 'rating', 'genre'], axis = 1)

romance_df = romance_df.groupby(['item_id'])['tag'].apply(', '.join).reset_index()
historical_df = historical_df.groupby(['item_id'])['tag'].apply(', '.join).reset_index()
nonfiction_df = nonfiction_df.groupby(['item_id'])['tag'].apply(', '.join).reset_index()

##########################
# Loading Movielens Data #
##########################

print("* Loading Movielens Dataframes *")

action_df = pd.read_pickle('./dataframes/action_df.pkl') 
comedy_df = pd.read_pickle('./dataframes/comedy_df.pkl') 
adventure_df = pd.read_pickle('./dataframes/adventure_df.pkl') 

print("* Pre-processing Movielens Dataframes *\n")

action_df = action_df.drop(['genres', 'rating', 'userId'], axis=1)
comedy_df = comedy_df.drop(['genres', 'rating', 'userId'], axis=1)
adventure_df = adventure_df.drop(['genres', 'rating', 'userId'], axis=1)

action_df['movieId'] = action_df['movieId'].astype(str)
action_df['tag'] = action_df['tag'].astype(str)

comedy_df['movieId'] = comedy_df['movieId'].astype(str)
comedy_df['tag'] = comedy_df['tag'].astype(str)

adventure_df['movieId'] = adventure_df['movieId'].astype(str)
adventure_df['tag'] = adventure_df['tag'].astype(str)

action_df = action_df.groupby(['movieId'])['tag'].apply(', '.join).reset_index()
comedy_df = comedy_df.groupby(['movieId'])['tag'].apply(', '.join).reset_index()
adventure_df = adventure_df.groupby(['movieId'])['tag'].apply(', '.join).reset_index()

# Creating tuples where first value in tuple is item_id and second value is embedding e.g. (1, [0.12,0.6,..])

print("* Creating Tuples for Amazon Dataset *")

software_tups = amazon_with_tag_emb(software_df, glove_embedding)
video_games_tups = amazon_with_tag_emb(video_games_df, glove_embedding)
music_instruments_tups = amazon_with_tag_emb(music_instruments_df, glove_embedding)

print("* Creating Tuples for Movielens Dataset *")

action_tups = movielens_with_tag_emb(action_df, glove_embedding)
adventure_tups = movielens_with_tag_emb(adventure_df, glove_embedding)
comedy_tups = movielens_with_tag_emb(comedy_df, glove_embedding)

print("* Creating Tuples for Books Dataset *\n")

romance_tups = books_with_tag_emb(romance_df, glove_embedding)
historical_tups = books_with_tag_emb(historical_df, glove_embedding)
nonfiction_tups = books_with_tag_emb(nonfiction_df, glove_embedding)

sample_num = 100

print("* Sampling 100 items from Software Domain *")

random.seed(44)
software = random.sample(software_tups, sample_num)
music_instruments = random.sample(music_instruments_tups, sample_num)
video_games = random.sample(video_games_tups, sample_num)

print("* Sampling 100 items from Books Domain *")

random.seed(44)
romance = random.sample(romance_tups, sample_num)
historical = random.sample(historical_tups, sample_num)
nonfiction = random.sample(nonfiction_tups, sample_num)

print("* Sampling 100 items from Movielens Domain *\n")

random.seed(44)
action = random.sample(action_tups, sample_num)
adventure = random.sample(adventure_tups, sample_num)
comedy = random.sample(comedy_tups, sample_num)

# Opening files to write results
movielens_file = open('./pairwise_similarities/movielens_pairwise_sims.txt', 'w')
amazon_file = open('./pairwise_similarities/amazon_pairwise_sims.txt', 'w')
books_file = open('./pairwise_similarities/books_pairwise_sims.txt', 'w')

#####################
# Movielens Results #
#####################

print("* Calculating cross-domain similarities within Movielens dataset *")

action_adventure_sim = calc_cross_domain_sim(action, adventure)
action_comedy_sim = calc_cross_domain_sim(action, comedy)
comedy_adventure_sim = calc_cross_domain_sim(comedy, adventure)

# Writing Movielens Results
print("* Writing results to file *\n")

movielens_file.write("---------- Results for Movielens Pairwise Similarities ----------\n\n")
movielens_file.write("Cross-Domain Similarities........\n\n")
movielens_file.write("Action & Adventure: " + str(action_adventure_sim) + "\n")
movielens_file.write("Action & Comedy: " + str(action_comedy_sim) + "\n")
movielens_file.write("Comedy & Adventure: " + str(comedy_adventure_sim) + "\n\n")

#################
# Books Results #
#################

print("* Calculating cross-domain similarities within Books dataset *")

romance_historical_sim = calc_cross_domain_sim(romance, historical)
romance_nonfiction_sim = calc_cross_domain_sim(romance, nonfiction)
nonfiction_historical_sim = calc_cross_domain_sim(nonfiction, historical)

# Writing Books Results
print("* Writing results to file *\n")

books_file.write("---------- Results for Books Pairwise Similarities ----------\n\n")
books_file.write("Cross-Domain Similarities........\n\n")
books_file.write("Romance & Historical: " + str(romance_historical_sim) + "\n")
books_file.write("Romance & Nonfiction: " + str(romance_nonfiction_sim) + "\n")
books_file.write("Nonfiction & Historical: " + str(nonfiction_historical_sim) + "\n\n")

##################
# Amazon Results #
##################

print("* Calculating cross-domain similarities within Amazon dataset *")

software_music_instruments_sim = calc_cross_domain_sim(software, music_instruments)
software_video_games_sim = calc_cross_domain_sim(software, video_games)
video_games_music_instruments_sim = calc_cross_domain_sim(video_games, music_instruments)

# Writing Amazon Results
print("* Writing results to file *\n")

amazon_file.write("---------- Results for Amazon Pairwise Similarities ----------\n\n")
amazon_file.write("Cross-Domain Similarities........\n\n")
amazon_file.write("Software & Music Instruments: " + str(software_music_instruments_sim) + "\n")
amazon_file.write("Software & Video Games: " + str(software_video_games_sim) + "\n")
amazon_file.write("Video Games & Music Instruments: " + str(video_games_music_instruments_sim) + "\n\n")

# Closing Files
movielens_file.close()
books_file.close()
amazon_file.close()
#%%























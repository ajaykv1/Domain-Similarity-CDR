from utils import *

print("* Loading GloVe File *\n")
glove_embedding = KeyedVectors.load_word2vec_format('./GloVe_File/glove.6B.300d.txt', binary=False)

#######################
# Loading Amazon Data #
#######################

amazon_list_max = []
amazon_list_min = []

print("* Loading Amazon Dataframes *\n")

software_load_df = pd.read_pickle('./dataframes/software_df.pkl') 
print("* Gathering Software Domain Tags *")
software_tags = gather_tags_amazon(software_load_df)[0]
software_list = gather_tags_amazon(software_load_df)[1]
amazon_list_max.append(max(software_list.values()))
amazon_list_min.append(min(software_list.values()))

music_instruments_load_df = pd.read_pickle('./dataframes/music_instruments_df.pkl') 
print("* Gathering Music Instruments Domain Tags *")
music_instruments_tags = gather_tags_amazon(music_instruments_load_df)[0]
music_instruments_list = gather_tags_amazon(music_instruments_load_df)[1]
amazon_list_max.append(max(music_instruments_list.values()))
amazon_list_min.append(min(music_instruments_list.values()))

video_games_load_df = pd.read_pickle('./dataframes/video_games_df.pkl') 
print("* Gathering Video Games Domain Tags *\n")
video_games_tags = gather_tags_amazon(video_games_load_df)[0]
video_games_list = gather_tags_amazon(video_games_load_df)[1]
amazon_list_max.append(max(video_games_list.values()))
amazon_list_min.append(min(video_games_list.values()))

max_amazon = max(amazon_list_max)
min_amazon = max(amazon_list_min)

max_amazon_software = amazon_list_max[0]
max_amazon_music_instruments = amazon_list_max[1]
max_amazon_video_games = amazon_list_max[2]

print("Top 5 Software Tags:")
s = sorted(software_list.items(), key=lambda x: x[1], reverse = True)
print(s[:5])
print("Total Software Tags Count:  " + str(len(s)))

print("\nTop 5 Video Games Tags:")
v = sorted(video_games_list.items(), key=lambda x: x[1], reverse = True)
print(v[:5])
print("Total Video Games Tags Count:  " + str(len(v)))

print("\nTop 5 Music Instruments Tags:")
m = sorted(music_instruments_list.items(), key=lambda x: x[1], reverse = True)
print(m[:5])
print("Total Music Instruments Tags Count:  " + str(len(m)) + "\n")

print("*--- STATS ---*")
print("Max Tag Count: " + str(max_amazon))
print("Min Tag Count: " + str(min_amazon) + "\n")
print("Max Software Tag Count: " + str(max_amazon_software))
print("Max Music Instruments Tag Count: " + str(max_amazon_music_instruments))
print("Max Video Games Tag Count: " + str(max_amazon_video_games) + "\n")
print("Finished Amazon STATS\n")

print("* Converting Amazon Domains to Embeddings *\n")

software_embedding = get_domain_embedding(glove_embedding, software_list)
video_games_embedding = get_domain_embedding(glove_embedding, video_games_list)
music_instruments_embedding = get_domain_embedding(glove_embedding, music_instruments_list)

##########################
# Loading Movielens Data #
##########################

movielens_list_max = []
movielens_list_min = []

print("* Loading Movielens Dataframes *\n")

action_load_df = pd.read_pickle('./dataframes/action_df.pkl') 
print("* Gathering Action Domain Tags *")
action_tags = gather_tags_movie(action_load_df)[0]
action_list = gather_tags_movie(action_load_df)[1]
movielens_list_max.append(max(action_list.values()))
movielens_list_min.append(min(action_list.values()))

adventure_load_df = pd.read_pickle('./dataframes/adventure_df.pkl')
print("* Gathering Adventure Domain Tags *")
adventure_tags = gather_tags_movie(adventure_load_df)[0]
adventure_list = gather_tags_movie(adventure_load_df)[1]
movielens_list_max.append(max(adventure_list.values()))
movielens_list_min.append(min(adventure_list.values()))

comedy_load_df = pd.read_pickle('./dataframes/comedy_df.pkl') 
print("* Gathering Comedy Domain Tags *")
comedy_tags = gather_tags_movie(comedy_load_df)[0]
comedy_list = gather_tags_movie(comedy_load_df)[1]
movielens_list_max.append(max(comedy_list.values()))
movielens_list_min.append(min(comedy_list.values()))

max_movielens = max(movielens_list_max)
min_movielens = min(movielens_list_min)

max_movielens_action = movielens_list_max[0]
max_movielens_adventure = movielens_list_max[1]
max_movielens_comedy = movielens_list_max[2]

print("Top 5 Action Tags:")
a = sorted(action_list.items(), key=lambda x: x[1], reverse = True)
print(a[:5])
print("Total Action Tags Count:  " + str(len(a)))

print("\nTop 5 Adventure Tags:")
a = sorted(adventure_list.items(), key=lambda x: x[1], reverse = True)
print(a[:5])
print("Total Adventure Tags Count:  " + str(len(a)))

print("\nTop 5 Comedy Tags:")
c = sorted(comedy_list.items(), key=lambda x: x[1], reverse = True)
print(c[:5])
print("Total Comedy Tags Count:  " + str(len(c)) + "\n")

print("*--- STATS ---*")
print("Max Tag Count: " + str(max_movielens))
print("Min Tag Count: " + str(min_movielens))
print("Max Action Tag Count: " + str(max_movielens_action))
print("Max Adventure Tag Count: " + str(max_movielens_adventure))
print("Max Comedy Tag Count: " + str(max_movielens_comedy))
print("Finished Movielens STATS\n")

print("* Converting Movielens Domains to Embeddings *\n")

action_embedding = get_domain_embedding(glove_embedding, action_list)
adventure_embedding = get_domain_embedding(glove_embedding, adventure_list)
comedy_embedding = get_domain_embedding(glove_embedding, comedy_list)

######################
# Loading Books Data #
######################

books_list_max = []
books_list_min = []

print("* Loading Books Dataframes *\n")

nonfiction_load_df = pd.read_pickle('./dataframes/nonfiction_df.pkl') 
print("* Gathering Nonfiction Domain Tags *")
nonfiction_tags = gather_tags_books(nonfiction_load_df)[0]
nonfiction_list = gather_tags_books(nonfiction_load_df)[1]
books_list_max.append(max(nonfiction_list.values()))
books_list_min.append(min(nonfiction_list.values()))

historical_load_df = pd.read_pickle('./dataframes/historical_df.pkl') 
print("* Gathering Historical Domain Tags *")
historical_tags = gather_tags_books(historical_load_df)[0]
historical_list = gather_tags_books(historical_load_df)[1]
books_list_max.append(max(historical_list.values()))
books_list_min.append(min(historical_list.values()))

romance_load_df = pd.read_pickle('./dataframes/romance_df.pkl')
print("* Gathering Romance Domain Tags *") 
romance_tags = gather_tags_books(romance_load_df)[0]
romance_list = gather_tags_books(romance_load_df)[1]
books_list_max.append(max(romance_list.values()))
books_list_min.append(min(romance_list.values()))

max_books = max(books_list_max)
min_books = min(books_list_min)

max_books_nonfiction = books_list_max[0]
max_books_historical = books_list_max[1]
max_books_romance = books_list_max[2]

print("Top 5 Romance Tags:")
r = sorted(romance_list.items(), key=lambda x: x[1], reverse = True)
print(r[:5])
print("Total Romance Tags Count:  " + str(len(r)))

print("\nTop 5 Nonfiction Tags:")
n = sorted(nonfiction_list.items(), key=lambda x: x[1], reverse = True)
print(n[:5])
print("Total Nonfiction Tags Count:  " + str(len(n)))

print("\nTop 5 Historical Tags:")
h = sorted(historical_list.items(), key=lambda x: x[1], reverse = True)
print(h[:5])
print("Total Historical Tags Count:  " + str(len(h)) + "\n")

print("*--- STATS ---*")
print("Max Tag Count: " + str(max_books))
print("Min Tag Count: " + str(min_books) + "\n")
print("Max Nonfiction Tag Count: " + str(max_books_nonfiction))
print("Max Romancce Tag Count: " + str(max_books_romance) + "\n")
print("Finished Books STATS\n")

print("* Converting Books Domains to Embeddings *\n")

romance_embedding = get_domain_embedding(glove_embedding, romance_list)
historical_embedding = get_domain_embedding(glove_embedding, historical_list)
nonfiction_embedding = get_domain_embedding(glove_embedding, nonfiction_list)

###################################
# Saving Amazon Domain Embeddings #
###################################

print("* Saving Domain Embeddings from Amazon Dataset *")

with open('./domain_embeddings/software_tags_embedding.pkl', 'wb') as f:
    pickle.dump(software_embedding, f)
    
with open('./domain_embeddings/music_instruments_tags_embedding.pkl', 'wb') as f:
    pickle.dump(music_instruments_embedding, f)

with open('./domain_embeddings/video_games_tags_embedding.pkl', 'wb') as f:
    pickle.dump(video_games_embedding, f)

######################################
# Saving Movielens Domain Embeddings #
######################################

print("* Saving Domain Embeddings from Movielens Dataset *")

with open('./domain_embeddings/action_tags_embedding.pkl', 'wb') as f:
    pickle.dump(action_embedding, f)
    
with open('./domain_embeddings/adventure_tags_embedding.pkl', 'wb') as f:
    pickle.dump(adventure_embedding, f)
    
with open('./domain_embeddings/comedy_tags_embedding.pkl', 'wb') as f:
    pickle.dump(comedy_embedding, f)

##################################
# Saving Books Domain Embeddings #
##################################

print("* Saving Domain Embeddings from Books Dataset *")

with open('./domain_embeddings/romance_tags_embedding.pkl', 'wb') as f:
    pickle.dump(romance_embedding, f)

with open('./domain_embeddings/historical_tags_embedding.pkl', 'wb') as f:
    pickle.dump(historical_embedding, f)

with open('./domain_embeddings/nonfiction_tags_embedding.pkl', 'wb') as f:
    pickle.dump(nonfiction_embedding, f)








from utils import *

######################################
# Loading Movielens Glove Embeddings #
######################################

print("* Loading Domain Embeddings From Movielens Dataset *")

action_tags_file = open('./domain_embeddings/action_tags_embedding.pkl', 'rb')
action_glove_emb = pickle.load(action_tags_file)

adventure_tags_file = open('./domain_embeddings/adventure_tags_embedding.pkl', 'rb')
adventure_glove_emb = pickle.load(adventure_tags_file)

comedy_tags_file = open('./domain_embeddings/comedy_tags_embedding.pkl', 'rb')
comedy_glove_emb = pickle.load(comedy_tags_file)

action_tags_file.close()
adventure_tags_file.close()
comedy_tags_file.close()

##################################
# Loading Books Glove Embeddings #
##################################

print("* Loading Domain Embeddings From Books Dataset *")

romance_tags_file = open('domain_embeddings/romance_tags_embedding.pkl', 'rb')
romance_glove_emb = pickle.load(romance_tags_file)

historical_tags_file = open('./domain_embeddings/historical_tags_embedding.pkl', 'rb')
historical_glove_emb = pickle.load(historical_tags_file)

nonfiction_tags_file = open('./domain_embeddings/nonfiction_tags_embedding.pkl', 'rb')
nonfiction_glove_emb = pickle.load(nonfiction_tags_file)

romance_tags_file.close()
historical_tags_file.close()
nonfiction_tags_file.close()

###################################
# Loading Amazon Glove Embeddings #
###################################

print("* Loading Domain Embeddings From Amazon Dataset *\n")

software_tags_file = open('./domain_embeddings/software_tags_embedding.pkl', 'rb')
software_glove_emb = pickle.load(software_tags_file)

music_instruments_tags_file = open('./domain_embeddings/music_instruments_tags_embedding.pkl', 'rb')
music_instruments_glove_emb = pickle.load(music_instruments_tags_file)

video_games_tags_file = open('./domain_embeddings/video_games_tags_embedding.pkl', 'rb')
video_games_glove_emb = pickle.load(video_games_tags_file)

software_tags_file.close()
music_instruments_tags_file.close()
video_games_tags_file.close()

#########################################################################################
# Creating Files to Write Similarity Values Between Domain Combinations in Each Dataset #
#########################################################################################

amazon_file = open("./domain_embedding_similarity_results/amazon_domains_sim.txt", "w")
movielens_file = open("./domain_embedding_similarity_results/movielens_domains_sim.txt", "w")
books_file = open("./domain_embedding_similarity_results/books_domains_sim.txt", "w")

amazon_file.write("***** AMAZON DATA *****\n\n")
movielens_file.write("***** MOVIELENS DATA *****\n\n")
books_file.write("***** BOOKS DATA *****\n\n")

# Movielens Glove Similarities

print("* Computing Similarities Between Domain Embeddings in the Movielens Dataset *")

sim_action_adventure = get_similarity_embedding(action_glove_emb, adventure_glove_emb)
sim_action_comedy = get_similarity_embedding(action_glove_emb, comedy_glove_emb)
sim_adventure_comedy = get_similarity_embedding(adventure_glove_emb, comedy_glove_emb)

print("* Writing Similarity Results to File *\n")

movielens_file.write("Tags Represented as Glove Embeddings (Weighted Average).....\n\n")

movielens_file.write("----- Cosine Similarity -----\n")
movielens_file.write("Action and Adventure: " + str(sim_action_adventure) + "\n")
movielens_file.write("Action and Comedy: " + str(sim_action_comedy) + "\n")
movielens_file.write("Adventure and Comedy: " + str(sim_adventure_comedy) + "\n")

# Books Glove Similarities

print("* Computing Similarities Between Domain Embeddings in the Books Dataset *")

sim_nonfiction_romance = get_similarity_embedding(nonfiction_glove_emb, romance_glove_emb)
sim_nonfiction_historical = get_similarity_embedding(nonfiction_glove_emb, historical_glove_emb)
sim_romance_historical = get_similarity_embedding(romance_glove_emb, historical_glove_emb)

print("* Writing Similarity Results to File *\n")

books_file.write("Tags Represented as Glove Embeddings (Weighted Average).....\n\n")

books_file.write("----- Cosine Similarity -----\n")
books_file.write("nonfiction and romance: " + str(sim_nonfiction_romance) + "\n")
books_file.write("nonfiction and historical: " + str(sim_nonfiction_historical) + "\n")
books_file.write("romance and historical: " + str(sim_romance_historical) + "\n")

# Amazon GloVe Similarities

print("* Computing Similarities Between Domain Embeddings in the Amazon Dataset *")

sim_software_music_instruments = get_similarity_embedding(software_glove_emb, music_instruments_glove_emb)
sim_software_video_games = get_similarity_embedding(software_glove_emb, video_games_glove_emb)
sim_music_instruments_video_games = get_similarity_embedding(music_instruments_glove_emb, video_games_glove_emb)

print("* Writing Similarity Results to File *\n")

amazon_file.write("Tags Represented as Glove Embeddings (Weighted Average).....\n\n")

amazon_file.write("----- Cosine Similarity -----\n")
amazon_file.write("Software and Music Instruments: " + str(sim_software_music_instruments) + "\n")
amazon_file.write("Software and Video Games: " + str(sim_software_video_games) + "\n")
amazon_file.write("Music Instruments and Video Games: " + str(sim_music_instruments_video_games) + "\n\n")

# Closing Files to Write

amazon_file.close()
movielens_file.close()
books_file.close()






















from custom_utils import *

print("* Loading GloVe File *\n")
glove_embedding = KeyedVectors.load_word2vec_format('../Full_Project_Code/GloVe_File/GloVe_File/glove.6B.300d.txt', binary=False)

################
# Loading Data #
################

print("* Loading Dataframes *")

source_domain_df = pd.read_pickle('./dataframes/source_domain_df.pkl') 
target_domain_df = pd.read_pickle('./dataframes/target_domain_df.pkl')  

print("* Pre-processing Dataframes *\n")

source_domain_df['movieId'] = source_domain_df['movieId'].astype(str)
source_domain_df['tag'] = source_domain_df['tag'].astype(str)

target_domain_df['movieId'] = target_domain_df['movieId'].astype(str)
target_domain_df['tag'] = target_domain_df['tag'].astype(str)

source_domain_df = source_domain_df.groupby(['movieId'])['tag'].apply(', '.join).reset_index()
target_domain_df = target_domain_df.groupby(['movieId'])['tag'].apply(', '.join).reset_index()

# Creating tuples where first value in tuple is item_id and second value is embedding e.g. (1, [0.12,0.6,..])

print("* Creating Tuples for Domain Datasets *")

souce_domain_tups = custom_data_with_tag_emb(source_domain_df, glove_embedding)
target_domain_tups = custom_data_with_tag_emb(target_domain_df, glove_embedding)

sample_num = 100

print("* Sampling 100 items from custom_data Domain *\n")

random.seed(44)
source = random.sample(source_domain_tups, sample_num)
target = random.sample(target_domain_tups, sample_num)

# Opening file to write results
results_file = open('./pairwise_similarities/custom_pairwise_sims.txt', 'w')

#################################
# Pairwise Similarities Results #
#################################

print("* Calculating cross-domain similarities for custom dataset *")

source_target_sim = calc_cross_domain_sim(source, target)

# Writing custom_data Results
print("* Writing results to file *\n")

results_file.write("---------- Results for Pairwise Similarities ----------\n\n")
results_file.write("Cross-Domain Similarities........\n\n")
results_file.write("source & Target: " + str(source_target_sim) + "\n")

# Closing Files
results_file.close()























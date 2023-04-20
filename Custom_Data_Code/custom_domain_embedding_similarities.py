from custom_utils import *

######################################
# Loading Movielens Glove Embeddings #
######################################

print("* Loading Domain Embeddings From Custom Dataset *")

source_domain_tags_file = open('./domain_embeddings/source_domain_embedding.pkl', 'rb')
source_domain_glove_emb = pickle.load(action_tags_file)

target_domain_tags_file = open('./domain_embeddings/target_domain_embedding.pkl', 'rb')
target_domain_glove_emb = pickle.load(adventure_tags_file)

action_tags_file.close()
adventure_tags_file.close()

#########################################################################################
# Creating Files to Write Similarity Values Between Domain Combinations in Each Dataset #
#########################################################################################

results_file = open("./domain_embedding_similarity_results/custom_domains_sim.txt", "w")

results_file.write("***** CUSTOM DATA *****\n\n")

# Glove Similarities

print("* Computing Similarities Between Domain Embeddings in the Custom Dataset *")

sim_source_target = get_similarity_embedding(source_domain_glove_emb, target_domain_glove_emb)

print("* Writing Similarity Results to File *\n")

results_file.write("Tags Represented as Glove Embeddings (Weighted Average).....\n\n")

results_file.write("----- Cosine Similarity -----\n")
results_file.write("Source and Target: " + str(sim_source_target) + "\n")

# Closing Files to Write
results_file.close()






















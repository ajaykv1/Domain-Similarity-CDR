from custom_utils import *

print("* Loading GloVe File *\n")
glove_embedding = KeyedVectors.load_word2vec_format('../Full_Project_Code/GloVe_File/GloVe_File/glove.6B.300d.txt', binary=False)

#######################
# Loading Custom Data #
#######################

data_list_max = []
data_list_min = []

print("* Loading Custom Dataframes *\n")

source_domain_load_df = pd.read_pickle('./dataframes/source_domain_df.pkl') 
print("* Gathering Source Domain Tags *")
source_domain_tags = gather_tags_custom(source_domain_load_df)[0]
source_domain_list = gather_tags_custom(source_domain_load_df)[1]
data_list_max.append(max(source_domain_list.values()))
data_list_min.append(min(source_domain_list.values()))

target_domain_load_df = pd.read_pickle('./dataframes/target_domain_df.pkl') 
print("* Gathering Target Domain Tags *")
target_domain_tags = gather_tags_custom(target_domain_load_df)[0]
target_domain_list = gather_tags_custom(target_domain_load_df)[1]
data_list_max.append(max(target_domain_list.values()))
data_list_min.append(min(target_domain_list.values()))

max_data = max(data_list_max)
min_data = max(data_list_min)

max_source_domain = data_list_max[0]
max_target_domain = data_list_max[1]

print("Top 5 Source Domain Tags:")
s = sorted(source_domain_list.items(), key=lambda x: x[1], reverse = True)
print(s[:5])
print("Total Source Domain Tags Count:  " + str(len(s)))

print("\nTop 5 Target Domain Tags:")
v = sorted(target_domain_list.items(), key=lambda x: x[1], reverse = True)
print(v[:5])
print("Total Target Domain Tags Count:  " + str(len(v)))

print("*--- STATS ---*")
print("Max Tag Count: " + str(max_data))
print("Min Tag Count: " + str(min_data) + "\n")
print("Max Source Domain Tag Count: " + str(max_source_domain))
print("Max Target Domain Tag Count: " + str(max_target_domain))
print("Finished Custom Dataset STATS\n")

print("* Converting Source & Target Domains to Embeddings *\n")

source_domain_embedding = get_domain_embedding(glove_embedding, source_domain_list)
target_domain_embedding = get_domain_embedding(glove_embedding, target_domain_list)

############################################
# Saving Source & Target Domain Embeddings #
############################################

print("* Saving Domain Embeddings from Custom Dataset *")

with open('./domain_embeddings/source_domain_embedding.pkl', 'wb') as f:
    pickle.dump(source_domain_embedding, f)
    
with open('./domain_embeddings/target_domain_embedding.pkl', 'wb') as f:
    pickle.dump(target_domain_embedding, f)








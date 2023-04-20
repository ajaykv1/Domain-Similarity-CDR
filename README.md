# Quantifying Similarity Between Domains for Cross-Domain Recommender Systems

 Abstract:

Cross-domain recommender systems have recently emerged as an effective way to alleviate the cold-start and sparsity issues that single-domain recommender systems face. They leverage information from different source domains, and transfer the information to the target domain to improve recommendations. In this paper, we present a novel systematic approach for quantifying similarity between domains in the context of cross-domain recommendation. Within the systematic approach, we present two original similarity metrics that quantify similarity between pairs of domains. The first similarity metric makes use of natural language processing (NLP) techniques to represent each domain as an embedding, in order to compute similarity. The second similarity metric finds the item-level similarities across domains to measure similarity. Our extensive empirical evaluation on different domain combinations demonstrate that the state-of-the-art cross-domain algorithms do not perform significantly better when using source domains that are more similar to the target domain, compared to using source domains that are less similar to the target domain. We measure similarity between domains using the two novel metrics, but we find that no matter how similarity is measured, it does not correlate with recommendation performance of the state-of-the-art algorithms, and thus the algorithms are not yet able to exploit similarity in an effective way.

# Usage

This is the code repository for this research project. This repository contains the source code of the two similarity metrics that were presented in the paper (***Embedding-based Domain Similarity*** & ***Inter-domain Item Similarity***). There are two folders: `Full_Project_Code` and `Custom_Project_Code`. The `Full_Project_Code` folder contains the code and the data we used to run the experiments from our paper. The `Custom_Project_Code` folder allows users to use our similarity metrics to compute similarities between their own source and target domain data. 

## Files/Directories Used to Run Experiments From Our Paper

The data that we used to run experiements is in the `Full_Project_Code` directory, and below are details regarding the files:

1. `GloVe_File`: This directory contains the pre-trained GloVe embeddings file that we use to retrieve embeddings for tags.
2. `dataframes`: This directory contains the dataframes for every domain that we used in the study.
3. `domain_embeddings`: This directory contains domain embeddings for each domain, which were created by running `create_domain_embeddings.py`. 
4. `domain_embedding_similarity_results`: This directory contains the similarity values between different domain combinations across three datasets using the ***Embedding-based Domain Similarity*** method.
5. `pairwise_similarities`: This directory contains the similarity values between different domain combinations across three datasets using the ***Inter-domain Item Similarity*** method.
6. `create_domain_embeddings.py`: This file created the domain embeddings for each domain based on the dataframes for each domain.
7. `domain_embedding_similarities.py`: This file computes the similarity between domain embeddings using the ***Embedding-based Domain Similarity*** method, and writes the results to the `domain_embedding_similarity_results` directory.
8. `pairwise_similarities.py`: This file computes the similarity betwween domains using the ***Inter-domain Item Similarity*** method, and writes the results to the `pairwise_similarities` directory.
9. `utils.py`: This function contains helper functions that are used throughout the python files in this project repository.

### Running Experiments From Our Paper

1. Enter the folder that contains the data we used for experimentaion 
   - `cd Full_Project_Code`
2. To retrieve the similarity values between domains using the ***Embedding-based Domain Similarity*** method, run the command below:
   - `python3 domain_embedding_similarities.py`
3. To retrieve the similarity values between domains using the ***Inter-domain Item Similarity*** method, run the command below:
   - `python3 pairwise_similarities.py`

## Using Custom Data Run Your Own Experiments

To run the similarity metrics using your own data, navigate to the `Custom_Project_Code` directory:

1. `dataframes`: This directory should contain the dataframes for your source and target domains. Make sure each dataframe has only two columns (`item_id` & `tags`). The `item_id` column should be an integer, and the `tags` columns should be a string with tags seperated by commas. Convert your dataframes into pickle files using `pandas.DataFrame.to_pickle()` and place the pickle files in this directory. The names of the dataframes **must** be `source_domain_df` and `target_domain_df`. 
2. `domain_embeddings`: This directory will contain domain embeddings for your source and target domains, which are created by running `custom_domain_embeddings.py`. 
3. `domain_embedding_similarity_results`: This directory will contain the similarity values between your source and target domains using the ***Embedding-based Domain Similarity*** method.
4. `pairwise_similarities`: This directory contains the similarity values between your source and target domains using the ***Inter-domain Item Similarity*** method.
5. `custom_domain_embeddings.py`: This file creates the domain embeddings for your source and target domains based on the dataframes in the `dataframes` directory.
6. `custom_domain_embedding_similarities.py`: This file computes the similarity between your domain embeddings using the ***Embedding-based Domain Similarity*** method, and writes the results to the `domain_embedding_similarity_results` directory.
7. `custom_pairwise_similarities.py`: This file computes the similarity between your domains using the ***Inter-domain Item Similarity*** method, and writes the results to the `pairwise_similarities` directory.
8. `custom_utils.py`: This function contains helper functions that are used throughout the python files in this project repository and extra functions to deal with your custom data.

### Running ***Embedding-based Domain Similarity*** Using Custom Source and Target Domains

1. Enter the folder that contains the data we used for experimentaion 
   - `cd Custom_Project_Code`
2. Create two dataframes called `source_domain_df` and `target_domain_df` that have two columns (`item_id` & `tags`).
3. Convert your dataframes into pickle files, and add them to the `dataframes` directory.
4. Create domain embeddings for both your source and target domains:
   - `python3 custom_domain_embeddings.py`
6. To retrieve the similarity values between domains using the ***Embedding-based Domain Similarity*** method, run the command below:
   - `python3 custom_domain_embedding_similarities.py`

### Running ***Inter-domain Item Similarity*** Using Custom Source and Target Domains

1. Enter the folder that contains the data we used for experimentaion 
   - `cd Custom_Project_Code`
2. Create two dataframes called `source_domain_df` and `target_domain_df` that have two columns (`item_id` & `tags`).
3. Convert your dataframes into pickle files, and add them to the `dataframes` directory.
4. To retrieve the similarity values between domains using the ***Inter-domain Item Similarity*** method, run the command below:
   - `python3 custom_pairwise_similarities.py`









`

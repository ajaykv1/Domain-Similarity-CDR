# Quantifying Similarity Between Domains for Cross-Domain Recommender Systems

 Abstract:

Cross-domain recommender systems have recently emerged as an effective way to alleviate the cold-start and sparsity issues that single-domain recommender systems face. They leverage information from different source domains, and transfer the information to the target domain to improve recommendations. In this paper, we present a novel systematic approach for quantifying similarity between domains in the context of cross-domain recommendation. Within the systematic approach, we present two original similarity metrics that quantify similarity between pairs of domains. The first similarity metric makes use of natural language processing (NLP) techniques to represent each domain as an embedding, in order to compute similarity. The second similarity metric finds the item-level similarities across domains to measure similarity. Our extensive empirical evaluation on different domain combinations demonstrate that the state-of-the-art cross-domain algorithms do not perform significantly better when using source domains that are more similar to the target domain, compared to using source domains that are less similar to the target domain. We measure similarity between domains using the two novel metrics, but we find that no matter how similarity is measured, it does not correlate with recommendation performance of the state-of-the-art algorithms, and thus the algorithms are not yet able to exploit similarity in an effective way.

# Usage

This is the code repository for this research project. This repository contains the source code of the two similarity metrics that were presented in the paper (Embedding-based Domain Similarity & Inter-domain Item Similarity). There are two folders: "Full_Project_Code" and "Custom_Project_Code". The Full_Project_Code folder contains the code and the data we used to run the experiments from our paper. The Custom_Project_Code folder allows users to use our similarity metrics to compute similarities between their own source and target domain data. 

'''
hello
'''

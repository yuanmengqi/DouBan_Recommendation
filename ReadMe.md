# Intro
This is the repository for the project **Data Collection, Retrieval, and Recommendation on Douban**

#### Traditional Recommendation Methods
* Developed web crawlers in `Data_Collection_and_Retrieval/book.ipynb` and `Data_Collection_and_Retrieval/movie.ipynb`.
* Used PKU segmentation for tokenization, built an inverted index table, and compressed the inverted index table in `Data_Collection_and_Retrieval/pku_seg.ipynb`.
  * Merged synonyms using the synonym dictionary: `Data_Collection_and_Retrieval/chinese_dictionary-master/dict_antonym.txt`.
  * Removed stop words using the stop words list: `Data_Collection_and_Retrieval/stopwords-master`.
* Implemented boolean retrieval in `Data_Collection_and_Retrieval/search.ipynb` and `Data_Collection_and_Retrieval/search_zipped.ipynb` (for block-compressed inverted index).
* Implemented item-based and user-based recommendations in `Data_Collection_and_Retrieval/Recommendation/Item-based_cf_recommend.ipynb`, `Data_Collection_and_Retrieval/Recommendation/user_based_cf_recommend_book.ipynb`, and `Data_Collection_and_Retrieval/Recommendation/user_based_cf_recommend_movie.ipynb`.

#### Knowledge Graph-Based Recommendation
* Knowledge Graph Generation:
  * Implemented graph extraction in `Knowledge_Graph_Recommendation/entity_extract.ipynb`.
  * Mapped graph entities to numbers in `Knowledge_Graph_Recommendation/entity2number.ipynb`.
* Knowledge Graph-Based Recommendation:
  * Implemented embedding-based recommendation in `Knowledge_Graph_Recommendation/recommendation/stage2/data_loader/loader_Embedding_based.py` and `Knowledge_Graph_Recommendation/recommendation/stage2/model/Embedding_based.py`.
  * Implemented GNN-based recommendation in `Knowledge_Graph_Recommendation/recommendation/stage2/data_loader/loader_GNN_based.py` and `Knowledge_Graph_Recommendation/recommendation/stage2/model/GNN_based.py`.

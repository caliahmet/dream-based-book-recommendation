# dream-based-book-recommendation

## [link](https://dream-based-book-recommendation.streamlit.app/)

Based on user's dream, this program extracts the features from that dream and recommend 5 books from the book dataset which has 1966 books.
It extracts themes with KeyBERT. It extracts emotions with using "j-hartmann/emotion-english-distilroberta-base" model.
It also uses 768-dimensional word embedding model called "all-mpnet-base-v2".
There are 7 possible emotions in this program.
1. Anger
2. Disgust
3. Fear
4. Joy
5. Neutral
6. Sadness
7. Surprise

It uses cosine similarity between books and user's inputs and recommend top 5 similar books.
This program is designed only to provide a spark for more advanced programs.

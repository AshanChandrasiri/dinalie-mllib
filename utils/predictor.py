# from pyspark.ml.feature import Word2VecModel
# from pyspark.ml.classification import LogisticRegressionModel
# # from utils.preprocess import preprocess_lyrics
#
# def load_mw2v_models():
#     mw2v_model_path = "merged_word2vec_model"
#     loaded_mw2v_model = Word2VecModel.load(mw2v_model_path)
#     return loaded_mw2v_model
#
# def load_lr_model():
#     mlr_loaded_model_path = "merged_word2vec_model"
#     loaded_mlr_model = LogisticRegressionModel.load(mlr_loaded_model_path)
#     return load_lr_model
#
#
# # def predict_genre_probabilities(lyrics, session):
#
# #   processed_lyrics = preprocess_lyrics(lyrics=lyrics, session=session)
#
# # #   # Transform the preprocessed lyrics into a Word2Vec vector
# # #   new_df = session.createDataFrame([(processed_lyrics,)], ["lyrics_stemmed"])
#
# # #   w2v_model = load_mw2v_models()
# # #   new_df_with_vectors = w2v_model.transform(new_df)
#
# # #   # Make predictions using the loaded Logistic Regression model
# # #   lr_model = load_lr_model()
# # #   prediction = lr_model.transform(new_df_with_vectors)
#
# # #   # Get the predicted genre index
# # #   probability_vector = prediction.select("probability").first()[0]
# # #   probability_percentages = [float(prob) for prob in probability_vector]
#
# # #   return probability_percentages
#
# # def predict_genre_probabilities_with_genre_names(lyrics):
# #   genre_probabilities = predict_genre_probabilities(lyrics)
# # #   print(genre_probabilities)
# # #   distinct_genres_indices = merged_df_indexed.select("genre_index").distinct().orderBy("genre_index").collect()
# # #   distinct_genres = {0.0:'pop', 1.0:'country', 2.0:'blues', 3.0:'rock', 4.0:'jazz', 5.0:'reggae', 6.0:'hip hop', 7.0:'soul'}
#
# # #   most_probable_genre = distinct_genres[genre_probabilities.index(max(genre_probabilities))]
# # #   print(f"Most Probable Genre: {most_probable_genre}")
# # #   print(f"Most Probable Genre: {distinct_genres[int(most_probable_genre)]}")
#
# # #   for index, probability in enumerate(genre_probabilities):
#
# # #     if 0 <= index < len(distinct_genres):
# # #       genre_name = distinct_genres[index]
# # #       print(f"Genre: {genre_name}, Probability: {probability}")
# # #     else:
# # #       print(f"Genre: Unknown, Probability: {probability}")

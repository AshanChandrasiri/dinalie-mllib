# from pyspark.ml.feature import Tokenizer
# from pyspark.ml.feature import StopWordsRemover
# from nltk.stem import PorterStemmer
# import pyspark.sql.functions as F
# from pyspark.sql.types import ArrayType, StringType


# def remove_punctuation(lyrics):
#   if lyrics is not None:
#     return lyrics.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
#   else:
#     return lyrics
  
# def stem_words(tokens):
#   ps = PorterStemmer()
#   stemmed_tokens = [ps.stem(token) for token in tokens]
#   return stemmed_tokens  
  

# def preprocess_lyrics(lyrics, session):
#   lyrics = lyrics.lower()
#   lyrics = remove_punctuation(lyrics)
#   tokenizer = Tokenizer(inputCol="lyrics", outputCol="lyrics_tokens")
#   df = session.createDataFrame([(lyrics,)], ["lyrics"])
#   df = tokenizer.transform(df)

#   # remover = StopWordsRemover(inputCol="lyrics_tokens", outputCol="lyrics_tokens_no_stopwords")
#   # df = remover.transform(df)

#   # stem_words_udf = F.udf(stem_words, ArrayType(StringType()))

#   # df = df.withColumn("lyrics_stemmed", stem_words_udf(df["lyrics_tokens_no_stopwords"]))
#   # stemmed_tokens = df.select("lyrics_stemmed").first()[0]
#   # return stemmed_tokens
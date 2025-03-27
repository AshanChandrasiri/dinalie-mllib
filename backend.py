import traceback
import findspark

import matplotlib

matplotlib.use('Agg')

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import matplotlib.pyplot as plt
import io
import base64

import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from nltk.stem import PorterStemmer
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.classification import LogisticRegressionModel

from pyspark.ml.feature import Tokenizer
import pyspark
import py4j
import webbrowser
from threading import Timer

app = Flask(__name__)

CORS(app)

print("PySpark Version:", pyspark.__version__)
print("Py4J Version:", py4j.__version__)
# Dummy data for genre prediction model (For simplicity, you can replace this with your actual data)
data = {
    'lyrics': ["Country road"],
    'genre': ["pop", "country", "country", "rock", "jazz", "reggae", "hip hop"]
}
findspark.init()
spark = SparkSession.builder.appName("Music Genre Predictor Two").getOrCreate()


# spark = (SparkSession.builder
#          .appName("MusicClassification-app")
#          # .config("spark.sql.execution.arrow.pyspark.enabled", "true")
#          .getOrCreate())


# df = pd.DataFrame(data)

# # Vectorizer and models
# vectorizer = CountVectorizer()

# # Logistic Regression model
# logistic_model = make_pipeline(vectorizer, LogisticRegression())

# # Random Forest model
# rf_model = make_pipeline(vectorizer, RandomForestClassifier())

# # Train the models (use real training data for better results)
# logistic_model.fit(df['lyrics'], df['genre'])
# rf_model.fit(df['lyrics'], df['genre'])


def generate_chart(prediction_results, chart_type='bar'):
    """Generates and returns a chart as a base64 image.

    Args:
        prediction_results (dict): A dictionary with genre names as keys and probabilities as values.
        chart_type (str): The type of chart to generate ('bar' or 'pie').

    Returns:
        str: A base64 encoded string representing the chart image.
    """
    # Prepare data for plotting
    labels = list(prediction_results.keys())
    values = list(prediction_results.values())

    # Create figure
    fig, ax = plt.subplots()

    if chart_type == 'bar':
        ax.bar(labels, values, color='skyblue')
        ax.set_xlabel('Genres')
        ax.set_ylabel('Probabilities')
        ax.set_title('Genre Prediction Bar Chart')

    elif chart_type == 'pie':
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Genre Prediction Pie Chart')

    # Save it to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image as base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return img_base64


# -------------------------------------------------------------------------------

def load_mw2v_models():
    mw2v_model_path = "merged_word2vec_model"
    loaded_wtov_model = Word2VecModel.load(mw2v_model_path)
    return loaded_wtov_model


def load_lr_model():
    mlr_loaded_model_path = "merged_lr_model"
    loaded_lr_model = LogisticRegressionModel.load(mlr_loaded_model_path)
    return loaded_lr_model


def remove_punctuation(lyrics):
    if lyrics is not None:
        return lyrics.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    else:
        return lyrics


loaded_mlr_model = load_lr_model()
loaded_mw2v_model = load_mw2v_models()


def stem_words(tokens):
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def preprocess_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = remove_punctuation(lyrics)
    tokenizer = Tokenizer(inputCol="lyrics", outputCol="lyrics_tokens")
    df = spark.createDataFrame([(str(lyrics),)], ["lyrics"])
    df = tokenizer.transform(df)
    remover = StopWordsRemover(inputCol="lyrics_tokens", outputCol="lyrics_tokens_no_stopwords")
    df = remover.transform(df)

    stem_words_udf = F.udf(stem_words, ArrayType(StringType()))

    df = df.withColumn("lyrics_stemmed", stem_words_udf(df["lyrics_tokens_no_stopwords"]))
    stemmed_tokens = df.select("lyrics_stemmed").first()[0]
    return stemmed_tokens


def predict_genre_probabilities(lyrics):
    processed_lyrics = preprocess_lyrics(lyrics)

    # Transform the preprocessed lyrics into a Word2Vec vector
    new_df = spark.createDataFrame([(processed_lyrics,)], ["lyrics_stemmed"])
    new_df_with_vectors = loaded_mw2v_model.transform(new_df)
    # Make predictions using the loaded Logistic Regression model
    prediction = loaded_mlr_model.transform(new_df_with_vectors)

    # Get the predicted genre index
    probability_vector = prediction.select("probability").first()[0]
    probability_percentages = [float(prob) for prob in probability_vector]

    return probability_percentages


def predict_genre_probabilities_with_genre_names(lyrics):
    """Predicts genre probabilities and prints them along with genre names.

    Args:
        lyrics: The lyrics of a song as a string.
    """
    genre_probabilities = predict_genre_probabilities(lyrics)

    # Get the unique genre indices from the training data
    distinct_genres = {0: 'pop', 1: 'country', 2: 'blues', 3: 'rock', 4: 'jazz', 5: 'reggae', 6: 'hip hop', 7: 'soul'}

    # Find the index of the most probable genre
    most_probable_genre_index = genre_probabilities.index(max(genre_probabilities))

    # Use this index to fetch the genre name
    most_probable_genre = distinct_genres[most_probable_genre_index]

    # Construct prediction results as a dictionary of genre names and probabilities
    pred_results = {distinct_genres[i]: probability for i, probability in enumerate(genre_probabilities)}

    return most_probable_genre, pred_results


def start_app():
    webbrowser.open_new("http://127.0.0.1:5000")


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the user
    data = request.get_json()

    # Ensure we have the necessary keys in the request
    if 'lyrics' not in data or 'model' not in data:
        return jsonify({"error": "Missing 'lyrics' or 'model' in request data"}), 400

    lyrics = data['lyrics']
    model_choice = data['model']

    try:
        # Get prediction results
        most_probable_genre, pred_results = predict_genre_probabilities_with_genre_names(lyrics)

        # Generate both bar and pie chart images
        # bar_chart_base64 = generate_chart(pred_results, chart_type='bar')
        # pie_chart_base64 = generate_chart(pred_results, chart_type='pie')

        # Return the prediction results along with the generated charts
        return jsonify({
            "predicted_genre": most_probable_genre,
            "prediction_results": pred_results,
            # "bar_chart": bar_chart_base64,
            # "pie_chart": pie_chart_base64
        })

    except Exception as e:
        # Print the exception for debugging
        print("Exception occurred:", e)
        print(traceback.format_exc())

        # Return an error response
        return jsonify({"error": "An error occurred during prediction"}), 500


if __name__ == '__main__':
    # Timer(2, start_app).start()
    app.run(debug=True)

document.getElementById('predictButton').addEventListener('click', function() {
    const lyrics = document.getElementById('lyricsInput').value.trim();
    const selectedModel = document.getElementById('modelSelect').value;

    // Validate that the user has entered lyrics
    if (lyrics === "") {
        alert("Please enter some lyrics!");
        return;
    }

    // Prepare the data for the request
    const requestData = {
        lyrics: lyrics,
        model: selectedModel
    };

    // Make a POST request to the Flask API
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        // Display the predicted genre
        document.getElementById('predictedGenre').textContent = data.predicted_genre;
        document.getElementById('resultContainer').classList.add('show');
        document.getElementById('resultContainer').classList.remove('hidden');

        // Display the prediction probabilities
        let resultText = '';
        for (const genre in data.prediction_results) {
            resultText += `${genre}: ${data.prediction_results[genre]}<br>`;
        }
        document.getElementById('predictionDetails').innerHTML = resultText;

        // Display the bar chart
        document.getElementById('barChart').src = 'data:image/png;base64,' + data.bar_chart;

        // Display the pie chart
        document.getElementById('pieChart').src = 'data:image/png;base64,' + data.pie_chart;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    });
});

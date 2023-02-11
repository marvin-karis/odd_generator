import requests
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Prompt for input of two team names
homeTeam = input("Enter the name of first football team: ")
awayTeam = input("Enter the name of second football team: ")

# API endpoint to get head to head match results between two teams
url = f"https://api-football-v1.p.rapidapi.com/fixtures/h2h?homeTeam={homeTeam}&awayTeam={awayTeam}"

headers = {
    "X-RapidAPI-Key": "####YOUR_API_KEY######",
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# Make a request to the API
response = requests.request("GET", url, headers=headers)

# Parse the response data
data = response.json()

# Get the last 5 most recently head to head match results
final_score = data["headToHeadResults"][:5]

# Print the head to head results
for result in final_score:
    print(
        f"{result['homeTeam']} {result['homeTeamScore']} - {result['awayTeamScore']} {result['awayTeam']}")


# Create a dataset of match results
X = np.array([[result["homeTeamScore"], result["awayTeamScore"]]
             for result in final_score])
y = np.array([1 if result["homeTeamScore"] >
             result["awayTeamScore"] else 0 for result in final_score])

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Calculate the probability of homeTeam winning a match
prob_homeTeam_winning = clf.predict_proba([[1, 0]])[0][1]

# Calculate the probability of awayTeam winning a match
prob_awayTeam_winning = clf.predict_proba([[0, 1]])[0][0]

# Print the probabilities
print(f"Probability of {homeTeam} winning: {prob_homeTeam_winning:.2f}")
print(f"Probability of {awayTeam} winning: {prob_awayTeam_winning:.2f}")

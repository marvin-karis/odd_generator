import requests
import json
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Prompt for input of two team names
homeTeam = int(input("Enter the id of first football team:"))
awayTeam = int(input("Enter the id of second football team:"))

# API endpoint to get head to head match results between two team
url = "https://v3.football.api-sports.io/fixtures/headtohead"

querystring = {"h2h" : f"{homeTeam}-{awayTeam}", "last" : "3"}
querrystring = json.dumps(querystring, separators=('=' '-'))

headers = {
	"X-RapidAPI-Key": "############YOUR API KEY#############",
	"X-RapidAPI-Host": "v3.football.api-sports.io"
}

responses = requests.request("GET", url, params=querystring,  headers=headers )
# Parse the response data
data = responses.json()
#Get the last 5 most recently head to head match results
final_score = data["response"]
#print(final_score)
# Print the head to head results
for result in final_score:
    print(f"{result['teams']['home']['name']} {result['goals']['home']} - {result['goals']['away']} {result['teams']['away']['name']}")

# Create a dataset of match results
X = np.array([[result['goals']['home'], result['goals']['away']] for result in final_score])
y = np.array([1 if result['goals']['home'] > result['goals']['away'] else 0 for result in final_score])

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Calculate the probability of homeTeam winning a match
prob_homeTeam_winning = clf.predict_proba([[1, 0]])[0][1]

# Calculate the probability of awayTeam winning a match
prob_awayTeam_winning = clf.predict_proba([[0, 1]])[0][0]

# Print the probabilities
print(f"Probability of {0} winning: {prob_homeTeam_winning:.2f}")
print(f"Probability of {awayTeam} winning: {prob_awayTeam_winning:.2f}") 
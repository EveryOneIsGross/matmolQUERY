import openai
import numpy as np
import json
from nltk.sentiment import SentimentIntensityAnalyzer
import dotenv
import os

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return sia.polarity_scores(text)["compound"]

def generate_chat_model(prompt, temperature=0.5, max_tokens=50):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You answer questions. Do not generate lists. Respond only in statements."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=0.8,
        presence_penalty=0.2
    )
    return response.choices[0].message.content.strip()

def matmol_responses(prompt, size):
    responses = []
    for i in range(size):
        layer_responses = []
        for j in range(size):
            # Adjust the temperature based on the position in the matrix
            temperature = (i + j) / (2 * size)
            if i == 0 or i == size - 1 or j == 0 or j == size - 1:
                num_responses = 3
            else:
                num_responses = 5
            node_responses = []
            for k in range(num_responses):
                response = generate_chat_model(prompt, temperature=temperature, max_tokens=103)
                node_responses.append(response)
            layer_responses.append(node_responses)
        responses.append(layer_responses)
    return responses

def get_final_summary(summaries, responses):
    final_summary = ""
    for i, summary in enumerate(summaries):
        for j, row in enumerate(responses[i]):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = i + dx, j + dy
                if (
                    0 <= nx < len(responses)
                    and 0 <= ny < len(responses[i])
                    and summaries[nx][ny] == ""
                ):
                    sentiment = get_sentiment(responses[i][j])
                    if sentiment > 0:
                        summaries[nx][ny] = responses[i][j]  # Use the corresponding response
    for summary in summaries:
        final_summary += " ".join(summary) + " "
    return final_summary.strip()

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    prompt = input("Enter your prompt: ")
    size = int(input("Enter the size of the table: "))

    responses = matmol_responses(prompt, size)
    summaries = [["" for _ in range(size)] for _ in range(size)]

    for i in range(len(responses)):
        for j in range(len(responses[i])):
            if summaries[i][j] == "":
                summaries[i][j] = responses[i][j][0]

    final_summary = get_final_summary(summaries, responses)

    print("Summary:", final_summary)

    data = {
        'prompt': prompt,
        'responses': responses,
        'summaries': summaries,
        'final_summary': final_summary,
    }

    save_to_json(data, 'output.json')

if __name__ == "__main__":
    main()

# Multiplexed AI Text Generation with Distributed Sentiment Analysis

This project uses OpenAI api to generate a variety of responses to a given prompt, and then summarizes the responses based on sentiment analysis. The unique aspect of this project is the use of a populating matrix table to create a multi-layered, multiplexed response set.

![Absolute_Reality_v16_neo_from_the_matrix_represented_as_a_seri_1](https://github.com/EveryOneIsGross/matmolQUERY/assets/23621140/7ff51410-9fe1-4c9a-83d3-a277784e992c)

## Diverse Responses: 
By using different temperature settings for each cell in the table, this method ensures a wide range of responses from the AI model. Lower temperatures yield more predictable, focused responses, while higher temperatures yield more creative, diverse responses. This variability is beneficial when you want a mixture of both predictable and innovative responses.

##Sentiment Analysis:
This method does not just consider the literal content of each response, but also its sentiment. By calculating sentiment scores and using them to weight the responses, the final summary can more accurately reflect the overall sentiment of the responses. This is useful when the emotional tone of the responses is significant, such as in customer service, public relations, or social media management.

##Spread of Sentiment:
The sentiment of each response influences the weight of its neighboring cells in the table. This mechanism allows the sentiment of one cell to "spread" to its neighbors, leading to a more nuanced final summary that takes into account not only the content of the responses but also their sentiment.

## Weighted Summaries:
The summaries are not just a simple combination of the responses. Instead, they are weighted based on the sentiment scores of the responses. This weighting process ensures that responses with positive sentiment have a more significant impact on the final summary. This feature is beneficial when you want to focus on the positive aspects of a situation or when you are dealing with sensitive topics where the tone is critical.

## Use Cases:
This method could be particularly useful in several scenarios. For example, when dealing with large volumes of text data, like reviews or feedback, and you need a comprehensive summary that takes into account both the content and sentiment of the responses. It can also be used in interactive AI applications where you need the AI to respond to prompts with a certain emotional tone.

## Sentiment Propagation and Weighting
The sentiment scores are used to adjust the weights of the summaries in the table. Specifically, if a response has a positive sentiment score, it is considered valuable and contributes positively to the corresponding summary. This process of sentiment propagation helps spread valuable responses across the matrix, allowing the sentiment of one cell to influence the sentiment in its adjacent cells.

## Summary Generation
The final summary response is generated by combining all responses in the table, taking into account their adjusted weights. In this way, the summary response is a weighted combination of all the original responses, where the weights are influenced by the sentiment scores of each response and its neighbors.

## Code Components
The code is composed of several key functions:
```
generate_chat_model Function:
This function uses the OpenAI API to generate a response based on the input prompt.

matmol_responses Function:
Generates a matrix of responses by calling the generate_chat_model function for each cell in the matrix.

get_final_summary Function:
Constructs the final summary. It iterates over each cell in the matrix, checks the sentiment of the response, and if the sentiment is positive, the response is added to the summary.

save_to_json Function:
This function saves the data to a JSON file.

main Function:
This is the main entry point of the program. It prompts the user to enter a prompt and the size of the table.

The code checks if the script is being executed directly and calls the main function accordingly. The script generates multiple responses for each cell in the table, performs sentiment analysis on each response, adjusts the weights of the summaries based on the sentiment scores, and generates a final summary that incorporates the positive information extracted from the responses across the matrix.

```

In summary, this project presents a novel way of generating, weighting, and summarizing responses from an AI model. By considering both the content and sentiment of the responses and by propagating this sentiment across a table, the method ensures a final summary that is both comprehensive and nuanced.

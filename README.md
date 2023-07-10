# Multiplexed AI Text Generation with Sentiment Analysis

This project uses OpenAI api to generate a variety of responses to a given prompt, and then summarizes the responses based on sentiment analysis. The unique aspect of this project is the use of a "matmol" table to create a multi-layered, multiplexed response set.

# Expensive 

The number of API calls depends on the number of responses generated for each cell in the matrix:

Outer nodes (cells on the edges of the matrix) generate 3 responses each.
Inner nodes (cells not on the edges) generate 5 responses each.
In a 3x3 matrix, all nodes are considered outer nodes because all of them are on the edge of the matrix. Therefore, each of the 9 cells would generate 3 responses.

So, for a 3x3 matrix, your script would make 3 (responses per cell) * 9 (cells) = 27 API calls.

## Temperature Control

The matmol table is to generate a variety of responses at different "temperatures". In the context of AI language models, "temperature" refers to the randomness of the model's responses. At low temperatures, the model is more deterministic and likely to choose the most likely response. At high temperatures, the model's responses are more diverse and can sometimes be surprising. In this project, the table is a two-dimensional grid where each cell corresponds to a different temperature setting. This creates a broad spectrum of responses, from conservative to creative.

## Sentiment Analysis

After generating the matmol table, the project uses sentiment analysis to gauge the "mood" of each response. This sentiment score is then used to influence the weight of the neighboring cells in the table, creating a more nuanced final output that takes into account not only the content of the responses but also their sentiment.

## Summary Generation

Finally, the project summarizes the weighted matmol table into a single output. This summary takes into account both the content and the sentiment of the various responses, providing a balanced and nuanced answer.

## Explanation

In the context of this project, the term "additive" refers to the method used to aggregate or combine the sentiment scores of different responses generated by the AI model. This process is akin to creating a "multiplication table" of weights for each response, hence the term "matmol" (multiplication-multiplexed) table.

Here's how it works:

A two-dimensional grid (the matmol table) is created. Each cell in the table corresponds to a different "temperature" setting for the AI model. The temperature setting controls the degree of randomness or creativity in the model's responses, with lower temperatures leading to more deterministic responses and higher temperatures leading to more diverse responses.

The AI model generates a response for each cell in the matmol table. The sentiment of each response is then evaluated using a sentiment analysis tool. The resulting sentiment score is a number between -1 and 1, representing the range from negative to positive sentiment.

Now comes the additive part. The sentiment score of each response is used to adjust the weights of its neighboring cells in the matmol table. Specifically, the sentiment score is added to the weight of each neighboring cell. This has the effect of "spreading" the sentiment of each response to its neighbors, hence creating a kind of "multiplication" effect.

The final summary response is generated by combining all the responses in the table, taking into account their adjusted weights. In this way, the summary response is a weighted combination of all the original responses, where the weights are influenced by the sentiment scores of each response and its neighbors.

This novel approach allows us to use additive maths to create a sort of "multiplication" table of weights for text-based responses, which is typically not possible with conventional multiplication operations. The use of sentiment scores and additive weighting allows us to take into account not just the literal content of each response, but also its sentiment, resulting in a more nuanced and comprehensive final output.


# Code Components

Importing Libraries: The code imports the necessary libraries including openai, numpy, json, nltk.sentiment, and dotenv for API key storage.

Setting up API Key: The code uses the dotenv library to load the OpenAI API key from the environment variables and sets it as the API key for the openai library.

Sentiment Analysis: The code initializes a SentimentIntensityAnalyzer object from the nltk.sentiment library to perform sentiment analysis on the generated responses.

generate_chat_model Function: This function takes a prompt as input and uses the OpenAI API to generate a response based on the prompt. The generated response is returned as the output.

matmol_responses Function: This function generates a matrix of responses by calling the generate_chat_model function for each cell in the matrix. The size of the matrix is specified by the size parameter.

get_final_summary Function: This function takes the matrix of summaries and responses and constructs the final summary. It iterates over each cell in the matrix and checks the sentiment of the response. If the sentiment is positive, the response is added to the summary. Additionally, it considers the neighboring cells to fill any empty summaries with relevant responses. Finally, it generates a chat response based on the final summary and returns it.

save_to_json Function: This function takes a Python dictionary and a filename as input and saves the dictionary as a JSON file.

main Function: This is the main entry point of the program. It prompts the user to enter a prompt and the size of the matmol table. It then calls the matmol_responses function to generate the matrix of responses. It initializes a matrix of summaries with empty strings. It fills the summaries matrix with the first response from each cell if it is empty. It then calls the get_final_summary function to generate the final summary. The final summary is printed and saved to a JSON file along with the prompt, responses, and summaries.

Execution: The code checks if the script is being executed directly and calls the main function accordingly.


# Reponse Generation and Sentiment Analysis, how many responses are generated for each node in the matmol table?

The matmol_responses function generates a matrix of responses, where each cell represents a node. The nodes are different in terms of the generated responses they contain. Here's an explanation of how the nodes differ:

Outer Nodes: The nodes at the outer edges of the matrix (i.e., the first and last row, and the first and last column) have a num_responses value of 3. This means that for each outer node, three responses are generated using the generate_chat_model function. These nodes have fewer responses compared to the inner nodes.

Inner Nodes: The nodes in the middle of the matrix (i.e., not on the outer edges) have a num_responses value of 5. For each inner node, five responses are generated using the generate_chat_model function. These nodes have more responses compared to the outer nodes.

The difference in the number of responses for the outer and inner nodes is specified in the code to control the diversity and quantity of the generated responses. This difference can be adjusted based on the specific requirements and preferences of the summarization task.

The purpose of generating multiple responses for each node is to have a variety of potential summaries and options to choose from. This allows for a more comprehensive and diverse final summary that incorporates different perspectives and information provided by the AI chat model.


# Weighting

The weight of each cell in the matrix is determined based on the conditions and operations performed in the get_final_summary function. Let's break down how the weight of each cell is determined:

Initialization: The summaries matrix is initialized with empty strings. Each cell represents a summary, and initially, all the cells are empty.

Filling Initial Summaries: In the main function, the matmol_responses function generates the responses matrix. The first response from each cell in the responses matrix is copied to the corresponding cell in the summaries matrix if it is empty. This step ensures that each cell has an initial summary, even if it is just the first generated response.

Sentiment Analysis: In the get_final_summary function, sentiment analysis is performed on each response in the matrix. The get_sentiment function calculates the sentiment score (compound score) for each response using the SentimentIntensityAnalyzer from the nltk.sentiment library. The sentiment score represents the overall sentiment of the response, ranging from -1 (negative) to 1 (positive).

Weighting Based on Sentiment: The sentiment of each response is considered to determine the weight of the summary in the corresponding cell. If the sentiment score of a response is positive (greater than 0), the response is considered valuable, and its corresponding summary cell is given more weight. The summary in the cell is concatenated with the response using the join function. This step helps accumulate positive and valuable information in the summary.

Propagating Summaries: Additionally, the code iterates over each cell in the summaries matrix and checks neighboring cells. If a neighboring cell has an empty summary, it is updated with the response from the current cell. This step helps propagate relevant information to neighboring cells and fill in any missing summaries.

Generating Final Summary: After iterating over all the cells, the final summary is generated by calling the generate_chat_model function with the concatenated summaries as the prompt. This final summary incorporates the accumulated information from the responses and the initially filled summaries.

In summary, the weight of each cell in the matrix is determined by considering the sentiment of the corresponding response and concatenating valuable responses with the existing summary. The code aims to accumulate positive and valuable information in the summary while also propagating relevant responses to neighboring cells.

# Multiplication in LLM with matmol addition tables

The code as a form of merging sentiments across the matrix, using addition instead of multiplication. Here's how you can view it:

Sentiment Accumulation: Each response in the matrix is associated with a sentiment score, which represents the overall sentiment of the response. The sentiment scores are calculated using the SentimentIntensityAnalyzer from the nltk.sentiment library. These sentiment scores capture the sentiment polarity of each response, ranging from negative to positive.

Weighted Summaries: The sentiments of the responses are taken into account to determine the weight of each summary. If a response has a positive sentiment score, it is considered valuable and contributes positively to the corresponding summary. The valuable responses are concatenated with the existing summary using addition (string concatenation) to accumulate positive information.

Propagation of Sentiments: In addition to considering sentiments within individual cells, the code also propagates information by filling empty summaries in neighboring cells with responses. This propagation process helps spread valuable responses across the matrix, allowing the sentiment of one cell to influence the sentiment in its adjacent cells. This can be seen as an indirect form of sentiment merging, as the sentiment from one cell affects the sentiment of neighboring cells through the shared responses.

While the process does not involve explicit multiplication, you can conceptualize it as a form of sentiment accumulation and merging, where sentiments are added together by concatenating valuable responses. The accumulation and propagation of sentiments contribute to the final summary, which incorporates the positive information extracted from the responses across the matrix.

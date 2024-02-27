# Text Processing Model/Program

## Introduction

This model/program is designed to handle various text processing tasks efficiently.

## Methodology

We employ Lemmatization to extract the root forms of words in sentences and utilize a stop word list for filtering.

## Part 1: Perplexity

Perplexity is used as a metric to evaluate the language model's performance. A lower perplexity indicates better model performance.

- Without preprocessing, sentences containing conjunctions tend to have lower perplexity.
- After removing stop words and applying lemmatization, perplexity may slightly increase, but overall, the difference is minimal.

## Part 2: Pre-training Steps

1. **Masked Token Prediction**: Randomly mask tokens in the input and predict the masked tokens.
2. **Next Sentence Prediction**: Predict whether two sentences are consecutive.

## Part 3: Model Architecture

- **Embedding Layers**: Converts text into vector representations.
- **LSTM Layers**: Utilizes Long Short-Term Memory networks for processing sequential data.
- **Linear Layer**: Generates final prediction results.

## 4 Bert Application Scenarios

1. **Sentiment Analysis**: Classify sentences into positive or negative sentiment.
2. **POS Tagging**: Predict the part of speech for each word in a sentence.
3. **Natural Language Inference (NLI)**: Determine if a hypothesis can be inferred from a premise.

## Differences Between Bert and DistilBERT

DistilBERT reduces model size by 40% through knowledge distillation while retaining language understanding abilities. It is faster and lighter compared to Bert.

## Troubleshooting

- If encountering `RuntimeError: Expected all tensors to be on the same device...`, ensure test inputs are on the same device as the model by using `test_input.to('cuda')`.

## Conclusion

In conclusion, this model/program showcases effective text processing capabilities through its architecture and preprocessing techniques.




Session 2:
Humans are not consistent, we may change over time. 
You obviously cannot  do hyper parameter tuning for foundation model.
GPT model performance is different if the language changes English vs Spanish for example
MMLU: Massive Task : dataset assessing model for various tasks. 
We can use transaltion for languages since English is dominant, but then we have moe points of filure. 
Transformeer architecture reduces the cost of generrating text  for NLP. 
For attention the weights relevancy for sequences is learnt
we get these relevancy weights from previous training? Deciding which part has more weight for that particular token?

You basically do a dot product of input query with the embedded vector (you trained on) and then sample across multiple heads with respect to their probability. 
The softmax function itself can be used in different contexts - but in the context of LLMs and token prediction, it helps building a probability distribution over the vocabulary (and the vocabulary is basically a set of categorical outputs) to predict what is the most probable next token
It’s totally possible for the temperature to be > 1 - it just means there is gonna be more randomness in the outputs
in practical, does temp. = 0 give us a fully deterministic output? I assume not. -> Very good question 🙂 mathematically… yes it’s supposed to be fully deterministic 
However, if you run the same thing on two different machines, you might face different outputs bc of low-level computation differences (rounding, floating-point precision... in the GPU calculations w/ matrices)

how Retrieval-Augmented Generation (RAG) changes the probability of a model? Usually, RAG methods feed additional context to the model, which can change the probability distribution of the output
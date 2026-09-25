# Transformers and Attention

## From recurrent networks to attention

Recurrent neural networks process a sequence one token at a time, which makes training hard to parallelize and makes it difficult to carry information across long distances. The transformer architecture, introduced in the 2017 paper "Attention Is All You Need", replaced recurrence entirely with attention, allowing every position in a sequence to be processed in parallel.

## Scaled dot-product attention

Each token is projected into three vectors: a query, a key, and a value. The attention weights for a token are computed by taking the dot product of its query with every key, dividing by the square root of the key dimension, and applying a softmax. The output is the weighted sum of the value vectors. The scaling by the square root of the dimension keeps the dot products from growing too large, which would push the softmax into regions with very small gradients. Because every token attends to every other token, the cost of attention grows quadratically with sequence length.

## Multi-head attention and positional information

Multi-head attention runs several attention operations in parallel, each with its own learned projections, so that different heads can focus on different kinds of relationships, such as syntax or coreference. Attention by itself has no notion of word order, so transformers add positional information to the token embeddings, either with fixed sinusoidal encodings, learned position embeddings, or rotary position embeddings (RoPE) that rotate queries and keys by an angle that depends on position.

## Encoders, decoders and model families

Encoder-only models such as BERT use bidirectional attention and are pre-trained with masked language modelling, where random tokens are hidden and the model predicts them. They are well suited to classification and producing embeddings. Decoder-only models such as the GPT family use causal masking so that each token can only attend to earlier tokens, and they are trained to predict the next token, which makes them natural text generators. Encoder-decoder models such as T5 are used for sequence-to-sequence tasks like translation.

## Fine-tuning

Pre-trained transformers are adapted to new tasks by fine-tuning. Full fine-tuning updates all weights, which is expensive for large models. Parameter-efficient methods such as LoRA freeze the original weights and learn small low-rank matrices that are added to selected layers, reducing the number of trainable parameters by orders of magnitude while retaining most of the quality.

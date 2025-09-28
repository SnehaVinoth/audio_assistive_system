from nltk.translate.bleu_score import sentence_bleu

# Reference (correct) caption
reference = "a black horse running through a grassy field"
# Candidate captions
candidate_transformer = "a black horse running through a grassy field"
candidate_cnnrnn = "in in in in in in in in way in way in way in"

# Tokenize captions
reference_tokens = reference.split()
candidate_transformer_tokens = candidate_transformer.split()
candidate_cnnrnn_tokens = candidate_cnnrnn.split()

# Compute BLEU scores
bleu_transformer = sentence_bleu([reference_tokens], candidate_transformer_tokens)
bleu_cnnrnn = sentence_bleu([reference_tokens], candidate_cnnrnn_tokens)

print("BLEU score - Transformer:", bleu_transformer)
print("BLEU score - CNN-RNN:", bleu_cnnrnn)

# import nltk
# from nltk.metrics import edit_distance
# nltk.download('punkt_tab')

# print(nltk.tokenize.word_tokenize("Hello World"))
# print(edit_distance("Mork", "Mindy"))

# def camel_case_split(s):
#     u = True  # case of previous char
#     w = b = ''  # current word, buffer for last uppercase letter
#     for c in s:
#         o = c.isupper()
#         if u and o:
#             w += b
#             b = c
#         elif u and not o:
#             if len(w)>0:
#                 yield w
#             w = b + c
#             b = ''
#         elif not u and o:
#             yield w
#             w = ''
#             b = c
#         else:  # not u and not o:
#             w += c
#         u = o
#     if len(w)>0 or len(b)>0:  # flush
#         yield w + b

def camel_case_split(s):
    u = True  # case of previous char
    w = b = ''  # current word, buffer for last uppercase letter
    for c in s:
        if c == '_':
            if len(w) > 0 or len(b) > 0:
                yield w + b
            w = b = ''
            continue
        o = c.isupper()
        if u and o:
            w += b
            b = c
        elif u and not o:
            if len(w) > 0:
                yield w
            w = b + c
            b = ''
        elif not u and o:
            yield w
            w = ''
            b = c
        else:  # not u and not o:
            w += c
        u = o
    if len(w) > 0 or len(b) > 0:  # flush
        yield w + b



tokens = list(camel_case_split("KingQueenSchoolZebraKingMagazine_Princess_prize"))

from gensim.models import Word2Vec

# Create a Word2Vec model and train it on the tokens list.
model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)

token_encodings = {token: model.wv[token] for token in tokens}

# for token, encoding in token_encodings.items():
#     print(f"Token: {token}, Encoding: {encoding}")

from scipy.spatial.distance import cosine
from itertools import combinations

# Generate all pairs from the list tokens
token_pairs = list(combinations(tokens, 2))
# print("All pairs from tokens list:", token_pairs)
# for token1, token2 in token_pairs:
#     distance = euclidean(token_encodings[token1], token_encodings[token2])
#     print(f"Distance between embeddings of '{token1}' and '{token2}': {distance}")

# Calculate distances and pair them with the token pairs
token_distances = [(token1, token2, cosine(token_encodings[token1], token_encodings[token2])) for token1, token2 in
                   token_pairs]

# Sort the token pairs based on the distances
token_distances.sort(key=lambda x: x[2])

# Print the sorted token pairs with their distances
for token1, token2, distance in token_distances:
    print(f"Distance between embeddings of '{token1}' and '{token2}': {distance}")

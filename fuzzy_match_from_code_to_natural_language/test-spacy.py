import spacy
from itertools import product
nlp = spacy.load("en_core_web_lg")
# nlp = spacy.load("en_core_web_sm")nlp
print(nlp._path)


terms = [
u'policy number',
u'pol number',
u'form number',
u'annual premium',
u'earned premium',
u'guideline annual premium'
]

term_nlp_tuples = [(term, nlp(term)) for term in terms]


pairs = product(term_nlp_tuples, repeat=2)

similarity_results = []

for (term1, doc1), (term2, doc2) in pairs:
    similarity = (1 - doc1.similarity(doc2)) * 10
    similarity_results.append(((term1, term2), similarity))

# Sort the results by similarity score in descending order
similarity_results.sort(key=lambda x: x[1], reverse=False)

# Print the sorted results
max_term_length = max(len(term) for term, _ in term_nlp_tuples)
header = f"{'Term1'.ljust(max_term_length)}    {'Term2'.ljust(max_term_length)}    Similarity"
print(header)
print('-' * len(header))

# Print the sorted results in a table format
for (term1, term2), similarity in similarity_results:
    print(f"{term1.ljust(max_term_length)}    {term2.ljust(max_term_length)}    {similarity:.4f}")

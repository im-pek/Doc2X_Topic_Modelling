from gensim.models.doc2vec import TaggedDocument, Doc2Vec 
import numpy as np
import pandas as pd
from helper import *
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
import string

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 

def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    return two

df = pd.read_csv(open("abstracts for 'automat'.csv", errors='ignore'))
df=df.astype(str)

text = df.applymap(cleaning)['paperAbstract']
text_list = [i.split() for i in text]

all_joined=[]

for element in text_list:
    joined=' '.join(element)
    all_joined.append(joined)
    
tagss=np.arange(len(text_list))

import multiprocessing ########
cores = multiprocessing.cpu_count() ########

documents = [TaggedDocument(all_joined, tagss)]
model = Doc2Vec(documents, min_count=1, workers=cores, iter=3, dm=0, hs=1, negative=2, minn=1) #vector_size=100, window=8 #my current computer has 8 cores
#min_count: ignores all words with total frequency lower than this
#dm=0 is dbow, dm=1 is dm
#hs=1 employs hierarchical softmax
#negative > 0 employs negative sampling. 2-5 for large datasets, 5-20 for small datasets

entire_corpus=' '.join(all_joined)
entire_corpus_strings=entire_corpus.split(' ')

#model.build_vocab(entire_corpus_strings, update=True) #!
#model.train(entire_corpus_strings, total_examples=model.corpus_count, epochs=model.iter) #!

# Then you can infer new vector and compute most similar documents:
vector = model.infer_vector(entire_corpus_strings) #one list of strings, each of their MIs computed
#inferred_vector = model.infer_vector(doc_words=words, alpha=0.025, min_alpha=0.0001, steps=150)

top_MI_values=model.docvecs.most_similar(positive=[vector], topn=30)

top_MI_values_list=[]

for tuplee in top_MI_values:
    listed=list(tuplee)
    top_MI_values_list.append(listed)
    
for item in top_MI_values_list:
    test=all_joined[int(item[0])]
    item[0]=test

#print (top_MI_values_list) #shows MIs too
#print ('\n')

top_docs=[]
for ix,thingy in enumerate(top_MI_values_list):
    top_docs.append(top_MI_values_list[ix][0])

print ('Doc2Vec (Document-Based) Topics:')
print (top_docs)
print ('\n')

allll=[]
for element in top_docs:
    within=element.split(' ')
    allll.append(within)



from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct

vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df=0.01)
vector=vectorizer.fit_transform(allll).todense()

vocab=vectorizer.vocabulary_ 

numpy_array=np.array(vector, dtype=int)

# Sparse matrices are also supported
X = ss.csr_matrix(numpy_array)

# WORD LABELS for each column can be provided to the model
all_vocabs=list(vocab.keys())

# DOCUMENT LABELS for each row can be provided
topics = np.arange(1)

seed = 1 #CHANGE THE SEED HERE

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=10, seed=seed)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=all_vocabs, docs=topics)

topics = topic_model.get_topics()

print (topics) #shows PMIs too
print ('\n')

print ('Corex Topics:')
for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n+1)+': '+','.join(words)
    print(topic_str)
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from joint_log_lik import joint_log_lik
from sample_topic_assignment import sample_topic_assignment


bagofwords = loadmat('./nips_abstract/bagofwords_nips.mat')
WS = bagofwords['WS'][0] - 1  #go to 0 indexed
DS = bagofwords['DS'][0] - 1

WO = loadmat('./nips_abstract/words_nips.mat')['WO'][:,0]
titles = loadmat('./nips_abstract/titles_nips.mat')['titles'][:,0]



#This script outlines how you might create a MCMC sampler for the LDA model

alphabet_size = WO.size

document_assignment = DS
words = WS

#subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
#PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
words = words[document_assignment < 100]
document_assignment  = document_assignment[document_assignment < 100]

n_docs = document_assignment.max() + 1

#number of topics
n_topics = 20

#initial topic assigments
topic_assignment = np.random.randint(n_topics, size=document_assignment.size)

#within document count of topics
doc_counts = np.zeros((n_docs,n_topics))

for d in range(n_docs):
    #histogram counts the number of occurences in a certain defined bin
    doc_counts[d] = np.histogram(topic_assignment[document_assignment == d], bins=n_topics, range=(-0.5,n_topics-0.5))[0]

#doc_N: array of size n_docs count of total words in each document, minus 1
doc_N = doc_counts.sum(axis=1) - 1

#within topic count of words
topic_counts = np.zeros((n_topics,alphabet_size))

for k in range(n_topics):
    w_k = words[topic_assignment == k]

    topic_counts[k] = np.histogram(w_k, bins=alphabet_size, range=(-0.5,alphabet_size-0.5))[0]



#topic_N: array of size n_topics count of total words assigned to each topic
topic_N = topic_counts.sum(axis=1)

#prior parameters, alpha parameterizes the dirichlet to regularize the
#document specific distributions over topics and gamma parameterizes the 
#dirichlet to regularize the topic specific distributions over words.
#These parameters are both scalars and really we use alpha * ones() to
#parameterize each dirichlet distribution. Iters will set the number of
#times your sampler will iterate.
alpha = 5
gamma = 1.5 
iters = 10


jll = []
for i in tqdm(range(iters)):
    jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))
    
    prm = np.random.permutation(words.shape[0])
    
    words = words[prm]   
    document_assignment = document_assignment[prm]
    topic_assignment = topic_assignment[prm]
    
    topic_assignment, topic_counts, doc_counts, topic_N = sample_topic_assignment(
                                topic_assignment,
                                topic_counts,
                                doc_counts,
                                topic_N,
                                doc_N,
                                alpha,
                                gamma,
                                words,
                                document_assignment)
                        
jll.append(joint_log_lik(doc_counts,topic_counts,alpha,gamma))

print(jll)
plt.plot(jll)


### find the 10 most probable words of the 20 topics:
#TODO:
topic_counts_norm = topic_counts / np.expand_dims(topic_counts.sum(axis=1), 1)
top_10_idx = np.argsort(topic_counts_norm, axis=1)[:, -1:-11:-1]

fstr = ''
for k in range(topic_N.shape[0]):
    fstr += 'topic {}: '.format(k)
    for w in top_10_idx[k]:
        fstr += '{}, '.format(WO[w][0])
    fstr += '\n'

with open('most_probable_words_per_topic','w') as f:
    f.write(fstr)
    
    
    
#most similar documents to document 0 by cosine similarity over topic distribution:
#normalize topics per document and dot product:
#TODO:
d0 = doc_counts[0].reshape(1, -1)
sim = np.matmul(d0, doc_counts.T) / (np.linalg.norm(d0) * np.linalg.norm(doc_counts, axis=1))
top_10_idx = np.argsort(sim)[0][-1:-12:-1]
fstr = 'Titles most similar to {} are:\n'.format(titles[0][0])
for i, idx in enumerate(top_10_idx):
    fstr += '{}. {}\n'.format(i, titles[idx][0])

with open('most_similar_titles_to_0','w') as f:
    f.write(fstr)

    


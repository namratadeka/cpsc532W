import numpy as np

def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.
    
    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words        
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1
        
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of wors
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    #TODO
    for d in range(doc_counts.shape[0]):
        z_d = topic_assignment[document_assignment==d]
        words_d = words[document_assignment==d]
        for n in range(int(doc_N[d])):
            # get current topic of nth word in dth doc.
            z_d_n = z_d[n]

            # decrement counts for topic z_d_n.
            doc_counts[d][z_d_n] -= 1
            topic_counts[z_d_n][words_d[n]] -= 1
            topic_N[z_d_n] -= 1

            # sample a new topic for nth word in dth doc.
            p_T_D = (alpha + doc_counts[d]) / np.sum(alpha + doc_counts[d])
            p_W_T = (gamma + topic_counts[:, words_d[n]]) / (np.sum(topic_counts, axis=1) + gamma)
            p_z = p_W_T * p_T_D
            p_z /= p_z.sum()
            z = np.random.choice(topic_N.size, p=p_z)

            # set new assignment.
            doc_counts[d][z] += 1
            topic_counts[z][words_d[n]] += 1
            topic_N[z] += 1
            z_d[n] = z

        topic_assignment[document_assignment==d] = z_d
        
    return topic_assignment, topic_counts, doc_counts, topic_N
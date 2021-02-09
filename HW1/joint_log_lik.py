import numpy as np
from scipy.special import gamma as gamma_fn

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model
    
    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        ll: the joint log likelihood of the model
    """
    #TODO

    jll = 0
    for k in range(topic_counts.shape[0]):
        jll += np.sum(np.log(gamma_fn(topic_counts[k, :] + gamma))) - np.log(gamma_fn(np.sum(topic_counts[k, :] + gamma)))
        jll -= topic_counts.shape[1] * np.log(gamma_fn(gamma)) - np.log(gamma_fn(topic_counts.shape[1] * gamma))

    for d in range(doc_counts.shape[0]):
        jll += np.sum(np.log(gamma_fn(doc_counts[d, :] + alpha))) - np.log(gamma_fn(np.sum(doc_counts[d, :] + alpha)))
        jll -= doc_counts.shape[1] * np.log(gamma_fn(alpha)) - np.log(gamma_fn(doc_counts.shape[1] * alpha))

    return jll

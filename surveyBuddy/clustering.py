from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import csv
from tqdm import tqdm
import numpy as np


def readFile():
    listCorpus = []
    listSize= []
    with open('Citi_Unsupervised_yake.csv', 'r') as read_file:
        line_count = 0
        csv_reader = csv.reader(read_file, delimiter='\t')
        for row in tqdm(csv_reader):
            if line_count != 0:
                if(len(row)!=0):
                    listWords = row[0].split("/")
                    listCorpus += listWords
                    listSize.append(len(listWords))
            line_count += 1
    return listCorpus, listSize


def main():
    listCorpus, listSize = readFile()
    embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    # Corpus with example sentences
    corpus = ['A man is eating food.',
              'A man is eating a piece of bread.',
              'A man is eating pasta.',
              'The girl is carrying a baby.',
              'The baby is carried by the woman',
              'A man is riding a horse.',
              'A man is riding a white horse on an enclosed ground.',
              'A monkey is playing drums.',
              'Someone in a gorilla costume is playing a set of drums.',
              'A cheetah is running behind its prey.',
              'A cheetah chases prey on across a field.']
    # listCorpus=corpus
    # listSize=[1,1,1,1,1,1,1,1,1,1,1]
    corpus_embeddings = embedder.encode(listCorpus)

    num_clusters = 150
    clustering_model = KMeans(n_clusters=num_clusters)
    cluster_dist = clustering_model.fit_transform(corpus_embeddings)
    cluster_dist = cluster_dist.min(1)
    cluster_assignment = clustering_model.labels_
    final_assignment = -1 * np.ones(len(listCorpus))
    keywords_list=[]
    for i in range(0,num_clusters):
        theta = (cluster_dist * (cluster_assignment == i))
        if len(np.nonzero(theta)[0]) == 0:
            continue

        idx = np.where(theta == np.min(theta[np.nonzero(theta)]))
        final_assignment[cluster_assignment==i] = idx[0][0]
        keywords_list.append(listCorpus[idx[0][0]])

    final_assignment1 = [listCorpus[int(i)] for i in final_assignment]

    start = 0
    line_no = 0
    # with open('citi_file_cluster.csv', 'w', newline='') as write_file:
    #     for i in range(0, len(listSize)):
    #         writer = csv.writer(write_file)
    #         col1 = "/".join(listCorpus[start:start + listSize[line_no]])
    #         col2 = "/".join(final_assignment1[start:start + listSize[line_no]])
    #         col3 = "/".join(list(set(final_assignment1[start:start + listSize[line_no]])))
    #         start += listSize[line_no]
    #         writer.writerow([col1, col2, col3])
    #         line_no += 1
    #
    with open('keywords_list.csv', 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(keywords_list)


if __name__ == '__main__':
    main()
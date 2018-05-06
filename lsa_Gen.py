import re, sys
import numpy as np
from collections import Counter
import pandas as pd

word_pattern = re.compile("\w[\w\-\']*\w|\w")

novel_metadata = {} # <- dict of dicts
novel_counts = {} # <- dict of Counters
word_counts = Counter()

## Open the metadata file and read the header line
metadatadf = pd.read_csv('data/metadata.csv', skipinitialspace=True)
reviewsdf = pd.read_csv('data/reviews.csv', skipinitialspace=True)
reviews = reviewsdf.review
reviews = [str(i).split() for i in reviews]
## Read the rest of the lines of the file
for index, row in metadatadf.iterrows():
    
     ## Save it for later with the filename as key
    filename = row.doc_id
    novel_metadata[filename] = row

    ## Now count the words in the novel
    counter = Counter()
    counter.update(reviews[filename])
    
    ## And save those counts for later
    novel_counts[filename] = counter
    
    ## Record the total number of times each word occurs
    word_counts.update(counter)

## All the distinct word types in descending order by frequency
vocabulary = [x[0] for x in word_counts.most_common()]

vocabulary = vocabulary[0:10000]
reverse_vocabulary = {}
for word_id, word in enumerate(vocabulary):
    reverse_vocabulary[word] = word_id

## All the filenames
filenames = list(novel_counts.keys())
titles = [novel_metadata[id]["album"] for id in filenames]

## Allocate a matrix with two rows for every file and one column for every word type
file_word_counts = np.zeros([ len(filenames), len(vocabulary) ])

## Convert a map of file-level counters to a single matrix
## We'll use two index variables, file_id and word_id. These will be 
##  *numbers*, not strings, that point to a string in either of
##  the two arrays.
for file_id in range(len(filenames)):
    counter = novel_counts[ filenames[file_id] ]
    
    for word_id in range(len(vocabulary)):
        file_word_counts[file_id,word_id] = counter[ vocabulary[word_id] ]
    
    ## Normalize for length
    file_word_counts[file_id,:] /= np.sum(file_word_counts[file_id,:])

## Run the singular value decomposition
(file_vectors, weights, word_vectors) = np.linalg.svd(file_word_counts, full_matrices=False)

## transpose word vectors
word_vectors = word_vectors.T

## Absorb the factor weights into the matrices
weighted_word_vectors = word_vectors.dot( np.diag(np.sqrt(weights)) )
weighted_file_vectors = file_vectors.dot( np.diag(np.sqrt(weights)) )

## Write data to files, which we can load with R

with open("file_vectors.tsv", "w") as out:
    for i in range(len(file_vectors[0,:])):
        out.write("V{}\t".format(i))
    out.write("Title\n")
    
    for file_id in range(len(filenames)):
        for i in range(len(file_vectors[file_id,:])):
            out.write("{:.6f}\t".format(file_vectors[file_id,i]))

        out.write("{}\n".format(titles[file_id]))
with open("word_vectors.tsv", "w") as out:
    for i in range(len(word_vectors[0,:])):
        out.write("V{}\t".format(i))
    out.write("Word\n")
    
    for word_id in range(len(vocabulary)):
        for i in range(len(word_vectors[word_id,:])):
            out.write("{:.6f}\t".format(word_vectors[word_id,i]))
        out.write("{}\n".format(vocabulary[word_id]))


## Pass in a vector and a list of strings that define a meaning for each
##  element in the vector. Sort the elements by value and print the top
##  and bottom strings.
def sort_vector(v, names):
    sorted_list = sorted(list(zip(v, names)))
    for pair in sorted_list[:10]:
        print(pair)
    for pair in sorted_list[-10:]:
        print(pair)



from gensim import *
from gensim.models import *
import re
import unicodedata
import os
import codecs
import matplotlib.pyplot as plt
import gensim
import numpy as np
index = similarities.MatrixSimilarity.load('models/similarity.index')

dictionary = corpora.Dictionary.load('models/orig_text.dict')
corpus = corpora.MmCorpus('models/orig_text.mm')
lsi = LsiModel.load('models/twodlsa.lsi')
lda = LdaModel.load('models/lda/lda.lda')
totalvec_lda = [(0, 0.032843836), (1, 0.04208189), (2, 0.10474158), (5, 0.042373378), (6, 0.013250324), (9, 0.036808446), (11, 0.023320341), (12, 0.021665176), (14, 0.10429024), (15, 0.030837856), (16, 0.017426088), (18, 0.018971419), (20, 0.049614906), (21, 0.018747063), (22, 0.11087632), (26, 0.14842194), (28, 0.010651322), (33, 0.032480825), (35, 0.07782937)]
pdf = {'a': 0.43293275377141394,
 'c': 0.16456149322423932,
 'b': 0.2904116594221427,
 'd': 0.09920736384556379,
 'f': 0.012886729736640245}

with open("stopwords.txt") as f:
	stopwords = [word for line in f for word in line.split()]
def remove_stop(m):
	return '' if m.group() in stopwords else m.group()
def clean_text(text):
	temp = text
	temp=re.sub('\S+[\'\’\‘]\S+','',(temp))
	temp=re.sub('\w*\d\S*','',(temp))
	temp=re.sub('(?<!^|$)(?<!([(\.)(\!)(\?)(\“)]\s))([A-Z]\S+)','',(temp))
	temp=re.sub('[^A-z\s\-\–\&]',' ',(temp))
	temp=re.sub('\su\ss\s',' U.S. ',(temp))
	temp=re.sub('\[#.+]\|+','',(temp))
	temp = temp.lower()
	return temp

REVIEW = '''During his time working at the Trump Organization, Mr. Cohen became a minor real estate baron in his own right.

While his business is dwarfed by Mr. Trump’s assemblage of properties, it is not insignificant. From 2011 to 2015, limited liability companies connected to Mr. Cohen purchased at least five buildings in Manhattan, public records show.

Like many of Mr. Cohen’s business dealings, the transactions were unconventional. His companies would buy a building, often in cash. Soon after, they would flip the building in another all-cash deal for four or five times the previous purchase price. The buyer was generally another limited liability company.

In October 2011, for example, a limited liability company listing as its address Mr. Cohen’s apartment at Trump Park Avenue purchased a building on Rivington Street on Manhattan’s Lower East Side for $2.1 million. In 2014, Mr. Cohen sold the building for $10 million in cash — nearly five times what he paid three years earlier.

That same day, Mr. Cohen sold three other Manhattan buildings, also entirely for cash. In total, the four buildings sold for $32 million — nearly triple what Mr. Cohen had paid for them in the span of no more than three years.

It is not known who owns the limited liability companies that purchased the properties from Mr. Cohen or why they agreed to pay such rich prices. Mr. Cohen told McClatchy, which first reported the transactions, that the sales were in cash to help the buyers defer taxes in other transactions.

Richard K. Gordon, director of the Financial Integrity Institute at Case Western Reserve University’s law school, said that such real estate transactions — large profits, achieved quickly, involving cash purchases by L.L.C.s — should raise red flags.

“If I were the bank, I’d either refuse his business up front or rate him extra high risk,” said Mr. Gordon, who once led anti-money-laundering efforts for the International Monetary Fund.

Mr. Trump is closely monitoring the investigations into his longtime counselor. In recent weeks, he has moved to simultaneously embrace Mr. Cohen and keep him at arm’s length.

“Most people will flip if the Government lets them out of trouble,” Mr. Trump wrote on Twitter last month. “Sorry, I don’t see Michael doing that.”

A few days later, Mr. Trump asserted that the investigation was unrelated to Mr. Cohen’s work for him. '''


yackreview = re.sub('[\s\n\r\t]+',' ',re.sub(r'\w+(\-|\—|\.|\&|\’)?(\w+)?', remove_stop, clean_text(REVIEW))).split()
yackvec_bow = dictionary.doc2bow(yackreview)
yackvec_lda = lda[yackvec_bow]
dense1 = gensim.matutils.sparse2full(yackvec_lda, lda.num_topics)
dense2 = gensim.matutils.sparse2full(totalvec_lda, lda.num_topics)
score = np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())
grade = ''
if(score<.45):
   grade=('f')
elif(score<.53):
	grade=('d')
elif(score<.58):
	grade=('c')
elif(score<.64):
	grade=('b')
else:
	grade=('a')

string = ("Your grade is {} with a {} percent likelihood of it being a Pitchfork Review".format(grade,round(pdf[grade],3)*100))
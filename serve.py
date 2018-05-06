#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import *
from gensim.models import *
import re
import unicodedata
import os
import codecs
import gensim
import numpy as np
from flask import Flask, jsonify, g, redirect, request, url_for
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/userreview',methods=['GET', 'OPTIONS'])
def userreview():
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
	
	REVIEW = request.args.get('input')
	
	
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

	return jsonify({"success":{"grade":grade,"probability":round(pdf[grade],3)*100}})
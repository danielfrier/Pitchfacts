from gensim import *
from gensim.models import *
import re
import unicodedata
import os
import codecs

wem = Word2Vec.load('models/wem/writerviz.wem')
REVIEW = '''Naruto Uzumaki is the fictional protagonist of the Japanese manga series Naruto, created by Masashi Kishimoto. A carefree, optimistic and boisterous teen ninja who befriends other ninja, he aspires to become the leader of his fictional village, Konohagakure. He appears in anime, films, video games and original video animations, as well as a sequel Boruto: Naruto Next Generations, with his son Boruto Uzumaki as the protagonist. Kishimoto initially aimed to keep the character "simple and stupid", while giving him many attributes of an ideal hero, and a tragic past. The author has revised Naruto's image many times, providing the character with different clothes intended to appeal to Western audiences and to make him easier to illustrate. Naruto is voiced by Junko Takeuchi (pictured) in the original animated series and Maile Flanagan in the English adaptations. The character's development has been praised by anime and manga publications, and has drawn scholarly attention.'''
with open("stopwords.txt") as f:
    stopwords = [word for line in f for word in line.split()]
def remove_stop(m):
    return '' if m.group() in stopwords else m.group()
user_review = str(re.sub('—',' ',REVIEW).decode('unicode_escape').encode('ascii','ignore') )
user_review = re.sub(r'\w+(\-|\—|\.|\&|\’)?(\w+)?', remove_stop, user_review) 
user_review = re.sub(r'\-+\s', ' ', user_review)
sentences = re.split('[\.\!\?]\s',re.sub('[^A-z\.\!\?\-\&]+',' ',user_review.lower()))
sentences = [sentences[i].split(' ') for i in list(range(len(sentences)))]

score = wem.score(sentences,len(sentences))
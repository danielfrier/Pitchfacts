from gensim import *
from gensim.models import *
import re
import unicodedata
import os
import codecs
import matplotlib.pyplot as plt
index = similarities.MatrixSimilarity.load('models/similarity.index')

dictionary = corpora.Dictionary.load('models/orig_text.dict')
corpus = corpora.MmCorpus('models/orig_text.mm')
lsi = LsiModel.load('models/twodlsa.lsi')

REVIEW = '''Let's be frank. Getting old is not for sissies. Our bodies begin to show signs of wear and tear as we age. Not surprisingly, depending on our gene pool and the life we've led, stuff starts to wear out. The good news is, modern science has made it possible to replace a lot of what ails us. The bad news? When it comes to our hearing, even the best technology cannot restore this sense to normal.
Vision

Perfect vision is rare. In fact, it's estimated that more than 160 million people in the United States wear corrective lenses. Whether you're nearsighted or farsighted, your eyeglass lenses use curved pieces of glass to bend the light rays so your eyes can focus clearly. Lasik eye surgery actually reshapes the cornea so no eyeglasses are needed, in most cases. And those with cataracts can often have their cloudy lenses surgically replaced with clear, manmade lenses. As long as the optic nerve functions properly and you don't have any other medically-related issues that inhibit your eyesight, these methods are usually successful in restoring vision to near perfect levels.
eye test
Having your hearing tested regularly is just as 
important as your vision and dental check-ups!

It works a bit differently with our hearing. Although cochlear implants are the closest thing our ears have to lens implants or Lasik surgery, the result is not the same. Cochlear implants are surgically implanted in the inner ear and activated by a device worn outside the ear. They bypass damaged parts of the inner ear to help those with severe hearing impairments hear sound. Most cochlear implant patients report improved communication skills; however, must learn how to interpret the sounds they are hearing. Even digital hearing aid technology is incapable of restoring our hearing the way eyeglasses restore our vision.

That's because the part of our auditory system affected by old age is a bit more complex than the mechanics involved with reshaping a cornea or replacing a lens. Think of how your ears, in tandem with your brain, are able to focus in on the conversation in a noisy sports arena or busy restaurant. Or how they can distinguish the highest pitches of a church bell from the lowest sounds of a bass guitar -- and everything in between. Today's electronic hearing devices do a good job of mimicking our body's ability to identify and interpret the rapid, sometimes chaotic, symphony of sounds we hear almost constantly, but they still haven't perfected the unique transformation process that happens in the cochlea when noise is translated into electrical impulses for the brain to understand.
Dental health

If you live long enough, you'll probably hear your dentist tell you your gums are receding. It's a common problem for individuals over the age of 40. When your gums recede, they often expose the roots of your teeth, leaving them open to decay and sensitivity. Of course, your dentist is going to tell you to keep your mouth clean (so did your mom, right?) by brushing regularly and flossing daily. If this advice fails you, orthodontists may be able to graft from another part of your mouth to fill the gaps. This procedure not only stops gum recession, it also stimulates re-growth.

Dental implants are another effective way of replacing badly worn teeth. Unlike removable dentures or partials, implants are surgically inserted into the bone under the gum and are the closest thing to natural permanent teeth.
Why doesn't it work that way with our hearing?

As we age, our hearing deteriorates, too. Age-related hearing loss is known as presbycusis, a gradual condition that usually occurs in both ears. Commonly, presbycusis is the result of damage to the hair cells in our inner ear and may be more noticeable with higher pitched sounds, such as chirping birds or smoke alarms.

And although modern science has found a way to transplant hair follicles onto your head and graft skin onto other parts of our bodies, they haven't found a way to transplant those tiny hair cells of our inner ear, also known as stereocilia. That might be because these sensory receptors aren't actually hair at all, but extensions of the cells that comprise the Organ of Corti inside the cochlea. Noise makes these hair cells wiggle and, when they do, they produce electrical impulses that travel along the auditory nerve to the brain, which interprets them as sound.
Hearing health and old age

Presbycusis is sensorineural in nature, meaning it involves damage to the inner ear or auditory nerve. Although most cases of presbycusis occur gradually, some may be the result of prolonged exposure to loud noise, changes in the blood supply to the ears as a result of heart disease or diabetes, or abnormalities of the outer ear or inner ear. Individuals with presbycusis may also have tinnitus, a condition commonly described as a constant ringing in the ears.

According to the American Speech-Language-Hearing Association (ASHA), between 25-40 f Americans 65 years of age and older are hearing impaired. If you suspect you have hearing loss, see your doctor or a qualified hearing health professional. They will examine your ears and administer a hearing test to determine what type of hearing loss you have and recommend the best course of treatment.

Take heart. Although modern science hasn't perfected manmade hearing replacement, today's hearing aids are much more effective than those from even ten years ago. They can connect wirelessly to your television, smart phone and computer. They can be programmed to work in every facet of your lifestyle, whether you are a bird watcher, avid golfer, or just enjoy talking to your grandkids on the telephone. They're small and discreet, with models that fit almost invisibly behind your ear or completely in your ear canal if you desire. Your audiologist can help you determine which style and model works the best for you - and may even offer a generous trial period to make sure you're completely satisfied with their functionality. It's worth the effort - the majority of hearing aid users report more confidence and happiness in almost every aspect of life.

In short, hearing loss can be a normal part of the aging process. The sooner you identify the problem, the sooner you can be on your way to improved hearing health and quality of life.'''
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

review = clean_text(REVIEW)
vec_bow = dictionary.doc2bow(review.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space

sims = index[vec_lsi]
print(list(enumerate(sims)))
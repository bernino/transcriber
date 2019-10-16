import os
from multiprocessing.dummy import Pool
import speech_recognition as sr
import sys
import librosa
import soundfile as sf
import pke
import string
from nltk.corpus import stopwords

pool = Pool(8) # Number of concurrent threads
files = []
r = sr.Recognizer()

# Create list of wav files
# Currently the source file is chopped up in parts to annotate time
# and needs to be in mono 16khz sampled wav file
# Correct 16khz mono format for most transcribers:
# ffmpeg -i xxx.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav
# 30s pieces:
# ffmpeg -i out.wav -f segment -segment_time 30 -c copy parts/out%09d.wav
for file in sorted(os.listdir('parts/')):
    if file.endswith('.wav'):
        files.append(file) 


def keyworder(text):
    # using pke https://github.com/boudinfl/pke
    # https://boudinfl.github.io/pke/build/html/unsupervised.html#topicrank
    # 1. create a TopicRank extractor.
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text)
    pos = {'NOUN', 'PROPN', 'ADJ'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    # 4. build topics by grouping candidates with HAC (average linkage,
    #    threshold of 1/4 of shared stems). Weight the topics using random
    #    walk, and select the first occuring candidate from each topic.
    extractor.candidate_weighting(threshold=0.75, method='average')
    # 5. get the 10-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=5)
    return keyphrases


def transcribe(data):
    # transcribe each file using Google's API service
    # data is passed as variable containing the index for multithreading
    idx, file = data
    name = "parts/" + file
    print(name + " started")
    # Load audio file and preprocess to mono 16khz just in case
    # this and ffmpeg preprocessing should prob be done in function
    # however if ffmpeg done properly librosa shouldn't be needed
    x,_ = librosa.load(name, sr=16000)
    sf.write(name, x, 16000)
    with sr.AudioFile(name) as source:
        audio = r.record(source)
    # Transcribe audio file with google API
    # The only standalone network is Mozilla deepspeech and it sucks
    # Alternatives are Bing, IBM, ..., check the documentation of sr
    text = r.recognize_google(audio)
    keywords = keyworder(text)
    # we are not interested in probabilistic rank
    # so cleaning up here
    clean_keywords = ''
    for k, v in keywords:
        clean_keywords = k + " - " + clean_keywords
    clean_keywords = clean_keywords[:-2] 

    #, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
    print(name + " done")
    return {
        "idx": idx,
        "text": text,
        "keywords": clean_keywords
    }

# Each file is indexed and then calls transcribe() as variable "data"
# for multithreading to work.
# The structure is idx, file = data
all_text = pool.map(transcribe, enumerate(files))
pool.close()
pool.join()

transcript = ""
clean_transcript = ""

for t in sorted(all_text, key=lambda x: x['idx']):
    total_seconds = t['idx'] * 30
    # Cool shortcut from:
    # https://stackoverflow.com/questions/775049/python-time-seconds-to-hms
    # to get hours, minutes and seconds
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)

    # Format time as h:m:s - 30 seconds of text
    # add keywords for each 30s
    transcript = transcript + "{:0>2d}:{:0>2d}:{:0>2d} : {} \n \n {} \n \n".format(h, m, s, t['keywords'], t['text'])
    clean_transcript = clean_transcript + t['text']

global_keywords = keyworder(clean_transcript)
keywords = ''

for k, v in global_keywords:
    keywords = k + " - " + keywords
keywords = keywords[:-2] 

transcript = transcript + "\n\n" + keywords

print(transcript)

with open("transcript.txt", "w") as f:
    f.write(transcript)
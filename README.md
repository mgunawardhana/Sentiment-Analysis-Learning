[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=40&pause=1000&color=1FF77A&center=true&vCenter=true&width=750&height=60&lines=SENTIMENT+ANALYSIS+LEARNING...)](https://git.io/typing-svg)

## ꜱᴇᴛᴜᴘ ᴋᴀɢɢʟᴇ ᴀᴜᴛʜ
```python
import os
os.environ['KAGGLE_USERNAME'] = data['username']
os.environ['KAGGLE_KEY'] = data['key']

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
```

```python
import numpy as np
import pandas as pd
import re
import string
import pickle

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

with open('../static/model/model.pickel', 'rb') as f:
    model = pickle.load(f)

with open('../static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

vocab = pd.read_csv('../static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " "
    .join(x.lower() for x in x.split())) # tweet column to lowercase
    data["tweet"] = data["tweet"]
    .apply(lambda x: " "
    .join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"] = data["tweet"].str.replace('\d+', '', regex=True) # removing numbers
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data['tweet'] = data['tweet']
    .apply(lambda x: " ".join(ps.stem(x) for x in x.split())) # getting base words
    return data['tweet']

def vectorizer(ds, vocabulary):
    vectorized_lst = []

    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))

        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1

        vectorized_lst.append(sentence_lst)

    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)

    return vectorized_lst_new

txt = 'awesome product i love it'
preprocessed_txt = preprocessing(txt)
vectorized_txt = vectorizer(preprocessed_txt, tokens)
prediction = model.predict(vectorized_txt)
prediction

get_prediction(prediction)
```


<div align="center">

### (★) ɪꜰ ʏᴏᴜ ʜᴀᴠᴇ ᴀɴʏ ʙᴜɢꜱ ᴏʀ ɪꜱꜱᴜᴇꜱ , ɪꜰ ʏᴏᴜ ᴡᴀɴᴛ ᴛᴏ ᴇxᴘʟᴀɪɴ ᴍʏ ᴄᴏᴅᴇ ᴏʀ ɪꜰ ʏᴏᴜ ɴᴇᴇᴅ ʜᴇʟᴘ ᴛᴏ ᴅᴇᴠᴇʟᴏᴘ ʏᴏᴜʀ ᴘʀᴏᴊᴇᴄᴛꜱ ᴘʟᴇᴀꜱᴇ ᴄᴏɴᴛᴀᴄᴛ ᴍᴇ ᴏɴ (★) 👇<br> <br> <br> maneesha.gunawardhana.contact@gmail.com

</div>

<div align="center">
 <h3>USED TECHNOLOGIES & TOOLS</h3>
     <img src="https://skillicons.dev/icons?i=py,pycharm,anaconda,github" />

</div>

<br><br>
<div align="center">

![repo size](https://img.shields.io/github/repo-size/mGunawardhana/sentiment_analysis_project_01?style=for-the-badge) &nbsp;
![GitHub](https://img.shields.io/github/license/mGunawardhana/sentiment_analysis_project_01?style=for-the-badge) &nbsp;
![GitHub Forks](https://img.shields.io/github/forks/mGunawardhana/sentiment_analysis_project_01?&labelColor=black&color=f7b731&style=for-the-badge) &nbsp;
![GitHub Watchers](https://img.shields.io/github/watchers/mGunawardhana/sentiment_analysis_project_01?style=for-the-badge) &nbsp;
![GitHub Last Commit](https://img.shields.io/github/last-commit/mGunawardhana/sentiment_analysis_project_01?style=for-the-badge) &nbsp;

</div>
<br><br>

<div align="center">

## © 2025 mGunawardhana. ᴀʟʟ ʀɪɢʜᴛꜱ ʀᴇꜱᴇʀᴠᴇᴅ.

</div>

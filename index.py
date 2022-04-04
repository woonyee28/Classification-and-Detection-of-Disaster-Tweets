import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
import re
import string
import streamlit as st

@st.cache(suppress_st_warning=True)
def predict_disaster(predict_msg):
    tweets = pd.read_csv("train.csv")

    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    def remove_html(text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)

    def remove_emoji(text):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_punct(text):
        table=str.maketrans('','',string.punctuation)
        return text.translate(table)

    def decontraction(text):
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
        
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)

        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text 

    def seperate_alphanumeric(text):
        words = text
        words = re.findall(r"[^\W\d_]+|\d+", words)
        return " ".join(words)

    def cont_rep_char(text):
        tchr = text.group(0) 
        
        if len(tchr) > 1:
            return tchr[0:2] 

    def unique_char(rep, text):
        substitute = re.sub(r'(\w)\1+', rep, text)
        return substitute

    tweets['text']=tweets['text'].apply(lambda x : remove_URL(x))

    tweets['text']=tweets['text'].apply(lambda x : remove_html(x))

    tweets['text']=tweets['text'].apply(lambda x: remove_emoji(x))

    tweets['text']=tweets['text'].apply(lambda x : remove_punct(x))

    tweets['text'] = tweets['text'].apply(lambda x : decontraction(x))

    tweets['text'] = tweets['text'].apply(lambda x : seperate_alphanumeric(x))

    tweets['text'] = tweets['text'].apply(lambda x : unique_char(cont_rep_char, x))

    is_disaster_tweets = tweets[tweets.target == 1]
    not_disaster_tweets = tweets[tweets.target == 0]
    not_disaster_tweets_downsampled = not_disaster_tweets.sample(n = len(is_disaster_tweets), random_state = 44)
    is_disaster_tweets_downsampled = is_disaster_tweets
    tweets_concat = pd.concat([not_disaster_tweets_downsampled,is_disaster_tweets_downsampled]).reset_index(drop=True)
    train_tweets, test_tweets, train_target, test_target = train_test_split(tweets_concat['text'], tweets_concat['target'], test_size=0.25, random_state=44)

    max_len = 50 
    trunc_type = "post" 
    padding_type = "post" 
    oov_tok = "<OOV>" 
    vocab_size = 500

    tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
    tokenizer.fit_on_texts(train_tweets)

    training_sequences = tokenizer.texts_to_sequences(train_tweets)
    training_padded = pad_sequences (training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )
    testing_sequences = tokenizer.texts_to_sequences(test_tweets)
    testing_padded = pad_sequences(testing_sequences, maxlen = max_len,padding = padding_type, truncating = trunc_type)

    vocab_size = 500 # As defined earlier
    embeding_dim = 16
    drop_value = 0.2 # dropout
    n_dense = 24

    model = Sequential()
    model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(drop_value))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
    num_epochs = 30
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(training_padded, train_target, epochs=num_epochs, validation_data=(testing_padded, test_target),callbacks =[early_stop], verbose=2)
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    return (model.predict(padded))

#To install streamlit, run "pip install streamlit" in the terminal
#To run the file, run "streamlit run index.py" in the terminal

st.title("Tweets Prediction (Disaster)")
st.write(""" We need some information to predict the likelihood of Disaster""")

input_text = st.text_input("Enter the tweet: ")

ok = st.button("Calculate likelihood of Disaster")
if ok:
    input_list = [input_text]
    ans = predict_disaster(input_list)
    if ans>=0.5:
        st.subheader("It is a disaster tweet!!! :'(")
    else:
        st.subheader("It is NOT a disaster tweet!!! :)")
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
## import new 
from nltk.stem import WordNetLemmatizer 



def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)

    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 

    # remove <br /><br />
    # text = text.split("<br /><br />")[1].replace("\n", " ")
    
    # 使用 NLTK 提供的句子分割器
    sent_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_segmenter.tokenize(text)

    # set stop word
    stop_word_list = stopwords.words('english')

    # Lemmatization
    
    lm = WordNetLemmatizer()
    s_list=[]
    for s in sentences:
        w = s.split(' ')
        for i in range(len(w)):
            lemma = lm.lemmatize(w[i],'v')
            if lemma == w[i]:
                lemma = lm.lemmatize(w[i],'n')
            s_list.append(lemma)
        
    filtered_token = [token for token in s_list if token.lower() not in stop_word_list]
    lemmatization_text = ' '.join(filtered_token)

    # print("lemma:\n")
    # print(lemmatization_text)
    # print("\n")
    # print('preprocessed:\n')
    # print(preprocessed_text)
    # End your code

    return preprocessed_text

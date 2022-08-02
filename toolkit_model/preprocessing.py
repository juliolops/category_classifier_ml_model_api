import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from unicodedata import normalize as norm
nltk.download('stopwords')

def text_cleaner(text):
    
    nltk_stopwords = stopwords.words('portuguese')

    collection_text = [ {"text" : text}]
    text = pd.DataFrame(collection_text)

    text['text'] = text['text'].astype('str')
    text['text'] = text['text'].str.lower()
    text['text'] = text['text'].str.replace('\n',' ')
    text['text'] = text['text'].str.replace('\r',' ')
    text['text'] = text['text'].apply(lambda x: norm('NFKD', x).encode('ascii', 'ignore').decode())
    text['text'] = text['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
    text['text'] = text['text'].apply(lambda x: re.sub(r'\s+',' ',x))
    pat = r'\b(?:{})\b'.format('|'.join(nltk_stopwords))
    text['text'] = text['text'].str.replace(pat,'')
    text = text['text'].values[0]

    return text



def text_categorizer(text,labels):
    
    final_class = ""
    
    for i in labels:
        
        regex = r'(?is){}'.format(i)
        
    
        if re.match(regex, text):

            final_class= final_class + i

    if final_class != "":
        
        return final_class
    
    
    else :

        return "Outros"



def create_dummies(df,list_dummies):
        
    for dummy_feature in list_dummies:
    
        if re.match(r"^query_.*",dummy_feature):
            
            df[dummy_feature] = df["query_cleaned_modified"].map(lambda x:1 if re.sub(r"^query_","",dummy_feature)==re.sub(r"\s","",x) else 0)
            
        elif re.match(r"^concatenated_tags_.*",dummy_feature):
            
            df[dummy_feature] = df["concatenated_tags_cleaned_modified"].map(lambda x:1 if re.sub(r"^concatenated_tags_","",dummy_feature)==re.sub(r"\s","",x) else 0)
            
        elif re.match(r"^title_.*",dummy_feature):

            
            df[dummy_feature] = df["title_cleaned_modified"].map(lambda x:1 if re.sub(r"^title_","",dummy_feature)==re.sub(r"\s","",x) else 0)


    return df
            
        
        
import re
import nltk
from nltk import word_tokenize, sent_tokenize
import pandas as pd

nltk.download('punkt', quiet=True)

class BasicTextProcessor:
    def __init__(self):
        pass

    def clean_text(self, text:str) -> str:
        text=text.lower()
        text=re.sub(r'http\S+|www\S+|https\S+', '', text)
        text=re.sub(r'\S+@\S+', '', text)
        text=re.sub(r'<.*?>', '', text)
        text=re.sub(r'^a-zA-Z0-9\s\.\?\,\!\:\;\-\(\)\[\]\{\}\<\>\'\"\\\/\_\=\+\*]', '', text)
        return text
    
    def tokenise_sentences(self,text:str):
        return [i.strip() for i in sent_tokenize(text) if i.strip() if i.strip()]
    
    def tokenise_words(self,text:str):
        return [i for i in word_tokenize(text) if re.search(r'[a-zA-Z0-9]', i)]
    
    def preprocess_text(self,text:str):
        cleaned_text = self.clean_text(text)
        sentences=self.tokenise_sentences(cleaned_text)
        words=self.tokenise_words(cleaned_text)
        return {
            'original_text':text,
            'cleaned_text':cleaned_text,
            'words':words ,
            'sentence length':len(sentences),
            'word length':len(words)
             }
    def preprocess_batch(self,texts):
        return pd.DataFrame([
            {**self.preprocess_text(text),'textid':i} for i,text in enumerate(texts)])
    
    def get_basic_stats(self, texts):
        words, sentences = [], []
        for text in texts:
            r = self.preprocess_text(text)
            words += r['words']
            sentences += r['sentences']
        word_lens = [len(w) for w in words]
        sent_lens = [len(s.split()) for s in sentences]
        return {
            'total_texts': len(texts),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'unique_words': len(set(words)),
            'avg_words_per_text': len(words) / len(texts),
            'avg_sentences_per_text': len(sentences) / len(texts),
            'avg_word_length': sum(word_lens) / len(word_lens),
            'avg_sentence_length': sum(sent_lens) / len(sent_lens)
        }
    
def demo():
    texts = [
        "Hello world! This is a test.",
        "Another example, with more text.",
        "This is a third sentence."
    ]
    
    processor = BasicTextProcessor()
    df = processor.preprocess_batch(texts)
    print(df)
    
    stats = processor.get_basic_stats(texts)
    print(stats)

if __name__ == "__main__":
    demo()
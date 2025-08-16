import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from typing import List, Dict ,Union 
import logging

nltk.data.find('tokenizers/punkt')

class BasicTextPreprocessor:
    """
        Initialize the text preprocessor.
        Focuses on cleaning and tokenization, and basic normalization.
    """
    def __init__(self):
        """        Initializes the BasicTextPreprocessor."""
        logging.basicConfig(level=logging.INFO)
    def clean_text(self, text: str)-> str:
        """
        Clean text by removing special characters and converting to lowercase.
        Args:
            text:Input text string.
        Returns:
            Cleaned text string.
        """
        if not isinstance(text,str):
            return""""""        
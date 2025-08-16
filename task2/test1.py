# 


from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Load 1% of the 2017 training set
dataset = load_dataset("nlphuji/coco2017_captions", split="train[:1%]")

print(f"Number of samples: {len(dataset)}")
print("Sample keys:", dataset[0].keys())


example = dataset[0]
image = example['image']
captions = [cap['text'] for cap in example['captions']]

plt.imshow(image)
plt.axis('off')
plt.title("\n".join(captions))
plt.show()



num_samples = len(dataset)
all_captions = [cap['text'] for sample in dataset for cap in sample['captions']]
caption_lengths = [len(cap.split()) for cap in all_captions]
avg_caption_len = np.mean(caption_lengths)

print(f"Total samples: {num_samples}")
print(f"Total captions: {len(all_captions)}")
print(f"Average caption length: {avg_caption_len:.2f} words")


## 📏 Image Resolution Distribution

resolutions = [sample['image'].size for sample in dataset]
widths, heights = zip(*resolutions)

plt.figure(figsize=(10,4))
plt.hist(widths, bins=20, alpha=0.7, label="Width")
plt.hist(heights, bins=20, alpha=0.7, label="Height")
plt.xlabel("Pixels")
plt.ylabel("Number of Images")
plt.title("Image Dimension Distribution")
plt.legend()
plt.show()


plt.figure(figsize=(8, 4))
sns.histplot(caption_lengths, bins=30, kde=True, color='skyblue')
plt.xlabel("Caption Length (Words)")
plt.ylabel("Frequency")
plt.title("Distribution of Caption Lengths")
plt.show()


from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
words = [word.lower() for cap in all_captions for word in cap.split() if word.lower() not in stop_words]
word_freq = Counter(words)

# Top 20 words
print("Top 20 frequent words:")
print(word_freq.most_common(20))

# Word cloud
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in Captions")
plt.show()



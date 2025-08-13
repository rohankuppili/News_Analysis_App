# import tkinter as tk
# from ipaddress import summarize_address_range
#
# import nltk
# from textblob import TextBlob
# from newspaper import Article
#
# def summarize():
#     url = utext.get('1.0', "end").strip()
#     article = Article(url)
#
#     article.download()
#     article.parse()
#
#     if not article.text:  # Check if content exists
#         summary.config(state='normal')
#         summary.delete('1.0', "end")
#         summary.insert('1.0', "Error: Unable to fetch article content.")
#         summary.config(state='disabled')
#         return
#
#     article.nlp()
#
#     title.config(state='normal')
#     author.config(state='normal')
#     publication.config(state='normal')
#     summary.config(state='normal')
#     sentiment.config(state='normal')
#
#     title.delete('1.0', "end")
#     title.insert('1.0', article.title)
#
#     author.delete('1.0', "end")
#     author.insert('1.0', ", ".join(article.authors) if article.authors else "N/A")
#
#     publication.delete('1.0', "end")
#     publication.insert('1.0', str(article.publish_date) if article.publish_date else "N/A")
#
#     summary.delete('1.0', "end")
#     summary.insert('1.0', article.summary if article.summary else "Summary not available.")
#
#     analysis = TextBlob(article.text)
#     sentiment.delete('1.0', 'end')
#     sentiment.insert('1.0', f'Polarity: {analysis.polarity}, Sentiment: {"Positive" if analysis.polarity > 0 else "Negative" if analysis.polarity < 0 else "Neutral"}')
#
#     title.config(state='disabled')
#     author.config(state='disabled')
#     publication.config(state='disabled')
#     summary.config(state='disabled')
#     sentiment.config(state='disabled')
#
#     root.update_idletasks()  # Ensure UI updates
#
# root = tk.Tk()
#
# root.title("News Summarizer")
# root.geometry('1200x600')
#
# tlabel = tk.Label(root, text= 'Title')
# tlabel.pack()
#
# title = tk.Text(root, height=1, width=140)
# title.config(state='disabled', bg='#dddddd')
# title.pack()
#
# alabel = tk.Label(root, text= 'Author')
# alabel.pack()
#
# author = tk.Text(root, height=1, width=140)
# author.config(state='disabled', bg='#dddddd')
# author.pack()
#
# plabel = tk.Label(root, text= 'Publication Date')
# plabel.pack()
#
# publication = tk.Text(root, height=1, width=140)
# publication.config(state='disabled', bg='#dddddd')
# publication.pack()
#
# slabel = tk.Label(root, text= 'Summary')
# slabel.pack()
#
# summary = tk.Text(root, height=20, width=140)
# summary.config(state='disabled', bg='#dddddd')
# summary.pack()
#
# selabel = tk.Label(root, text= 'Sentiment Analysis')
# selabel.pack()
#
# sentiment = tk.Text(root, height=1, width=140)
# sentiment.config(state='disabled', bg='#dddddd')
# sentiment.pack()
#
# ulabel = tk.Label(root, text= 'URL')
# ulabel.pack()
#
# utext = tk.Text(root, height=1, width=140)
# utext.pack()
#
# btn = tk.Button(root, text= 'Summarize', command = summarize)
# btn.pack()
#
# root.mainloop()



"""
import tkinter as tk
from textblob import TextBlob
from newspaper import Article
import spacy
import yake

# Load spaCy model once
nlp_spacy = spacy.load("en_core_web_sm")

def remove_redundant_keywords(keywords):
    filtered = []
    for word in keywords:
        if not any(word != other and word in other for other in keywords):
            filtered.append(word)
    return filtered

def summarize():
    url = utext.get('1.0', "end").strip()
    article = Article(url)

    article.download()
    article.parse()

    if not article.text:
        summary.config(state='normal')
        summary.delete('1.0', "end")
        summary.insert('1.0', "Error: Unable to fetch article content.")
        summary.config(state='disabled')
        return

    article.nlp()

    # Enable all fields
    for field in [title, author, publication, summary, sentiment, category, entities, keywords]:
        field.config(state='normal')

    # Title
    title.delete('1.0', "end")
    title.insert('1.0', article.title)

    # Author
    author.delete('1.0', "end")
    author.insert('1.0', ", ".join(article.authors) if article.authors else "N/A")

    # Publication Date
    publication.delete('1.0', "end")
    publication.insert('1.0', str(article.publish_date) if article.publish_date else "N/A")

    # Summary
    summary.delete('1.0', "end")
    summary.insert('1.0', article.summary if article.summary else "Summary not available.")

    # Sentiment
    analysis = TextBlob(article.text)
    polarity = analysis.polarity
    sentiment_value = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    sentiment.delete('1.0', 'end')
    sentiment.insert('1.0', f'Polarity: {polarity}, Sentiment: {sentiment_value}')

    # --- Article Categorization (Mock) ---
    categories = {
        'Politics': ['government', 'election', 'president', 'policy'],
        'Sports': ['game', 'tournament', 'score', 'player'],
        'Technology': ['software', 'AI', 'technology', 'gadget'],
        'Health': ['health', 'vaccine', 'medicine', 'doctor'],
        'Business': ['market', 'business', 'economy', 'finance']
    }

    lower_text = article.text.lower()
    cat_result = 'Uncategorized'
    for cat, keywords_list in categories.items():
        if any(word in lower_text for word in keywords_list):
            cat_result = cat
            break

    category.delete('1.0', 'end')
    category.insert('1.0', cat_result)

    # --- Named Entity Recognition ---
    doc = nlp_spacy(article.text)
    ents = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    entity_text = "\n".join(ents[:10]) if ents else "No named entities found."

    entities.delete('1.0', 'end')
    entities.insert('1.0', entity_text)

    # --- Keyword Extraction with YAKE ---
    kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)
    keywords_yake = kw_extractor.extract_keywords(article.text)
    keyword_list = [kw for kw, score in keywords_yake]
    keyword_list = remove_redundant_keywords(keyword_list)
    keyword_text = ", ".join(keyword_list)

    keywords.delete('1.0', 'end')
    keywords.insert('1.0', keyword_text)

    # Disable all fields again
    for field in [title, author, publication, summary, sentiment, category, entities, keywords]:
        field.config(state='disabled')

    root.update_idletasks()

# --- GUI Layout ---
root = tk.Tk()
root.title("News Summarizer")
root.geometry('1200x800')

# Title
tk.Label(root, text='Title').pack()
title = tk.Text(root, height=1, width=140, state='disabled', bg='#dddddd')
title.pack()

# Author
tk.Label(root, text='Author').pack()
author = tk.Text(root, height=1, width=140, state='disabled', bg='#dddddd')
author.pack()

# Publication Date
tk.Label(root, text='Publication Date').pack()
publication = tk.Text(root, height=1, width=140, state='disabled', bg='#dddddd')
publication.pack()

# Summary
tk.Label(root, text='Summary').pack()
summary = tk.Text(root, height=10, width=140, state='disabled', bg='#dddddd')
summary.pack()

# Sentiment
tk.Label(root, text='Sentiment Analysis').pack()
sentiment = tk.Text(root, height=1, width=140, state='disabled', bg='#dddddd')
sentiment.pack()

# Category
tk.Label(root, text='Category').pack()
category = tk.Text(root, height=1, width=140, state='disabled', bg='#dddddd')
category.pack()

# Entities
tk.Label(root, text='Named Entities').pack()
entities = tk.Text(root, height=4, width=140, state='disabled', bg='#dddddd')
entities.pack()

# Keywords
tk.Label(root, text='Top Keywords').pack()
keywords = tk.Text(root, height=4, width=140, state='disabled', bg='#dddddd')
keywords.pack()

# URL Input
tk.Label(root, text='URL').pack()
utext = tk.Text(root, height=1, width=140)
utext.pack()

# Button
btn = tk.Button(root, text='Summarize', command=summarize)
btn.pack()

root.mainloop()
"""



import tkinter as tk
from textblob import TextBlob
from newspaper import Article
import spacy
import yake
import tweepy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import ImageTk, Image
from transformers import pipeline

# Load spaCy model once
nlp_spacy = spacy.load("en_core_web_sm")

# Fake News Detection Model - Replace with a proper fake news model
fake_news_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")


# Tweepy setup for Twitter integration using your credentials
consumer_key = 'vAb5B22cPe7Q0ZGgaYtk7XZ6u'
consumer_secret = 'nX0Hh9c3rlC3lhkSHusBN1Urb0b5dRUnKmd6Fm1XWfgag3IVkc'
access_token = '1778123731061149697-tCGEhk5vyqX6IaYuz63tMGzxyVjDFv'
access_token_secret = 'pRIstDoIOmVZXscS0sbfU2WdJBaMYlcBxEscBu7KCOMwB'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def summarize():
    url = utext.get('1.0', "end").strip()
    if not url:
        print("No URL provided!")
        return

    print(f"Fetching article from: {url}")
    article = Article(url)

    try:
        article.download()
        article.parse()
    except Exception as e:
        print(f"Error fetching article: {e}")
        return

    if not article.text:
        summary.config(state='normal')
        summary.delete('1.0', "end")
        summary.insert('1.0', "Error: Unable to fetch article content.")
        summary.config(state='disabled')
        print("Article text could not be retrieved.")
        return

    print("Article fetched successfully.")
    article.nlp()

    # Enable all fields
    for field in [title, author, publication, summary, sentiment, category, entities, keywords, bias, fake_news, social_media]:
        field.config(state='normal')

    # Title
    title.delete('1.0', "end")
    title.insert('1.0', article.title)

    # Author
    author.delete('1.0', "end")
    author.insert('1.0', ", ".join(article.authors) if article.authors else "N/A")

    # Publication Date
    publication.delete('1.0', "end")
    publication.insert('1.0', str(article.publish_date) if article.publish_date else "N/A")

    # Summary
    summary.delete('1.0', "end")
    summary.insert('1.0', article.summary if article.summary else "Summary not available.")

    # Sentiment
    analysis = TextBlob(article.text)
    polarity = analysis.polarity
    sentiment_value = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    sentiment.delete('1.0', 'end')
    sentiment.insert('1.0', f'Polarity: {polarity}, Sentiment: {sentiment_value}')

    # --- Article Categorization (Mock) ---
    categories = {
        'Politics': ['government', 'election', 'president', 'policy'],
        'Sports': ['game', 'tournament', 'score', 'player'],
        'Technology': ['software', 'AI', 'technology', 'gadget'],
        'Health': ['health', 'vaccine', 'medicine', 'doctor'],
        'Business': ['market', 'business', 'economy', 'finance']
    }

    lower_text = article.text.lower()
    cat_result = 'Uncategorized'
    for cat, keywords_list in categories.items():
        if any(word in lower_text for word in keywords_list):
            cat_result = cat
            break

    category.delete('1.0', 'end')
    category.insert('1.0', cat_result)

    # --- Named Entity Recognition ---
    doc = nlp_spacy(article.text)
    ents = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    entity_text = "\n".join(ents[:10]) if ents else "No named entities found."

    entities.delete('1.0', 'end')
    entities.insert('1.0', entity_text)

    # --- Keyword Extraction with YAKE ---
    kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)
    keywords_yake = kw_extractor.extract_keywords(article.text)
    keyword_text = ", ".join([kw for kw, score in keywords_yake])

    keywords.delete('1.0', 'end')
    keywords.insert('1.0', keyword_text)

    # --- Fake News Detection ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # or your specific model name
    inputs = tokenizer(article.text, truncation=True, max_length=512, return_tensors="pt")
    outputs = fake_news_model.model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    label = fake_news_model.model.config.id2label[predicted_class]
    fake_news_result = [{"label": label}]

    fake_news.delete('1.0', 'end')
    fake_news.insert('1.0', f"Fake News Likely: {fake_news_result[0]['label']}")

    # --- Bias Detection ---
    blob = TextBlob(article.text)
    subjectivity = blob.sentiment.subjectivity
    bias_level = 'Low' if subjectivity < 0.3 else 'Moderate' if subjectivity < 0.7 else 'High'
    bias.delete('1.0', 'end')
    bias.insert('1.0', f"Bias Level: {bias_level}")

    # --- Social Media Integration ---
    keywords_list = [kw for kw, _ in keywords_yake]
    try:
        tweets = api.search_tweets(q=" OR ".join(keywords_list), lang="en", result_type="recent", count=5)
        social_media_text = "\n".join([tweet.text for tweet in tweets]) if tweets else "No relevant tweets found."
    except Exception as e:
        social_media_text = f"Twitter API error: {e}"

    social_media.delete('1.0', 'end')
    social_media.insert('1.0', social_media_text)

    # --- Word Cloud Visualization ---
    wordcloud = WordCloud(width=800, height=400).generate(article.text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud_image = BytesIO()
    plt.savefig(wordcloud_image, format='png')
    wordcloud_image.seek(0)

    img = Image.open(wordcloud_image)
    img_tk = ImageTk.PhotoImage(img)
    wordcloud_image_label.config(image=img_tk)
    wordcloud_image_label.image = img_tk  # Keep a reference to the image

    # Disable all fields again
    for field in [title, author, publication, summary, sentiment, category, entities, keywords, fake_news, bias, social_media]:
        field.config(state='disabled')

    root.update_idletasks()
    print("Summarization complete.")

# --- GUI Layout ---
root = tk.Tk()
root.title("News Summarizer")
root.geometry('1200x800')

# Scrollable frame
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# Title
tk.Label(scrollable_frame, text='Title').pack()
title = tk.Text(scrollable_frame, height=1, width=140, state='disabled', bg='#dddddd')
title.pack()

# Author
tk.Label(scrollable_frame, text='Author').pack()
author = tk.Text(scrollable_frame, height=1, width=140, state='disabled', bg='#dddddd')
author.pack()

# Publication Date
tk.Label(scrollable_frame, text='Publication Date').pack()
publication = tk.Text(scrollable_frame, height=1, width=140, state='disabled', bg='#dddddd')
publication.pack()

# Summary
tk.Label(scrollable_frame, text='Summary').pack()
summary = tk.Text(scrollable_frame, height=10, width=140, state='disabled', bg='#dddddd')
summary.pack()

# Sentiment
tk.Label(scrollable_frame, text='Sentiment Analysis').pack()
sentiment = tk.Text(scrollable_frame, height=1, width=140, state='disabled', bg='#dddddd')
sentiment.pack()

# Category
tk.Label(scrollable_frame, text='Category').pack()
category = tk.Text(scrollable_frame, height=1, width=140, state='disabled', bg='#dddddd')
category.pack()

# Entities
tk.Label(scrollable_frame, text='Named Entities').pack()
entities = tk.Text(scrollable_frame, height=4, width=140, state='disabled', bg='#dddddd')
entities.pack()

# Keywords
tk.Label(scrollable_frame, text='Top Keywords').pack()
keywords = tk.Text(scrollable_frame, height=4, width=140, state='disabled', bg='#dddddd')
keywords.pack()

# Fake News Detection
tk.Label(scrollable_frame, text='Fake News Detection').pack()
fake_news = tk.Text(scrollable_frame, height=1, width=140, state='disabled', bg='#dddddd')
fake_news.pack()

# Bias Detection
tk.Label(scrollable_frame, text='Bias Detection').pack()
bias = tk.Text(scrollable_frame, height=1, width=140, state='disabled', bg='#dddddd')
bias.pack()

# Social Media Integration
tk.Label(scrollable_frame, text='Relevant Tweets/Posts').pack()
social_media = tk.Text(scrollable_frame, height=6, width=140, state='disabled', bg='#dddddd')
social_media.pack()

# Word Cloud Image
tk.Label(scrollable_frame, text='Word Cloud').pack()
wordcloud_image_label = tk.Label(scrollable_frame)
wordcloud_image_label.pack()

# URL input and button
url_label = tk.Label(scrollable_frame, text="Enter Article URL")
url_label.pack()

utext = tk.Text(scrollable_frame, height=2, width=120)
utext.pack()

submit_btn = tk.Button(scrollable_frame, text="Summarize", command=summarize)
submit_btn.pack(pady=10)

root.mainloop()

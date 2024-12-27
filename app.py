import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Load your data
data = pd.read_csv('Movie_Review.csv')

# Customizing Streamlit
st.set_page_config(page_title="Movie Review Sentiment Analysis")
st.markdown(
    """
    <style>
   
    h1 {
        color: pink;
        font-size: 36px;
        position: relative;
        text-align:center;
        top:0px;
    }
    h1::after {
        content: '';
        position: absolute;
        left: 0;
        right: 0;
        bottom: 6px; /* Adjust the distance from the text */
        border-bottom: 2px solid pink; /* Adjust the thickness and color of the underline */ 
    }
     .stTextInput > label {
          top: 300px;
    }
    
    .emoji {
        width: 35%;
    }
    h2 {
        color: #4b0082;
        font-size: 28px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 10px 24px;
        margin-top: 10px;
    }
    .stTextInput>div>div>input {
        font-size: 20px;   
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Movie Review Sentiment Analysis")

# Input section
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.success('Negative Review')
        st.markdown('<img class="emoji" src="https://cdn.pixabay.com/photo/2020/09/22/14/55/sad-emoji-5593352_640.png">', unsafe_allow_html=True)  # Large emoji for negative review
    else:
        st.success('Positive Review')
        st.markdown('<img class="emoji" src="https://m.media-amazon.com/images/I/715vwvP5ZEL.png">', unsafe_allow_html=True)  # Large emoji for positive review

# Visualization sections

# Review Sentiment Distribution
st.subheader('Review Sentiment Distribution')
sentiment_counts = data['sentiment'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, hue=sentiment_counts.index, palette='viridis', ax=ax, legend=False)
ax.set_xlabel('Sentiment', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
st.pyplot(fig)

# Word Clouds
st.subheader('Word Cloud for Positive and Negative Reviews')
col1, col2 = st.columns(2)

positive_reviews = data[data['sentiment'] == 'pos']['text'].str.cat(sep=' ')
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_reviews)

negative_reviews = data[data['sentiment'] == 'neg']['text'].str.cat(sep=' ')
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(negative_reviews)

with col1:
    st.image(wordcloud_pos.to_array(), caption='Positive Reviews Word Cloud', use_column_width=True)
with col2:
    st.image(wordcloud_neg.to_array(), caption='Negative Reviews Word Cloud', use_column_width=True)

# Review Length Analysis
data['review_length'] = data['text'].apply(len)

st.subheader('Review Length Distribution')
fig, ax = plt.subplots()
sns.histplot(data['review_length'], bins=50, kde=True, color='purple', ax=ax)
ax.set_xlabel('Review Length', fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
st.pyplot(fig)

st.subheader('Review Length vs Sentiment')
fig, ax = plt.subplots()
sns.boxplot(x='sentiment', y='review_length', hue='sentiment', data=data, palette='coolwarm', ax=ax, legend=False)
ax.set_xlabel('Sentiment', fontsize=15)
ax.set_ylabel('Review Length', fontsize=15)
st.pyplot(fig)

# Confusion Matrix
data['sentiment_numeric'] = data['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)
y_true = data['sentiment_numeric']
y_pred = model.predict(scaler.transform(data['text']))

st.subheader('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='viridis')
ax.set_xlabel('Predicted Label', fontsize=15)
ax.set_ylabel('True Label', fontsize=15)
st.pyplot(fig)

import streamlit as st
import pandas as pd 
import numpy as np 
import re
import string
from datetime import datetime
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import joblib 
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import pickle

df = pd.read_csv("Streamlit/companySetting")

# Load model and preprocessing functions
#extra ML made just for uber
uber1 = pickle.loads("Streamlit/Uber1.pkl")
company = joblib.load("Streamlit/company.pkl")

nltk.download('stopwords')


def generate_N_grams(text,ngram):
  words=[word for word in text.split(" ") if word not in set(stopwords.words('english'))]  
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans



def get_prediction_proba(docx):
	results = company.predict_proba([docx])
	return results


def home():
    st.title("Company Review Analysis")
    st.write("The goal of this analysis is to analyze employee reviews submitted on Glassdoor, with the hope to help employers gain real insights on their employee engagement.") 
    image1= Image.open("Streamlit/company1.png")
    st.image(image1)
    st.write("The analysis aimed to answer these questions: ")
    st.write("What employees like and dislike about this company? What are the keywords that people say about this company? What can this company do to improve employee engagement?  ")
    
    with st.form(key='company'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

        # Preprocess data
        input_text_processed = raw_text.lower()
        input_text_processed = re.sub(r"\d+", "", input_text_processed)
        input_text_processed = input_text_processed.translate(str.maketrans("", "", string.punctuation))
        input_text_processed = input_text_processed.strip()

    if submit_text:
        col1,col2  = st.beta_columns(2)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Preprocessed Text")
            st.write(input_text_processed)

            st.success("Confidence")
            st.write("Confidence:{}".format(np.max(probability)))


        with col2:
            
            st.success("Prediction Probability")

            # Make prediction
            prediction = uber1.predict([input_text_processed])[0]

            # Display result
            st.write(f"Prediction: {prediction}")
            # Display result as a bar chart
            result_dict = {"Positive": 0, "Negative": 0}
            if prediction == 1:
                result_dict["Positive"] += np.max(probability)
                result_dict["Negative"] += 1-np.max(probability)
            else:
                result_dict["Negative"] += np.max(probability)
                result_dict["Positive"] += 1-np.max(probability)
            result_df = pd.DataFrame.from_dict(result_dict, orient="index", columns=["Count"])
            st.bar_chart(result_df)
                    
    image= Image.open("Streamlit/company.png")
    st.image(image)





def companys():
    st.title("Companys")
    st.header("Select the company you want to receive information about:")
    companies = ['IBM', 'Microsoft', 'Amazon', 'Meta', 'Booz Allen Hamilton',
       'JPMorgan Chase & Co', 'Google', 'Capital One', 'Deloitte',
       'Nielsen', 'Target', 'Accenture', 'AT&T', 'Apple',
       'Dell Technologies']
    selected_company = st.selectbox("Select a company", companies)

    gk = df.groupby('company')
    if selected_company in companies:
        st.write(f"You selected {selected_company}")

        col1,col2  = st.beta_columns(2)
        with col1:
            # Create Seaborn plot and sentiment analsis
            sns.countplot(x='sentiment',data=gk.get_group(f"{selected_company}"))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
    
        with col2:
            gk = df.groupby('company')
            gk1 = gk.get_group(f"{selected_company}")
            wordcloud = WordCloud(max_words=200, width =1280, height = 720, background_color="white").generate(" ".join(gk1["reviews"][df["sentiment"] == 1]))
            plt.figure(figsize=[15,15])
            # Display word cloud
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot()
    else:
        st.write("You selected an unknown company")

    positiveValues=defaultdict(int)
    negativeValues=defaultdict(int)

    #get the count of every word in both the columns of df_train and df_test dataframes where sentiment="positive"
    for text in gk.get_group(f"{selected_company}")[gk.get_group(f"{selected_company}").sentiment== 1 ].reviews:
        for word in generate_N_grams(text,2):
            positiveValues[word]+=1
    #get the count of every word in both the columns of df_train and df_test dataframes where sentiment="negative"
    for text in gk.get_group(f"{selected_company}")[gk.get_group(f"{selected_company}").sentiment== 0 ].reviews:
        for word in generate_N_grams(text,2):
            negativeValues[word]+=1
    

    #sort in DO wrt 2nd column in each of positiveValues,negativeValues and neutralValues
    df_positive=pd.DataFrame(sorted(positiveValues.items(),key=lambda x:x[1],reverse=True))
    df_negative=pd.DataFrame(sorted(negativeValues.items(),key=lambda x:x[1],reverse=True))
    col3,col4  = st.beta_columns(2)
    with col3:
        st.success("Top 15 words in positive dataframe-BIGRAM ANALYSIS")
        st.table(df_positive.head(15))

    with col4:
        st.success("Top 15 words in negative dataframe-BIGRAM ANALYSIS")
        st.table(df_negative.head(15))

def main():
    pages = {
        "Home": home,
        "Companys": companys
    }

    st.sidebar.title("Navigation")
    page_names = list(pages.keys())
    page = st.sidebar.radio("Go to", page_names)

    selection = st.sidebar.selectbox("", page_names, index=page_names.index(page))
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()

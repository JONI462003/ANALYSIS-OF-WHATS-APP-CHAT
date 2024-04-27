import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textblob import TextBlob

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    df=df[df['user']!='group_notification']
    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    if '' in user_list:
        user_list.remove('')

    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show Analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        st.title("Top Statistics")
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        st.title('Monthly Timeline')
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color= 'green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title('Daily Timeline')
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color= 'black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            st.pyplot(fig)
        with col2:
            st.header("Most Busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color = 'orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        most_common_df = helper.most_common_words(df, selected_user)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels = emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)


        st.title("Sentiment Analysis")
        sentiment_results = helper.sentiment_analysis(df)
        sentiment_df = pd.DataFrame({'Sentiment Polarity': sentiment_results['Sentiment Polarity']})
        st.write(sentiment_df)
        st.write("Positive messages:", sentiment_results['Positive'])
        st.write("Negative messages:", sentiment_results['Negative'])
        st.write("Neutral messages:", sentiment_results['Neutral'])
        st.write("Positive Percentage:", sentiment_results['Positive Percentage'], "%")
        st.write("Negative Percentage:", sentiment_results['Negative Percentage'], "%")
        st.write("Neutral Percentage:", sentiment_results['Neutral Percentage'], "%")

        fig, ax=plt.subplots()
        ax.hist(sentiment_results['Sentiment Polarity'], bins=20)
        ax.set_xlabel('Sentiment Polarity')
        ax.set_ylabel('Count')
        ax.set_title('Sentiment Polarity Histogram')
        st.pyplot(fig)

        st.title("Sentiment Analysis by Person")
        person_sentiment_results = helper.sentiment_analysis_by_person(df)
        st.write("Sentiment Analysis by Person:")
        st.write(person_sentiment_results)

        most_negative_person = person_sentiment_results.sort_values(by='Negative Percentage', ascending=False).index[0]
        most_positive_person = person_sentiment_results.idxmax()['Positive Percentage']
        most_neutral_person = person_sentiment_results.idxmax()['Neutral Percentage']

        st.write("Most Negative Person:", most_negative_person)
        st.write("Most Positive Person:", most_positive_person)
        st.write("Most Neutral Person:", most_neutral_person)

        if selected_user in [most_negative_person, most_positive_person]:
            st.title(f"Words for {selected_user}")

            # Fetching sentiment polarity for the selected user
            user_sentiment = person_sentiment_results.loc[selected_user, 'Sentiment Polarity']

            # Filter messages for the selected user based on sentiment polarity
            user_messages = df[df['user'] == selected_user]['message']
            negative_messages = [message for message, sentiment in zip(user_messages, user_sentiment) if sentiment < 0]
            positive_messages = [message for message, sentiment in zip(user_messages, user_sentiment) if sentiment > 0]

            # Joining the messages to create strings
            negative_text = ' '.join(negative_messages)
            positive_text = ' '.join(positive_messages)

            # Displaying negative and positive word clouds
            negative_wordcloud = helper.create_wordcloud_for_sentiment(user_sentiment, negative_text, 'negative')
            positive_wordcloud = helper.create_wordcloud_for_sentiment(user_sentiment, positive_text, 'positive')

            st.title("Word Cloud for Negative Statements")
            st.image(negative_wordcloud.to_array(), use_column_width=True)

            st.title("Word Cloud for Positive Statements")
            st.image(positive_wordcloud.to_array(), use_column_width=True)

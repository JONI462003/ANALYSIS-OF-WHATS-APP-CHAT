from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_message = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_message, len(links)


def most_busy_users(df):
    df_filtered = df[df['user']!='group_notification']
    x=df_filtered['user'].value_counts().head()

    df_percentage = (round((df_filtered['user'].value_counts() / df_filtered.shape[0]) * 100, 2)
                     .reset_index().rename(columns={'index': 'percent', 'user': 'name'}))
    return x, df_percentage

def create_wordcloud(selected_user, df):
    f = open("C:/Users/Joni/OneDrive/Desktop/JNTUK-UCEN/COLLEGE FILES/stop_words.txt", "r")
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media ommited>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(df['message'].str.cat(sep=' '))
    return df_wc


def most_common_words(df, selected_user):
    f = open("C:/Users/Joni/OneDrive/Desktop/JNTUK-UCEN/COLLEGE FILES/stop_words.txt", "r")
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df=df[df['user'] != 'group_notification']

    emojis = []
    for message in df['message']:
        for c in message:
            if emoji.is_emoji(c):
                emojis.append(c)

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df=df[df['user'] != 'group_notification']

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['user'] != 'group_notification']

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df=df[df['user'] != 'group_notification']

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df=df[df['user'] != 'group_notification']

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df=df[df['user'] != 'group_notification']

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def sentiment_analysis(df):
    sentiments = []
    for message in df['message']:
        blob = TextBlob(message)
        sentiments.append(blob.sentiment.polarity)
    # Count positive, negative, and neutral sentiments
    positive_count = sum(1 for sentiment in sentiments if sentiment > 0)
    negative_count = sum(1 for sentiment in sentiments if sentiment < 0)
    neutral_count = sum(1 for sentiment in sentiments if sentiment == 0)

    # Calculate percentages
    total_count = len(sentiments)
    positive_percentage = (positive_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100
    neutral_percentage = (neutral_count / total_count) * 100

    return {
        'Positive': positive_count,
        'Negative': negative_count,
        'Neutral': neutral_count,
        'Positive Percentage': positive_percentage,
        'Negative Percentage': negative_percentage,
        'Neutral Percentage': neutral_percentage,
        'Sentiment Polarity': sentiments
    }

def filter_messages_by_sentiment(sentiments, messages, polarity):
    filtered_messages = []
    for sentiment, message in zip(sentiments, messages):
        if (sentiment > 0 and polarity == 'positive') \
            or (sentiment < 0 and polarity == 'negative') \
            or (sentiment == 0 and polarity == 'neutral'):
            filtered_messages.append(message)
    return filtered_messages


def sentiment_analysis_by_person(df):
    unique_users = df['user'].unique()
    sentiment_results = []
    for user in unique_users:
        user_df = df[df['user'] == user]
        sentiment_result = sentiment_analysis(user_df)
        sentiment_results.append(sentiment_result)
    sentiment_df = pd.DataFrame(sentiment_results, index=unique_users)
    return sentiment_df

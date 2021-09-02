

import pandas as pd
import tweepy


class TweetManager(object):

    def __init__(self):
       
        auth = tweepy.OAuthHandler('9uc7undki6H4jjAIwFcG7qqiw','cENjaCa39AAhidb25x0KnN8qF12b00QOhLSgurUQAcPJVB4HMe')
        auth.set_access_token('724386088039624704-NYfyrqHov5feDoamtrO1XqqdvnAzHBR','AfXRYw6g2NC0yKiED9a1EgmPBNrFZrwUF6h8lqDn56OTA')
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)

    def get_tweets(self, query, result_type, count, lang='ar'):
        tweets = tweepy.Cursor(self.twitter_api.search, q=query, count=count, lang=lang, result_type=result_type)

        data = [[tweet.created_at, tweet.text] for tweet in tweets.items(count)]

        return pd.DataFrame(data, columns=['created_at', 'tweet'])


def main():
    """
    To test the classifier
    """
    df = TweetManager().get_tweets_dummy('corona', count=100, result_type='popular')
    df

if __name__ == '__main__':
    main()

"""
This script takes all tweets in the dataset and builds a tsv file data source.
"""

import os
import json
import re


def clean_tweet(tweet_text):
    """
    This function 'cleans' the background noise a tweet has that disturb learning process,
    such as a link to data,
    or noise that disturb the representation such as \n.
    :param tweet_text: the original tweet
    :return: a 'clean' tweet.
    """
    return re.sub(r"(?:\@|https?\://)\S+", "", tweet_text).replace("\t", "  ").replace("\n", "  ").rstrip()


path = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(path, "tweets.tsv")
path = os.path.join(path, "pheme-rnr-dataset")
themes = ("charliehebdo", "ferguson", "germanwings-crash", "ottawashooting", "sydneysiege")
labels = (("non-rumours", 1), ("rumours", 0))
with open(output_file, "w") as out_file:
    out_file.truncate(0)
    out_file.write("theme\tlabel\tid\ttweet_text\ttweet_author\n")
    for theme in themes:
        for label, label_int in labels:
            directory = os.path.join(path, theme, label)
            all_directories = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
            for id_dir in all_directories:
                directory_with_id = os.path.join(directory, id_dir, "source-tweet")
                with open(os.path.join(directory_with_id, id_dir + ".json")) as tweet_file:
                    tweet = json.loads(tweet_file.read())
                    tweet_text = clean_tweet(tweet["text"])
                    tweet_author = tweet["user"]["name"]
                    line = "{theme}\t{label}\t{id}\t{tweet_text}\t{tweet_author}\n".format(
                        theme=theme,
                        label=label_int,
                        id=id_dir,
                        tweet_text=tweet_text,
                        tweet_author=tweet_author
                    )
                    out_file.write(line)



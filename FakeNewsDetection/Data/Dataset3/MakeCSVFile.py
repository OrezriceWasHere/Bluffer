import os
import json
import re


def clean_tweet_url(tweet_text_with_link):
    return re.sub(r"(?:\@|https?\://)\S+", "", tweet_text_with_link)

path = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(path, "output.csv")
path = os.path.join(path, "pheme-rnr-dataset")
themes = ("charliehebdo", "ferguson", "germanwings-crash", "ottawashooting", "sydneysiege")
labels = ("non-rumours", "rumours")
with open(output_file, "w") as out_file:
    out_file.truncate(0)
    out_file.write("theme, label, id, tweet_text, tweet_author,\n")
    for theme in themes:
        for label in labels:
            label_int = 0 if label == "rumours" else 1
            directory = os.path.join(path, theme, label)
            for id_dir in [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]:
                directory_with_id = os.path.join(directory, id_dir, "source-tweet")
                with open(os.path.join(directory_with_id, id_dir + ".json")) as tweet_file:
                    tweet = json.loads(tweet_file.read())
                    tweet_text = clean_tweet_url(tweet["text"])
                    tweet_author = tweet["user"]["name"]
                    line = "{theme}, {label}, {id}, {tweet_text}, {tweet_author},\n".format(
                        theme=theme,
                        label=label_int,
                        id=id_dir,
                        tweet_text=tweet_text,
                        tweet_author=tweet_author
                    )
                    out_file.write(line)



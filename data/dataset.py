import requests
import os

url = "https://www.gutenberg.org/cache/epub/29345/pg29345.txt"

response = requests.get(url)
if response.status_code == 200:
    with open(os.path.join(os.curdir, "data/raw/downloaded_text.txt"), "wb") as file:
        file.write(response.content)
    print("text file downloaded and saved successfully!")
else:
    print("failed to download the file. status code: {}".format(response.status_code))
# from deep_translator import GoogleTranslator
# from googletrans import Translator
# from google.cloud import translate_v2 as translate
from easynmt import EasyNMT
import json
import os

INPUT_FOLDER_PATH = "./data/raw_data/"
OUTPUT_FOLDER_PATH = "./data/translated_data/"

# translator = GoogleTranslator(source="auto", target="en")
# translator = Translator()
# translator = translate.Client()
# select the model to use
model = EasyNMT("opus-mt")

# for each folder in the input folder
for filename in os.listdir(INPUT_FOLDER_PATH):
    # get the file
    file = open(f"{INPUT_FOLDER_PATH}/{filename}", "r", encoding="utf-8")
    # read the data
    data = json.loads(file.read())

    for x, item_x in enumerate(data):
        for y, item_y in enumerate(item_x):
            try:
                # for each conversation, translate the text
                translation = model.translate(
                    item_y, source_lang="zh", target_lang="en"
                )
                # print(f"translated to {translation}")
                # replace the text with the translated text
                data[x][y] = translation
            except Exception as e:
                # if there is any error, write it to log
                print(
                    f"\nproblem translating at {data[x][y]} at conversation number {x}"
                )

    new_file = open(
        f'{OUTPUT_FOLDER_PATH}/{filename.replace(".json","")}-translated.json',
        "w",
        encoding="utf-8",
    )
    # write the new data to the output folder
    json.dump(data, new_file, indent=4, ensure_ascii=False)
    file.close()
    new_file.close()

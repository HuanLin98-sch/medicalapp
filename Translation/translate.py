# rom deep_translator import GoogleTranslator
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
model = EasyNMT("opus-mt")

for filename in os.listdir(INPUT_FOLDER_PATH):
    file = open(f"{INPUT_FOLDER_PATH}/{filename}", "r", encoding="utf-8")
    data = json.loads(file.read())
    # try:
    #     for x, item_x in enumerate(data):
    #         for y, item_y in enumerate(item_x):
    #             if x == 3000:
    #                 raise Exception("Translated 3000 sentences, now stopping")
    #             print(f'Translating {item_y}')
    #             # translation = translator.translate(item_y)
    #             translation = model.translate(item_y, source_lang="zh", target_lang="en")
    #             print(f'Translated to {translation}\n')
    #             data[x][y] = translation
    # except Exception as e:
    #     print(f'\nstopped translating at {data[x][y]}')
    #     print(f'ERROR = {e}')

    for x, item_x in enumerate(data):
        for y, item_y in enumerate(item_x):
            try:
                translation = model.translate(
                    item_y, source_lang="zh", target_lang="en"
                )
                print(f"translated to {translation}")
                data[x][y] = translation
            except Exception as e:
                print(
                    f"\nproblem translating at {data[x][y]} at conversation number {x}"
                )
    new_file = open(
        f'{OUTPUT_FOLDER_PATH}/{filename.replace(".json","")}-translated.json',
        "w",
        encoding="utf-8",
    )
    json.dump(data, new_file, indent=4, ensure_ascii=False)
    file.close()
    new_file.close()
    # print(f'size of {filename} = {len(data)}')

# rom deep_translator import GoogleTranslator
# from googletrans import Translator
# from google.cloud import translate_v2 as translate
from easynmt import EasyNMT
import json
import os
import pandas as pd
import numpy as np
import spacy

# translator = GoogleTranslator(source="auto", target="en")
# translator = Translator()
# translator = translate.Client()
# model = EasyNMT("opus-mt")

# df = pd.read_csv("./data/raw_data/data.csv", encoding="utf-8")
# df = df.drop(columns=["id", "type", "subtype", "description"])
# df.to_csv(
#     "./data/raw_data/data_processed.csv", index=False, encoding="utf_8_sig"
# )

# df = pd.read_csv(
#     "./data/raw_data/data-mbart.csv",
#     encoding="utf-8",
#     names=["zh", "eng", "opus-mt", "mbart50", "m2m"],
# )

# df[["opus-mt", "mbart50", "m2m"]] = df[["opus-mt", "mbart50", "m2m"]].astype(
#     str
# )

# print(df.head())

# for index, row in df.iterrows():
#     translation = model.translate(
#         row["zh"], source_lang="zh", target_lang="en"
#     )
#     if index % 1000 == 0:
#         print(index)
#     df.at[index, "opus-mt"] = translation

# df.to_csv("./data/raw_data/data-opus.csv", encoding="utf_8_sig", index=False)


# model = EasyNMT("mbart50_m2m")

# for index, row in df.iterrows():
#     translation = model.translate(
#         row["zh"], source_lang="zh", target_lang="en"
#     )
#     if index % 1000 == 0:
#         print(index)
#     df.at[index, "mbart50"] = translation

# df.to_csv("./data/raw_data/data-mbart.csv", encoding="utf_8_sig", index=False)


# model = EasyNMT("m2m_100_418M")

# for index, row in df.iterrows():
#     translation = model.translate(
#         row["zh"], source_lang="zh", target_lang="en"
#     )
#     if index % 1000 == 0:
#         print(index)
#     df.at[index, "m2m"] = translation

# df.to_csv(
#     "./data/raw_data/data-translated.csv", encoding="utf_8_sig", index=False
# )

nlp = spacy.load("en_core_web_lg")

df = pd.read_csv(
    "./data/raw_data/data-translated.csv",
    encoding="utf-8",
    names=[
        "zh",
        "eng",
        "opus-mt",
        "mbart50",
        "m2m",
        "opus-mt-sim",
        "mbart50-sim",
        "m2m-sim",
    ],
)

df[["opus-mt-sim", "mbart50-sim", "m2m-sim"]] = df[
    ["opus-mt-sim", "mbart50-sim", "m2m-sim"]
].astype(str)

for index, row in df.iterrows():
    try:
        text = nlp(row["eng"])
        # print(f'text = {text}')
        opus_text = nlp(row["opus-mt"])
        # print(f'opus text = {opus_text}')
        mbart_text = nlp(row["mbart50"])
        # print(f'mbart text = {mbart_text}')
        m2m_text = nlp(row["m2m"])
        # print(f'm2m text = {m2m_text}')
        opus_sim = text.similarity(opus_text)
        # print(f'opus sim = {opus_sim}')
        mbart_sim = text.similarity(mbart_text)
        # print(f'mbart sim = {mbart_sim}')
        m2m_sim = text.similarity(m2m_text)
        # print(f'm2m sim = {m2m_sim}')
        df.at[index, "opus-mt-sim"] = opus_sim
        df.at[index, "mbart50-sim"] = mbart_sim
        df.at[index, "m2m-sim"] = m2m_sim
    except:
        print(f'error occured at {row["eng"]}')

df.to_csv(
    "./data/translated_data/data-evaluated.csv",
    encoding="utf_8_sig",
    index=False,
)

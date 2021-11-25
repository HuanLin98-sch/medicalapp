# rom deep_translator import GoogleTranslator
# from googletrans import Translator
# from google.cloud import translate_v2 as translate
from easynmt import EasyNMT
import json
import os
import pandas as pd
import numpy as np
import spacy

## Step 1
## read the original dictionary
# df = pd.read_csv("./data/raw_data/data.csv", encoding="utf-8")
## drop the redundant columns
# df = df.drop(columns=["id", "type", "subtype", "description"])
## write to new csv file
# df.to_csv(
#     "./data/raw_data/data_processed.csv", index=False, encoding="utf_8_sig"
# )

## read the csv file (read the appropriate file for the model)
# df = pd.read_csv(
#     "./data/raw_data/data.csv",
#     encoding="utf-8",
#     names=["zh", "eng", "opus-mt", "mbart50", "m2m"],
# )

## create new columns (only run this part of the code once)
# df[["opus-mt", "mbart50", "m2m"]] = df[["opus-mt", "mbart50", "m2m"]].astype(
#     str
# )

# print(df.head())

## Step 2
## select model opus-mt
# model = EasyNMT("opus-mt")

## for each row, translate the text
# for index, row in df.iterrows():
#     translation = model.translate(
#         row["zh"], source_lang="zh", target_lang="en"
#     )
#     if index % 1000 == 0:
#         print(index)
#     df.at[index, "opus-mt"] = translation

# df.to_csv("./data/raw_data/data-opus.csv", encoding="utf_8_sig", index=False)

## Step 3
## select model mbart50
# model = EasyNMT("mbart50_m2m")

## for each row, translate the text
# for index, row in df.iterrows():
#     translation = model.translate(
#         row["zh"], source_lang="zh", target_lang="en"
#     )
#     if index % 1000 == 0:
#         print(index)
#     df.at[index, "mbart50"] = translation

# df.to_csv("./data/raw_data/data-mbart.csv", encoding="utf_8_sig", index=False)

## Step 4
## select model m2m
# model = EasyNMT("m2m_100_418M")

## for each row, translate the text
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

# Step 5
# load the spacy model
nlp = spacy.load("en_core_web_lg")

# read the csv file
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

# create new columns
df[["opus-mt-sim", "mbart50-sim", "m2m-sim"]] = df[
    ["opus-mt-sim", "mbart50-sim", "m2m-sim"]
].astype(str)

# for each row, calculate the vector similarity between the translated text and the original text
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

# write to new csv file
df.to_csv(
    "./data/translated_data/data-evaluated.csv",
    encoding="utf_8_sig",
    index=False,
)

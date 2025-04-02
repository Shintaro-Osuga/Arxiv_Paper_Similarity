import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 
import torch.functional as f
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict
from tqdm import tqdm

import time

from sentence_transformers import SentenceTransformer

def load_data(file:str) -> pd.DataFrame:
    return pd.read_csv(file)


def get_stopwords(one_time_stopwords:list[str]=None, custom:bool=False) -> list[str]:
    from nltk.corpus import stopwords

    stpwrds = stopwords.words('english')
    new_words = []
    for word in new_words:
        stpwrds.append(word)

    
    if not one_time_stopwords == None:
        for word in one_time_stopwords:
            stpwrds.append(word)
            
    return stpwrds

def make_wordcloud(comments, stopwords = None, title:str='token'):
    import wordcloud

    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    if stopwords == None:
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white', min_font_size=10,
                              max_words=100).generate(' '.join(comments))
    else:
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white', min_font_size=10,
                              max_words=100, stopwords=stopwords).generate(' '.join(comments))
        
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title+" WordCloud")
    plt.axis("off")
    plt.show()

def make_hist(col_vc, threshold=None):
    alpha = threshold *col_vc.sum()
    print(alpha)
    vc = vc[vc>alpha]
    print(col_vc)
    col_vc.plot(kind='barh', title="Histogram of Categories", color='skyblue', edgecolor='black')
    plt.xticks(rotation=45) 
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    

def eda(df):
    # make_wordcloud(df["title"], title='Title')
    
    # make_wordcloud(df["summary"], title="Summary")
    
    vc = df["category"].value_counts()
    med = vc.median()
    alpha = 0.0075 *vc.sum()
    print(alpha)
    vc = vc[vc>alpha]
    print(vc)
    vc.plot(kind='barh', title="Histogram of Categories", color='skyblue', edgecolor='black')
    plt.xticks(rotation=45) 
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # vc = df["first_author"].value_counts()
    # med = vc.median()
    # alpha = 0.0002 *vc.sum()
    # # print(alpha)
    # vc = vc[vc>alpha]
    # print(vc)
    # vc.plot(kind='barh', title="Histogram of Authors", color='skyblue', edgecolor='black')
    # plt.xticks(rotation=45) 
    # plt.xlabel('Categories')
    # plt.ylabel('Counts')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

def make_heatmap(data:list[list[int]], title:str):
    
    plt.imshow(data, cmap='viridis', interpolation='nearest')

    # Add a colorbar for reference
    plt.colorbar()

    # Customize the plot (optional)
    plt.title('Heatmap of '+title)
    plt.xlabel('Columns of Summaries')
    plt.ylabel('Rows of Summaries')

    # Show the plot
    plt.show()

def get_top_thresh(alpha:float,
                   input:str,
                   input_embeds:pd.DataFrame):
    similarities = get_similarities(input, input_embeds)
    
    above_alpha = similarities[similarities['similarity'] > alpha]
    
    print(f"{len(above_alpha)} articles are similar to {input} with an alpha of {alpha}")

def get_summary_similarities(top_n:int, 
                             input_cat:str, 
                             input_date:str, 
                             cat_embeds:pd.DataFrame, 
                             date_embeds:pd.DataFrame,
                             sum_embeds:pd.DataFrame):
    from sklearn.metrics.pairwise import cosine_similarity
    
    combined_similarities = get_similarities(input_cat, cat_embeds, return_raw=True) + get_similarities(input_date, date_embeds, return_raw=True)
    # combined_similarities['sum_embeds'] = sum_embeds
    combined_similarities.sort_values('similarity', inplace=True, ascending=False)
    
    top_sums = sum_embeds.iloc[combined_similarities.iloc[0:top_n].index]
    
    sum_sims = []
    # print(top_sums)
    total_sim = 0
    lowest_sim = 1
    lowest_sim_idx = (0,0)
    highest_sim = 0
    highest_sim_idx = (0,0)
    second_hi = 0
    second_hi_idx = (0,0)
    for x in range(len(top_sums)):
        temp_sims = []
        for y in range(len(top_sums)):
            sim = cosine_similarity(top_sums.iloc[x, :].values.reshape(1,-1), top_sums.iloc[y, :].values.reshape(1,-1))[0][0]
            temp_sims.append(sim)
            total_sim += sim
            if x != y:
                if sim < lowest_sim:
                    lowest_sim = sim
                    # print(top_sums.iloc[x].name)
                    # print(top_sums.iloc[y].name)
                    # print(top_sums.iloc[x])
                    lowest_sim_idx = [top_sums.iloc[x].name, top_sums.iloc[y].name]
                if sim > highest_sim:
                    highest_sim = sim
                    highest_sim_idx = [top_sums.iloc[x].name, top_sums.iloc[y].name]
                    
                if sim > second_hi and sim < highest_sim:
                    second_hi = sim
                    second_hi_idx = [top_sums.iloc[x].name, top_sums.iloc[y].name]
                    
        sum_sims.append(temp_sims)
    
    print(f"The lowest Similarity score was {lowest_sim}\nThe highest Similarity score was {highest_sim} \nSecond highest Similarity score was {second_hi} \nThe average Similarity score was {total_sim/top_n**2}")
    return sum_sims, lowest_sim_idx, highest_sim_idx, second_hi_idx, total_sim/top_n**2

def get_similarities(input:str, embeddings:pd.DataFrame, return_raw:bool=False):
    from sklearn.metrics.pairwise import cosine_similarity

    MODEL_NAME = 'all-mpnet-base-v2'
    transformer = SentenceTransformer(MODEL_NAME)
    input_embed = transformer.encode(input, batch_size=1, device='cuda:0')
    
    similarities = [sim[0] for sim in cosine_similarity(embeddings, input_embed.reshape(1,-1))]
    
    sim_pd = pd.DataFrame({'similarity': similarities}, index=embeddings.index)
    
    if return_raw:
        return sim_pd
    
    sim_pd.sort_values('similarity', inplace=True, ascending=False)
    return sim_pd
    
def embed_text(col):
    MODEL_NAME = 'all-mpnet-base-v2'
    transformer = SentenceTransformer(MODEL_NAME)
    
    embeds = transformer.encode(col, batch_size=64, device='cuda:0')
    ncols = len(embeds[0])
    attnames = [f'F{i}' for i in range(ncols)]
    return pd.DataFrame(embeds, columns=attnames)


def main():
    dir_path = "~/Documents/Syracuse_MS/Winter_2025/IST_736/Project/archive/" 
    train_path = dir_path+"arXiv_scientific dataset.csv"

    df = load_data(train_path)
    # df['published_datetime'] = pd.to_datetime(df['published_date'], format='%m/%d/%y')
    # plt.hist(df['published_datetime'], bins=50, rwidth=0.9)  # Adjust bins as needed
    # plt.xlabel('Date')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Dates')
    # plt.show()
    # print(df['summary_word_count'].mean())
    
    catem_dir = dir_path+'cat_embeddings.csv'
    cat_embed = pd.read_csv(catem_dir)
    datem_dir = dir_path+'pub_date_embeddings.csv'
    date_embed = pd.read_csv(datem_dir)
    sumem_dir = dir_path+'summary_embeddings.csv'
    sum_embed = pd.read_csv(sumem_dir)
    
    # similiarities = get_similarities("Artificial non-Intelligence", cat_embed)
    # print(similiarities.head(10))
    all_avgs = []
    years = df['published_date'].str[-2:].unique()
    print(years)
    for year in years:
        print(year)
        sum_sims, lowest_sim_idx, highest_sim_idx, second_hi, avg = get_summary_similarities(top_n=100, input_cat="Deep Learning Optimizations",
                                                                            input_date="11/13/"+year, cat_embeds=cat_embed, 
                                                                            date_embeds=date_embed, sum_embeds=sum_embed)
        print(f"Lowest Similarity article titles \n1: {df.iloc[lowest_sim_idx[0]]} \n2: {df.iloc[lowest_sim_idx[1]]}")
        
        print(f"Highest Similarity article titles \n1: {df.iloc[highest_sim_idx[0]]} \n2: {df.iloc[highest_sim_idx[1]]}")
        
        print(f"Second Highest Similarity article titles \n1: {df.iloc[second_hi[0]]} \n2: {df.iloc[second_hi[1]]}")
        all_avgs.append(avg)
    
    plt.plot(years, all_avgs)
    plt.xlabel("Year")
    plt.ylabel("Average Score")
    plt.title("Average Score Over Years")
    plt.show()
    
    make_heatmap(sum_sims, title='Summary')
    # print(df.iloc[107098]['title'])
    # print(df.iloc[104735]['title'])
    # print(df.iloc[14427]['title'])
    # print(df.iloc[14426]['title'])
    # print(df.iloc[104543]['title'])
    # print(df.iloc[107097]['title'])
    # get_top_thresh(0.45, "Deep Learning Optimizations", cat_embed)
    # get_top_thresh(0.99, "11/13/17", date_embed)
    
    
    # ---------- MAKE EMBEDDING FILES ----------------
    # df = pd.read_pickle(dir_path+"cat_embeddings.pkl")
    # df.to_csv(f"{dir_path}/cat_embeddings.csv", index=False)
    
    # eda(df)
    
    # out = embed_text(df['category'])
    # # out.to_pickle(f'{dir_path}/embeddings.pkl')
    # out.to_csv(f"{dir_path}/cat_embeddings.csv", index=False)
    
    # out = embed_text(df['published_date'])
    # # out.to_pickle(f'{dir_path}/embeddings.pkl')
    # out.to_csv(f"{dir_path}/pub_date_embeddings.csv", index=False)
    
    # out = embed_text(df['title'])
    # # out.to_pickle(f'{dir_path}/embeddings.pkl')
    # out.to_csv(f"{dir_path}/title_embeddings.csv", index=False)
    # # print(out)
    
    # out = embed_text(df['summary'])
    # # out.to_pickle(f'{dir_path}/embeddings.pkl')
    # out.to_csv(f"{dir_path}/summary_embeddings.csv", index=False)
    
if __name__ == "__main__":
    sys.exit(main())
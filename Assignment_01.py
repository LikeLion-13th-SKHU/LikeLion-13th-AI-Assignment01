import pandas as pd
import re
import konlpy
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from tqdm import tqdm
import urllib.request

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-2, 2))

df = pd.read_csv('text_data.txt', header=None, names=['sentence'])

sent_dic = pd.read_csv('SentiWord_Dict.txt',sep = '\t',header=None)


class Aurora3:

    def __init__(self, df,sent_dic):
        self.df = df
        self.okt = konlpy.tag.Okt()
        self.sent_dic = sent_dic

    def get_df(self):# 최종 결과 반환
        print("문장을 토큰화 중입니다...")
        self.tokenizer_run()

        print("감성사전을 업데이트 중입니다...")
        self.expand_sent_dic()

        print("문장 감성분석 중입니다....")
        self.sent_analyze()
        return self.df

    def tokenizer_run(self): # 텍스트 전처리 & 토큰화
        tqdm.pandas()

        def text_preprocess(x):
            text=[]
            a = re.sub('[^가-힣0-9a-zA-Z\\s]', '',x)
            for j in a.split():
                text.append(j)
            return ' '.join(text)

        def tokenize(x):
            text = []
            tokens = self.okt.pos(x)
            for token in tokens :
                if token[1] == 'Adjective' or token[1]=='Adverb' or token[1] == 'Determiner' or token[1] == 'Noun' or token[1] == 'Verb' or 'Unknown':
                    text.append(token[0])
            return text
        self.df['comment'] = self.df['sentence'].apply(lambda x : text_preprocess(x))
        self.df['comment'] = self.df['comment'].progress_apply(lambda x: tokenize(x))

    def expand_sent_dic(self):
        sent_dic = self.sent_dic

        def make_sent_dict(x) :
            pos=[]
            neg=[]
            tmp={}

            for sentence in tqdm(x):
                for word in sentence :
                    target = sent_dic[sent_dic[0]==word]
                    if len(target)==1: # 기존에 있는 단어라면 그대로 사용
                        score = float(target[1])
                        if score > 0:
                            pos.append(word)
                        elif score < 0:
                            neg.append(word)
                    tmp[word] = {'W':0,'WP':0,'WN':0} # 감성사전 구성
            pos = list(set(pos))
            neg = list(set(neg))

            for sentence in tqdm(x):
                for word in sentence :
                    tmp[word]['W'] += 1 # 빈도 수
                    for po in pos :
                        if po in sentence:
                            tmp[word]['WP'] += 1 # 긍정단어과 같은 문장 내 단어일 때
                            break
                    for ne in neg:
                        if ne in sentence:
                            tmp[word]['WN'] += 1 # 부정단어와 같은 문장내 단어일 때
                            break
            return pos, neg, pd.DataFrame(tmp)

        def make_score_dict(d,p,n):
            N=sum(d.iloc[0,::])
            pos_cnt=sum(d.loc[::,p].iloc[0,::])
            neg_cnt=sum(d.loc[::,n].iloc[0,::])

            trans =d.T
            trans['neg_cnt']=neg_cnt
            trans['pos_cnt']=pos_cnt
            trans['N']=N

            trans['MI_P']=np.log2(trans['WP']*trans['N']/trans['W']*trans['pos_cnt'])
            trans['MI_N']=np.log2(trans['WN']*trans['N']/trans['W']*trans['neg_cnt'])
            trans['SO_MI']=trans['MI_P'] - trans['MI_N']

            trans = trans.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
            trans = trans.sort_values(by=['SO_MI'],ascending=False)
            return trans

        def update_dict(d):
            add_Dic = {0:[],1:[]}
            for i in d.T.items():
                if i[0] not in list(sent_dic[0]):
                    if len(i[0]) > 1:
                        add_Dic[0].append(i[0])
                        add_Dic[1].append(i[1]['SO_MI'])

            add_Dic=pd.DataFrame(add_Dic)
            Sentiment=pd.merge(sent_dic,add_Dic,'outer')
            return Sentiment

        self.pos, self.neg, self.new_dict = make_sent_dict(self.df['comment'].values)

        self.t_dict = make_score_dict(self.new_dict,self.pos,self.neg)
        self.t_dict['SO_MI'] = scaler.fit_transform(self.t_dict['SO_MI'].values.reshape(-1,1))

        self.add_dict =update_dict(self.t_dict)

    def sent_analyze(self): # 데이터 감성분석
        tqdm.pandas()

        def get_cnt(x):
            cnt = 0
            for word in list(set(x)):
                target = self.add_dict[self.add_dict[0]==word]
                if len(target)==1:
                    cnt += float(target[1])
            return cnt

        def get_ratio(x):
            score = x['score']
            length = np.log10(len(x['comment']))+1
            try:
                ratio= round(score/length,2)
            except:
                ratio = 0
            return ratio

        tqdm.pandas()
        self.df['score']= self.df['comment'].progress_apply(lambda x : get_cnt(x))
        self.df['ratio'] = self.df.apply(lambda x: get_ratio(x), axis = 1)


test = Aurora3(df, sent_dic)
res = test.get_df()
print(res.head(10))
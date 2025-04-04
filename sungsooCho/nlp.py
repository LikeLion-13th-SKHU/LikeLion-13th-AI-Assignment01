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

# 1. 텍스트 데이터를 불러와 필요한 부분만 남기고, 열 이름을 변경하여 감성 분석 작업에 적합한 형태로 데이터를 준비하는 과정입니다.

df = pd.read_table('word.txt') # 변수 df에 데이터 프레임 저장
df = df.iloc[:10000,:] # 첫 번째부터 만 번째 행까지, 열은 모두 포함
df = df[['id','document']] # id, docu 두 열만 선택
df = df.rename(columns={"id": "id", "document": "sentence"}) # column 이름 변경
df

# 2. 감성 분석을 위해 사전에 준비된 단어 사전을 불러오고, 단어들을 점수에 따라 세 부류로 나눈다. 이는 후속 작업에서 필요합니다..
sent_dic = pd.read_csv('SentiWord_Dict.txt',sep = '\t',header=None)
sent_dic.iloc[14850,0]='갈등'



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


# 3. 텍스트 전처리를 통해 특수 문자 등을 지우고 단어 단위로 배열에 저장한 다음 공백을 기준으로 단어를 연결하여 하나의 문자열 반환합니다.
#    그리고 okt.pos()를 통해 입력 된 문장을 형태소 단위로 분리한 다음 단어와 품사 정보를 제공, 단어의 품사가 조건을 만족하면 해당 단어는 배열에 추가합니다.
#    즉, 어절 단위로 바꾸고, 중요한 단어만 배열에 추가하는 것입니다.

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

# 5. 입려된 단어들을 긍정과 부정 두 가지 부류로 나눕니다.
#    주어진 토큰 문장을 순회하며 빈도수와 긍정 또는 부정으로 몇 번 쓰였는지를 기록하는 데이터 프레임을 생성합니다.
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
        
# 6. 단어별로 긍정적인지 부정적인지 점수를 나타냅니다.
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
        
# 7. 기존 감성 사전에 없는 단어와 감성 점수를 추가합니다. 확장된 감성 사전을 만드는 것이지요.
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

# 8. 확장된 감성 사전을 이용하여 문장 단위로 감성 점수를 합산합니다.
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


test = Aurora3(df,sent_dic)
res = test.get_df()


# print(df['comment'])
print(res)
# print(test.t_dict[['MI_P', 'MI_N', 'SO_MI']])


res[['sentence', 'score', 'ratio']].to_csv(
    'result.txt',      
    sep='\t',          
    index=False,       
    header=True,       
    encoding='utf-8-sig'
)



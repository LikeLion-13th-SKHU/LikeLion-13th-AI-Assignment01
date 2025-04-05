import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from konlpy.tag import Okt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

class Aurora3:
    def __init__(self, df, sent_dic):
        # 초기화: 데이터프레임과 감성 사전, 형태소 분석기 설정
        self.df = df
        self.okt = Okt()
        self.sent_dic = sent_dic

    def get_df(self):
        # 전체 분석 프로세스 실행
        print("문장을 토큰화 중입니다...")
        self.tokenizer_run()

        print("감성사전을 업데이트 중입니다...")
        self.expand_sent_dic()

        print("문장 감성분석 중입니다....")
        self.sent_analyze()

        return self.df

    def tokenizer_run(self):
        # 문장을 정제하고 형태소 분석기로 토큰화
        tqdm.pandas()

        def text_preprocess(x):
            # 특수 문자 제거 및 공백 정리
            a = re.sub('[^가-힣0-9a-zA-Z\\s]', '', x)
            return ' '.join(a.split())

        def tokenize(x):
            # 명사, 형용사, 동사, 부사 등을 추출
            return [w for w, t in self.okt.pos(x) if t in ['Adjective', 'Adverb', 'Determiner', 'Noun', 'Verb']]

        self.df['comment'] = self.df['sentence'].apply(text_preprocess)
        self.df['comment'] = self.df['comment'].progress_apply(tokenize)

    def expand_sent_dic(self):
        # 문장에서 단어 빈도 및 공존 빈도 기반으로 감성사전 확장
        sent_dic = self.sent_dic

        def make_sent_dict(x):
            # 단어별 빈도 및 공존 빈도 계산
            pos, neg = [], []
            tmp = {}
            for sentence in tqdm(x):
                for word in sentence:
                    target = sent_dic[sent_dic[0] == word]
                    if len(target) == 1:
                        score = float(target.iloc[0, 1])
                        if score > 0:
                            pos.append(word)
                        elif score < 0:
                            neg.append(word)
                    tmp[word] = {'W': 0, 'WP': 0, 'WN': 0}
            pos, neg = list(set(pos)), list(set(neg))

            for sentence in tqdm(x):
                for word in sentence:
                    tmp[word]['W'] += 1
                    if any(po in sentence for po in pos):
                        tmp[word]['WP'] += 1
                    if any(ne in sentence for ne in neg):
                        tmp[word]['WN'] += 1

            # 데이터프레임으로 전환
            df_tmp = pd.DataFrame.from_dict(tmp, orient='index').T
            return pos, neg, df_tmp

        def make_score_dict(d, p, n):
            # SO-PMI 점수 계산
            if d.empty:
                return pd.DataFrame()

            N = sum(d.loc['W', :])
            pos_cnt = sum(d.loc['WP', p])
            neg_cnt = sum(d.loc['WN', n])

            trans = d.T
            trans['pos_cnt'] = pos_cnt
            trans['neg_cnt'] = neg_cnt
            trans['N'] = N

            trans['MI_P'] = np.log2((trans['WP'] * trans['N']) / (trans['W'] * trans['pos_cnt'] + 1e-5))
            trans['MI_N'] = np.log2((trans['WN'] * trans['N']) / (trans['W'] * trans['neg_cnt'] + 1e-5))
            trans['SO_MI'] = trans['MI_P'] - trans['MI_N']
            trans = trans.replace([np.inf, -np.inf], np.nan).dropna()
            return trans.sort_values(by='SO_MI', ascending=False)

        def update_dict(d):
            # SO_MI 기반으로 새로운 단어를 감성사전에 추가
            add_Dic = {0: [], 1: []}
            for word, row in d.iterrows():
                if word not in list(sent_dic[0]) and len(word) > 1:
                    add_Dic[0].append(word)
                    add_Dic[1].append(row['SO_MI'])
            return pd.concat([sent_dic, pd.DataFrame(add_Dic)], ignore_index=True)

        # 전체 처리 실행
        self.pos, self.neg, self.new_dict = make_sent_dict(self.df['comment'].values)
        self.t_dict = make_score_dict(self.new_dict, self.pos, self.neg)

        if not self.t_dict.empty and 'SO_MI' in self.t_dict.columns and self.t_dict['SO_MI'].notna().any():
            self.t_dict['SO_MI'] = scaler.fit_transform(self.t_dict['SO_MI'].values.reshape(-1, 1))
        else:
            print("⚠️ SO_MI 값이 없어 정규화를 건너뜁니다.")
            self.t_dict['SO_MI'] = pd.Series(dtype='float64')

        self.add_dict = update_dict(self.t_dict)

    def sent_analyze(self):
        # 문장별 감성 점수 계산
        tqdm.pandas()

        def get_cnt(x):
            # 해당 문장의 단어 점수 총합
            return sum(float(self.add_dict[self.add_dict[0] == word].iloc[0, 1])
                       for word in set(x)
                       if word in list(self.add_dict[0]))

        def get_ratio(x):
            # 문장의 길이에 비례한 감성 비율 계산
            score = x['score']
            length = np.log10(len(x['comment']) + 1)
            return round(score / length, 2) if length > 0 else 0

        self.df['score'] = self.df['comment'].progress_apply(get_cnt)
        self.df['ratio'] = self.df.apply(get_ratio, axis=1)

# 1. 텍스트 데이터 로드 
df = pd.read_csv("text_data.txt", header=None, names=["sentence"])

# 2. 감성사전 로드
sent_dic = pd.read_csv("SentiWord_Dict.txt", sep="\t", header=None)
sent_dic.columns = [0, 1]  # 기존 코드 호환을 위해 컬럼명 변경
sent_dic[1] = pd.to_numeric(sent_dic[1], errors='coerce')  # 점수형 변환

# 3. 감성 분석 실행
model = Aurora3(df, sent_dic)
result_df = model.get_df()

# 4. 결과 출력 및 저장
print(result_df[['sentence', 'score', 'ratio']])
result_df[['sentence', 'score', 'ratio']].to_csv("result_sentiment.csv", index=False, encoding='utf-8-sig')

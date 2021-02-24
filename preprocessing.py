

import warnings
warnings.filterwarnings(action='ignore')

def PreprocessingHashtags(path):
    
    import pandas as pd
    import re
    
    # 데이터 불러오기
    data = pd.read_csv(path, index_col=0)
    
    # 해시태그에서 특수문자 제거 후 우물 정(#) 기준으로 분리
    p = re.compile(r'[가-힣# ]+')
    data['hashtags_splitted'] = data['hashtags'].apply(lambda x: ''.join(p.findall(str(x))).split('#'))
    
    # 빈 해시태그 제거
    data['hashtags_completed'] = ''
    for i in range(len(data)):
        ls = [word for word in data.iloc[i]['hashtags_splitted'] if word!='']
        data['hashtags_completed'].iloc[i] = ls

    # 컬럼 삭제
    data.drop(['hashtags', 'hashtags_splitted'], axis=1, inplace=True)
    
    # 컬럼명 변경
    data.rename({'hashtags_completed':'hashtags'}, axis=1, inplace=True)
    
    return data


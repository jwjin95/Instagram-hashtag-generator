def PreprocessingHashtags(path):

    """해시태그 전처리"""

    import pandas as pd
    import re

    # 데이터 불러오기
    data = pd.read_csv(path, index_col=0)

    # 해시태그에서 특수문자 제거 후 우물 정(#) 기준으로 분리
    p = re.compile(r'[가-힣# ]+')
    data['hashtags_cleaned'] = data['hashtags'].apply(lambda x: ''.join(p.findall(str(x))))
    data['hashtags_splitted'] = data['hashtags_cleaned'].apply(lambda x: x.split('#'))

    # 빈 해시태그 제거
    data['hashtags_completed'] = ''
    for i in range(len(data)):
        ls = [word for word in data.loc[i, 'hashtags_splitted'] if word != '']
        data.loc[i, 'hashtags_completed'] = ls

    # 장소를 해시태그에 추가
    l = data['location'].fillna('')
    for i in range(len(data)):
        result = re.compile(r'[가-힣]+').search(l[i])
        if result != None:
            data.loc[i, 'hashtags_completed'].append(result.group())

    # 컬럼 삭제
    data.drop(['hashtags', 'hashtags_cleaned', 'hashtags_splitted'], axis=1, inplace=True)

    # 컬럼명 변경
    data.rename({'hashtags_completed': 'hashtags'}, axis=1, inplace=True)

    return data

def DeleteLowFreqHashtags(data):

    """출현 빈도가 낮은(1회) 해시태그 제거"""

    import pandas as pd

    # 데이터 가져오기
    data = data

    # 출현빈도가 1회뿐인 해시태그 제거
    hashtags_list = []
    for x in range(len(data.hashtags)):
        for y in range(len(data.hashtags[x])):
            hashtags_list.append(data.hashtags[x][y])

    hashtags_set = set(hashtags_list)
    hashtags_count = [hashtags_list.count(i) for i in hashtags_set]
    hashtags_dict = dict(zip(hashtags_set, hashtags_count))

    hashtags_df = pd.DataFrame()
    hashtags_df['name'] = hashtags_dict.keys()
    hashtags_df['count'] = hashtags_dict.values()

    hashtags_df = hashtags_df[hashtags_df['count'] >= 2]
    hashtags_df = hashtags_df.reset_index().drop('index', axis=1)

    # 컬럼 변경 완료
    new_hashtags = []
    for x in range(len(data.hashtags)):
        temp = []

        for y in range(len(data.hashtags[x])):
            if data.hashtags[x][y] in hashtags_list:
                temp.append(data.hashtags[x][y])
        new_hashtags.append(temp)

    data['hashtags'] = new_hashtags

    return data
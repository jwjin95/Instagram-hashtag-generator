def NTokenConversion(model, t_list, alpha, n):

    """토큰 리스트에 대하여 토큰 전환"""

    res = []
    for t in t_list:
        similar_tokens = model.wv.most_similar(t[0])
        for i in range(len(similar_tokens)):
            if (alpha**n)*(similar_tokens[i][1])>=0.7:
                res.append(similar_tokens[i])

    return res


def TokenConversion(model, common_tokens, alpha):

    """공통 토큰 전체에 대하여 토큰 전환"""

    # 토큰 전환
    ls1 = NTokenConversion(model, common_tokens, alpha, n=0)
    ls2 = NTokenConversion(model, ls1, alpha, n=1)
    ls3 = NTokenConversion(model, ls2, alpha, n=2)
    ls4 = NTokenConversion(model, ls3, alpha, n=3)
    ls5 = NTokenConversion(model, ls4, alpha, n=4)
    ls = ls1 + ls2 + ls3 + ls4 + ls5

    # 중복 토큰 제거
    temp = []
    result = []
    for i in range(len(ls)):
        if ls[i][0] not in temp:
            temp.append(ls[i][0])
            result.append(ls[i])

    return result

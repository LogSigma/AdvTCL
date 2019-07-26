# AdvTCL
Advanced TCL

### Usages

    >>> from hangul_utils import split_syllable_char, split_syllables,
        join_jamos
    >>> print(split_syllable_char(u"안"))
    ('ㅇ', 'ㅏ', 'ㄴ')
    
    >>> print(split_syllables(u"안녕하세요"))
    ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ
    
    >>> sentence = u"앞 집 팥죽은 붉은 팥 풋팥죽이고, 뒷집 콩죽은 햇콩 단콩 콩죽.우리 집
        깨죽은 검은 깨 깨죽인데 사람들은 햇콩 단콩 콩죽 깨죽 죽먹기를 싫어하더라."
    >>> s = split_syllables(sentence)
    >>> print(s)
    ㅇㅏㅍ ㅈㅣㅂ ㅍㅏㅌㅈㅜㄱㅇㅡㄴ ㅂㅜㄺㅇㅡㄴ ㅍㅏㅌ ㅍㅜㅅㅍㅏㅌㅈㅜㄱㅇㅣㄱㅗ,
    ㄷㅟㅅㅈㅣㅂ ㅋㅗㅇㅈㅜㄱㅇㅡㄴ ㅎㅐㅅㅋㅗㅇ ㄷㅏㄴㅋㅗㅇ ㅋㅗㅇㅈㅜㄱ.ㅇㅜㄹㅣ
    ㅈㅣㅂ ㄲㅐㅈㅜㄱㅇㅡㄴ ㄱㅓㅁㅇㅡㄴ ㄲㅐ ㄲㅐㅈㅜㄱㅇㅣㄴㄷㅔ ㅅㅏㄹㅏㅁㄷㅡㄹㅇㅡㄴ
    ㅎㅐㅅㅋㅗㅇ ㄷㅏㄴㅋㅗㅇ ㅋㅗㅇㅈㅜㄱ ㄲㅐㅈㅜㄱ ㅈㅜㄱㅁㅓㄱㄱㅣㄹㅡㄹ
    ㅅㅣㅀㅇㅓㅎㅏㄷㅓㄹㅏ.
    
    >>> sentence2 = join_jamos(s)
    >>> print(sentence2)
    앞 집 팥죽은 붉은 팥 풋팥죽이고, 뒷집 콩죽은 햇콩 단콩 콩죽.우리 집 깨죽은 검은 깨
    깨죽인데 사람들은 햇콩 단콩 콩죽 깨죽 죽먹기를 싫어하더라.
    
    >>> print(sentence == sentence2)
    True

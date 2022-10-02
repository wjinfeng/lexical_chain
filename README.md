# Lexical Chain
提供中英文词汇链计算

## Related Source
- `nltk_data`, 请从 https://github.com/nltk/nltk_data 下载`nltk_data`，并放到指定位置
  ```
    >>> from nltk.corpora import wordnet as wn
    >>> wn.synsets(word)

    注意把nltk_data下package文件中所有子文件夹方法nltk下
  ```
  - `nltk.word_tokenize`, `nltk.pos_tag`, `nltk.corpus.wordnet`会使用到其中的资源
- 词林文件：https://github.com/ashengtx/CilinSimilarity/blob/master/data/cilin.txt
## EDU Level Lexical Chain
- 输入EDU/句子的list
- 返回邻接矩阵，1表示两个EDU/句子之间存在词汇衔接，0表示不存在
```python
from lexical_chain import EDU_level_LC

en_edu_lc = EDU_level_LC(
    lang='en', 
    distance=3, 
    thre=0.33, 
    pos_list=['NN', 'NNP', 'NNS', 'NNPS'], 
    cilin_path=None)

en_lc_graph = en_edu_lc.get_lc_graph(
    [
        'Travel is a very good means of broadening a persons perspective.',
        'It makes you come into contact with different cultures, meet people of different colors and go through peculiar rites an ceremonies.',
        'If you travel much, you will not only enrich your knowledge and experiences,',
        'but also be aware of the vastness of nature.'
    ]
)
print(en_lc_graph)

zh_edu_lc = EDU_level_LC(
    lang='zh', 
    distance=3, 
    thre=0.6, 
    pos_list=['n', 'nr', 'ns', 'nt', 'nz'], 
    cilin_path='cilin.txt')

zh_lc_graph = zh_edu_lc.get_lc_graph(
    [
        '打字机由三个部分组成，分别是滚筒、铅字盘和机头。',
        '字盘里面存放着几万个方块字，最特别的是，这些字都是倒放的。',
        '上面不仅有字，还有标点符号、运算符号、字母等。',
        '当我试着按下打字按钮时，打字机竟然还能灵活地运转起来。'
    ]
)
print(zh_lc_graph)
```  
## Word Level Lexical Chain
- 输入text
- 输出词汇链
```python
from lexical_chain import lexical_chain

en_lc = lexical_chain(
    lang='en',
    pos_list=['NN', 'NNS', 'NNP', 'NNPS'],
    thre=0.2,
)
en_lc.processing(
    """Travel is a very good means of broadening a persons perspective. 
    It makes you come into contact with different cultures, 
    meet people of different colors and go through peculiar rites an ceremonies.
    If you travel much, you will not only enrich your knowledge and experiences, 
    but also be aware of the vastness of nature.
    """
)
print(en_lc.getChainsString())

zh_lc = lexical_chain(
    lang='zh',
    pos_list=['n', 'nr', 'ns', 'nt', 'nz'],
    thre=0.1,
    cilin_path='cilin.txt'
)
zh_lc.processing(
    """'打字机由三个部分组成，分别是滚筒、铅字盘和机头。
    字盘里面存放着几万个方块字，最特别的是，这些字都是倒放的。
    上面不仅有字，还有标点符号、运算符号、字母等。
    当我试着按下打字按钮时，打字机竟然还能灵活地运转起来。
    """
)
print(zh_lc.getChainsString())
```
## Reference
- https://github.com/ashengtx/CilinSimilarity
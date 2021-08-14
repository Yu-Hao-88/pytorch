#!/usr/bin/env python
# coding: utf-8

# In[1]:


with open('../data/p1ch4/jane-austen/1342-0.txt', encoding='utf8') as f:
    text = f.read()


# In[12]:


lines = text.split('\n') #將不同列的文字存入串列 lines
line = lines[200] #取出串列 lines 中的第 200 列文字
line


# In[7]:


import torch

letter_t = torch.zeros(len(line), 128) #創建一個張量 letter_t，元素值皆初始化為 0，shape為((該行文字的字元數), ASCII 碼的字元數)

letter_t.shape #印出 letter_t 的 shape


# In[9]:


#將文字轉換為小寫，依序讀入該列文字中的字元
for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 0 else 0 #ord() 可取得讀入字元的 ASCII 碼
    letter_t[i][letter_index] = 1 #將對應到位置的元素值設為 1


# In[19]:


def clean_words(input_str):
    punctuation = '.,;:"!?“”_-' #定義一些常用的標點符號
    word_list = input_str.lower().split() #將文字都轉為小寫，並以空白字元進行切割
    word_list = [word.strip(punctuation) for word in word_list] #利用 strip() 去除文字前、後的標點符號
    return word_list
words_in_line = clean_words(line) #利用 clean_words()套用在之前的 line 上
line, words_in_line #文字被切割成一個個單字


# In[17]:


#將 clean_words(text) 傳回的單字串列轉為 Python 的 set 結構，然後再轉為依字母順序排列的串列
word_list = sorted(set(clean_words(text)))

#將 word_list 中的單字編入字典，key 為單字，value 為其編碼(就是該單字在單字串列中的索引)
word2index_dict = {word: i for (i, word) in enumerate(word_list)}

#text 文本中的單字數, 單字'impossible'在字典中的編碼(對應的索引值)
len(word2index_dict), word2index_dict['impossible'] 


# In[23]:


word_t = torch.zeros(len(words_in_line), len(word2index_dict)) #創建一個張量，用來儲存編碼後的單字向量

for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1 #進行 one-hot 編碼
    print('{} {} {}'.format(i, word_index, word))
print(word_t.shape) #(這段句子的單字數量, 文本中的相異單字數量)


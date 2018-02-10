# pydl-text-encoders

Text encoders for machine learning and nlp in python

The project provides encoders that tranform varied-length text sentences into fixed-length continuous/discrete numpy arrays. 
The objective is to convert unstructured text data into structured data such as matrix that can be fed into machine
learning  algorithms such as those in scikit-learn or tensorflow.

# Usage

# Install

git clone this project to your host computer. copy the "pydl_text_encoders" folder from the git-cloned project into your 
own project's root directory. Now you can use the the library in the same way as shown in the 
[pydl_text_encoders/demo](pydl_text_encoders/demo)

# Usage

### Encode Text Sequence to Fix-Dimension Array: GloVe

The code below shows show to use GloveModel from [pydl_text_encoders/library/glove_loader.py](pydl_text_encoders/library/glove_loader.py)
to convert text sequences of varied lengths (e.g., sentences) into fixed-length array

The sample code can be found in the [pydl_text_encoders/demo/encoders/text2vec_glove.py](pydl_text_encoders/demo/encoders/text2vec_glove.py).

```python
from pydl_text_encoders.library.glove_loader import GloveModel


def main():
    glove_model = GloveModel()
    glove_model.load('../very_large_data') # can be any folder that will be used to store large models downloaded form internet

    print('current encoding is: ', glove_model.embedding_dim) # length of the array encoded from text
    print(glove_model.encode_word('text')) # encode a single word to a 100-dimension array
    print(glove_model.encode_doc('Hello, how are you today?')) # encode a sentence to a 100-dimension array


if __name__ == '__main__':
    main()
```

By default the generated fixed-length array has a 100 dimension. However, if you want to change the dimensions on the 
array generated, you can call one of the following instead of glove_model.load('../very_large_data') in the above 
sample code:

```python
glove_model = ... 

# to generate array of dimension 50
glove_model.load('../very_large/data', embedding_dim=50)

# to generate array of dimension 100
glove_model.load('../very_large/data', embedding_dim=100)

# to generate array of dimension 200
glove_model.load('../very_large/data', embedding_dim=200)

# to generate array of dimension 300
glove_model.load('../very_large/data', embedding_dim=300)
```

### Encode Text Sequence to Fix-Dimension Array: GenSim Word2Vec

The code below shows show to use GenSimWord2VecModel from [pydl_text_encoders/library/gensim_loader.py](pydl_text_encoders/library/gensim_loader.py)
to convert text sequences of varied lengths (e.g., sentences) into fixed-length array

The sample code can be found in the [pydl_text_encoders/demo/encoders/text2vec_gensim.py](pydl_text_encoders/demo/encoders/text2vec_gensim.py).

```python
from pydl_text_encoders.library.gensim_loader import GenSimWord2VecModel


def main():
    sentences = []
    # Import `umich-sentiment-train.txt`
    with open('umich-sentiment-train.txt', mode='rt', encoding='utf8') as file:
        for line in file:
            label, sentence = line.strip().split('\t')
            sentences.append(sentence)

    gensim_model = GenSimWord2VecModel()
    # train the gensim model using text data from `umich-sentiment-train.txt`
    gensim_model.fit(sentences) 

    print('current encoding is: ', gensim_model.embedding_dim) # length of the array encoded from text
    print(gensim_model.encode_word('text')) # encode a single word to a 100-dimension array
    
    for sentence in sentences[0:10]:
        print('Origin: ', sentence)
        print('Encoded: ', gensim_model.encode_doc(sentence))  # encode a sentence to a 100-dimension array


if __name__ == '__main__':
    main()
```

By default the generated fixed-length array has a 100 dimension. However, if you want to change the dimensions on the 
array generated, you can call one of the following instead of gensim_model.fit(sentences) in the above 
sample code:

```python
gensim_model = ... 
sentences = ...

# to generate array of dimension 50
gensim_model.fit(sentences, embed_dim=50)

# to generate array of dimension 100
gensim_model.fit(sentences, embed_dim=100)

# to generate array of dimension 200
gensim_model.fit(sentences, embed_dim=200)

# to generate array of dimension 300
gensim_model.fit(sentences, embed_dim=300)
```
 
  


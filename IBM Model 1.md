
## Lexical Translation (based on Statistical Machine translation: Philipp Koehn)

Here we demonstrate using plain statistics how often a given word is translated into one of its equivalent forms

We run through the parallel corpus and find translations of a given word into one of its forms. Then we count the occurences and output the probability of being translated into one of the meanings.


```python
from collections import Counter, defaultdict
```


```python
de_word = 'haus'

en_words = { 'house', 'building', 'home', 'household', 'shell' }
```


```python
def corpus_reader(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()        
        return [l.replace('\n', '').split('\t') for l in lines]
```


```python
lines = corpus_reader('/Users/tmpaul/Downloads/deu-eng/deu.txt')
```


```python
lines[0]
```




    ['Hi.', 'Hallo!']




```python
def find_de_en_translations(lines, de_word, en_words):
    """Given parallel corpus find the count of de word being translated into one of en_words"""
    counter = Counter()
    for line in lines:
        de_part = line[1]
        en_part = line[0]
        if de_word in de_part.lower().split(' '):
            for en_word in en_words:
                if en_word in en_part.lower():
                    counter[en_word] += 1
                    
    return counter
```


```python
print('German word:', de_word)
print('English translations:', en_words)
print('\nTranslation probabilities\n')
print('\n'.join([k + ': ' + str(v) for k, v in find_de_en_translations(lines, de_word, en_words).most_common()]))
```

    German word: haus
    English translations: {'house', 'shell', 'home', 'household', 'building'}
    
    Translation probabilities
    
    house: 659
    home: 39
    building: 6


## Estimating Lexical Translation probability distribution

The goal is to estimate a lexical translation probability distribution accompanying the corpus above. Given this distribution, we can answer a question when  we have to translate a new German text: What is the most likley English translation for a foreign word like Haus ?

```
    p_f(e)
    
    f -> Foreign word
    e -> English translation
```


```python
def calculate_de_en_lexical_tr_probabilities(lines, de_word, en_words):
    """Given parallel corpus find the count of de word being translated into one of en_words"""
    counter = Counter()
    de_word_count = 0
    for line in lines:
        de_part = line[1]
        en_part = line[0]
        if de_word in de_part.lower().split(' '):
            de_word_count += 1
            for en_word in en_words:
                if en_word in en_part.lower():
                    counter[en_word] += 1
             
    prob_table = {}
    for k, v in counter.items():
        prob_table[k] = v / de_word_count
    return prob_table
```


```python
print('Lexical translation probabilities for {}'.format(de_word))
calculate_de_en_lexical_tr_probabilities(lines, de_word, en_words)
```

    Lexical translation probabilities for haus





    {'building': 0.008333333333333333,
     'home': 0.05416666666666667,
     'house': 0.9152777777777777}



### Conditional translation probability

Given a foreign word, we now find out the lexical translation probability tables for each foreign word.




```python
de_sent = ['das', 'Haus', 'ist', 'klein']

de_en_dict = {
    'das': ['the', 'that', 'which', 'who', 'this'],
    'haus': ['house', 'building', 'home', 'household', 'shell'],
    'ist': ['is', '\'s', 'exists', 'has', 'are'],
    'klein': ['small', 'little', 'short', 'minor', 'petty']
}
```


```python
def straight_alignment_translation(de_sent, de_en_dict, lines):
    for de_word in de_sent:
        de_word = de_word.lower().strip()
        table = Counter(calculate_de_en_lexical_tr_probabilities(lines, de_word, de_en_dict[de_word]))
        max_prob_word = table.most_common()[0][0]
        yield max_prob_word

```

### Straight alignment

For IBM Model 1, we assume straightforward alignment, i.e german word at position i -> english word at position i


```python
print('Most probable lexically aligned translation for \n"{}" is \n"{}"'.format(' '.join(de_sent) ,
      ' '.join(list(straight_alignment_translation(de_sent, de_en_dict, lines)))))
```

    Most probable lexically aligned translation for 
    "das Haus ist klein" is 
    "that home is small"



```python

```

### Learning Lexical Translation models

Instead of using translation probability tables, we will now **learn** these translation probability distributions from sentence-aligned parallel text. This method is the expectation maximization algorithm.


Earlier we used a dictionary containing potential translations for a given word. The idea is to automatically infer the alignment from the data presented to us.

### Expectation Maximization

The expectation maximization algorithm works as follows:

1. Initialize the model, typically with uniform distributions. 
2. Apply the model to the data (expectation step).
3. Learn the model from the data (maximization step).
4. Iterate steps 2 and 3 until convergence.


#### Probability model for different alignments

Given a sentence pair in the data, compute probabilities of different alignments.

```
    p(a | e, f) = p(e, a | f) / p( e | f
    
    Following the derivation in Koehn's book, we have:
    
    p(a | e, f) = product_j (t(ej | fa(j)) / (sum over foreign sentence)
```


```python
def compute_translation_pr(lines, num_iter=5, tol=1e-6):
    """Return the lexical translation probability of a english word given foreign word."""

    # Compute foreign vocab len
    foreign_vocab = set()
    # For counting each occurrence of e & f
    tr_ef_counter = Counter()
    
    count_ef = Counter()
    # For counting total counts of f
    total_f = Counter()
    for line in lines:
        for foreign_word in line[1].split(' '):
            foreign_vocab.add(foreign_word)
    
    foreign_vocab_len = len(foreign_vocab)

    # Uniform initialization
    # t( e | f) = 1 / (#foreign_words)
    uniform_prob = 1 / foreign_vocab_len 
    
    abs_change = 0
    prev_abs_change = 0

    print('Initializing translation probabilities')
    for line in lines:
        en_parts = line[0].split(' ')
        foreign_parts = line[1].split(' ')
        
        for en_part in en_parts:
            for foreign_part in foreign_parts:
                tr_ef_counter['{},{}'.format(en_part, foreign_part)] = uniform_prob

    print('Finished initializing probabilties to uniform value: {}'.format(uniform_prob))
    # Iterate until num_iter or till convergence is achieved
    for i in range(0, num_iter):
        print('Iteration', i)
        abs_change = 0
        # Compute sum over foreign words based on t(e | f)
        se_total = Counter()
        for line in lines:
            
            en_parts = line[0].split(' ')
            foreign_parts = line[1].split(' ')
            
            for en_part in en_parts:
                for foreign_part in foreign_parts:
                    se_total[en_part] += tr_ef_counter['{},{}'.format(en_part, foreign_part)]
            
            for en_part in en_parts:
                for foreign_part in foreign_parts:
                    ef = '{},{}'.format(en_part, foreign_part)
                    t_e_f = tr_ef_counter[ef]
                    count_ef[ef] +=  t_e_f / (se_total[en_part])
                    
                    total_f[foreign_part] += t_e_f / (se_total[en_part])
                    
            local_change = 0
            for foreign_part in foreign_parts:
                for en_part in en_parts:
                    ef = '{},{}'.format(en_part, foreign_part)
                    old_value = tr_ef_counter[ef]
                    tr_ef_counter[ef] = count_ef[ef] / total_f[foreign_part]
                    
                    local_change += abs(old_value - tr_ef_counter[ef])
                    
            abs_change += (local_change) / (len(foreign_parts) * len(en_parts))
                    
        abs_change /= len(lines)
           
        print('Absolute change', abs_change, 'Relative change', abs(abs_change - prev_abs_change))
        prev_abs_change = abs_change
    return tr_ef_counter
    
```


```python
tr_pr = compute_translation_pr(lines, num_iter=3)
```

    Initializing translation probabilities
    Finished initializing probabilties to uniform value: 1.7725467952353943e-05
    Iteration 0
    Absolute change 0.019326007981573124 Relative change 0.019326007981573124
    Iteration 1
    Absolute change 0.004153894026863428 Relative change 0.015172113954709696
    Iteration 2
    Absolute change 0.0009788983542427978 Relative change 0.00317499567262063



```python
tr_pr.most_common(20)
```




    [('Help!,Hülf!', 1.0),
     ('Goodbye!,Tschüss!', 1.0),
     ('Smile.,Lächeln!', 1.0),
     ('Stop!,Stopp!', 1.0),
     ('Terrific!,Hervorragend!', 1.0),
     ('Fantastic!,Fantastisch!', 1.0),
     ('Run!,Lauf!', 1.0),
     ('Welcome.,Willkommen!', 1.0),
     ('Freeze!,Stehenbleiben!', 1.0),
     ('Congratulations!,Glückwunsch.', 1.0),
     ('Terrific!,Sagenhaft!', 1.0),
     ('Wonderful!,Herrlich!', 1.0),
     ('Perfect!,Perfekt!', 1.0),
     ('Cheers!,Wohl!', 1.0),
     ('Unbelievable!,Unglaublich!', 1.0),
     ('cantankerous.,streitsüchtig.', 0.9999999714326055),
     ('extraordinary.,außergewöhnlich.', 0.9999999670187355),
     ('observant.,Beobachtungsgabe.', 0.999999937705235),
     ('predictable.,durchschaubar.', 0.999999928162166),
     ('cockroaches?,Küchenschaben?', 0.9999999215516227)]




```python
tr_pr['house,Haus']
```




    0.4107671485910031




```python

```

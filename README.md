#### language_detect

# Language Detect

Geoffrey Gordon Ashbrook, 2025-2026

Language-Detect is a GOFAI approach to language detection that can be highly optimized compared with statistical learning and deep learning approaches.

https://github.com/lineality/language_detect


# A particular case of LLM Weakness: Detecting an Absence

A mature NLP pipeline is often a phylogeny of NLP technologies, an evolutionary tree, a family of families.

A particular edge case for 2022-2026 (usually transformer type) foundation models is that generative models are not good at handling the edge case where the a key language input is empty, as in automated pipelines where at some stage a generative (or perhaps an embedding model) is performing some operation (such as screening, detection, categorization, translation, etc) on a given input, such as a text message, a text field from a form, or a help-desk question.

Guestimation-models, in a way perhaps highly related to the fundamental lack of a confidence level, are duty-bound to give their best guess about the analysis of something, with the option of abdicating being very often not used (no matter how many examples or prompt details are included).

This can be especially tricky when the input is not "empty" superficially and literally (such as might be detected with an 'empty-string' value detection) but instead contains some unpredictable combination of space, symbols, random letters, numbers, etc.

But using a combination of technologies, by supplementing sub-symbolic flexibility with GOFAI optimized edge case handling, a somewhat deep-learning pipeline can have more deterministic behavior for desired cases. On scale this may also significantly reduce the use or cost of computationally expensive sub-symbolic models.


# A case study in GOFAI, measurability, and explainability.

## not-not-language

Positively defining meaningful language is 'hard' (probably impossible);
negatively defining not-language is not hard, most of the time.


## Negative Language Detection Workflow
1. Define not-language
2. count failures to remove all not-language words
3. count failures to remove all not-language sentences
4. count and return failures

It is possible much of the time to define
what is likely not a word,
and what is likely not a sentence,
with a set of simple steps and rules
that are based on empirical data and
that are not controversial.

Using these rules and steps, is possible much of the time
to find effective word and sentence counts.


### Steps/Rules
0. start with input string
1. add spaces after periods
2. remove extra/duplicate spaces
3. remove extra/duplicate dashes
4. remove extra/duplicate hyphens

### Potential Words: potential_word_count
5. split text on newlines and spaces (or convert those into spaces first)
6. remove potential words that are too long
7. remove potential words that are too short
8. use LEN_TO_N_VOWELS = {word_length: [possible vowel quantities]}
   to check if a word of length N has a possible number of vowels.

### Potential Sentences: potential_sentence_count
9. put the (remaining) potential-words back into 'potential_sentence's:
- split on sentence-ending symbols (ignoring abbreviations, etc.)
10. remove sentences that are too short
11. split potential sentences that are too long
12. remove sentences that contain too few prepositions and standard verbs
13. remove sentences that contain too few stopwords

(other possible checks)
    maybe, longer sentences should contain more NLTK stop-words
    maybe: longer sentences should contain more prepositions/verbs/stopwords.
    probably not:
    N. remove words that contain too many capital letters
    expensive but maybe useful: looking for symbols not at the front or end
    of words but in the middle of words
    N. remove words with too few vowels
    N. remove words with too few consonants
    N. remove words that contain too many numbers
    N. remove words with too many strange symbols

14. Count What remains, those are output:
(potential_word_count, potential_sentence_count)




# Vowel Ratio Range Heuristic: literal quantized relationships

One of the most interesting parts of this system is the model of expected vowel to word length combinations.

I started out trying to statistically analyze the probabilities of vowel distributions, but I realized that there was a type of pattern that I had not frequently encountered in data science: literal quantized relationships.

Trying to draw an abstract curve based on a function through world lengths to reflect likely bounds on vowel numbers in a way that would not be binary in output or integer in input would be strange. And the quest to find an abstract map for the territory that is the optional vowel numbers is also strange.

Take a look at the picture-like list of empirically observed values below.
This is the set of possible vowels per word-length, which is what we need. Anything less than this would be problematic, and anything more would be unhelpful. The empirically observed options are the "model," or rather they literally are what they literally are (they are not a model of themselves).

Based on analysis of 215,784 Wikipedia article words:
these possibles numbers of
vowels per word length
were empirically observed.

(See: analyzer_slim_v5.py for analysis, results, and explanation.)

Empirical examination of quantized options is more fruitful
than abstract overall super-pattern searching.

LEN_TO_N_VOWELS = {
    # # {word_length: [possible vowel quantities]}
    2:  [1],

    3:  [1, 2, 3],
    4:  [1, 2, 3],
    5:  [1, 2, 3],

    6:  [1, 2, 3, 4],
    7:  [1, 2, 3, 4, 5],

    8:     [2, 3, 4, 5],
    9:     [2, 3, 4, 5, 6],
    10:    [2, 3, 4, 5, 6],

    11:       [3, 4, 5, 6],

    12:       [3, 4, 5, 6, 7],
    13:       [3, 4, 5, 6, 7],

    14:          [4, 5, 6, 7],

    15:             [5, 6, 7, 8],

    16:                [6, 7],

    17:                [6, 7, 8],
    18:                [6, 7, 8],
}
```


# Versions
There is a python version and a Rust version. Depending on the particular use-case, such as large files or large numbers of small files or batches of endpoint input etc., specific versions could be optimized for performance for a given use-case. Along these lines, the rust-version, for example, has separate functions for returning only word-count or only sentence count.


Note:
Currently Language detect is only adjusted to search for English words, but it would probably world for other target latin-alphabet (Unicode-OK) languages as well.

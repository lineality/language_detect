# Module for language detection unit tests below
# use/import form module with; from gofai_language_detect import lang_detect_word_sentence_counter as lang_detect
"""
# Language-Detect:

## Language-Detect (not-not-language)
Positively defining meaningful language is 'hard'(impossible);
negatively defining not-language is not hard: most of the time.
1. Define not-language
2. count failures to remove all not-language words
3. count failures to remove all not-language sentences
4. count and return failures

It is possible much of the time to define
what is likely not a word,
and what is likely not a sentence,
with a set of simple steps and rules
that are based on empirical data and
that are not controvertial.

Using these rules and steps, is possible much of the time
to find effective word and sentence counts.

Among the various configuration-settings,
there are three key (strict or loose) settings
that a user may want to adjust:

# Sentence structure rules 1
MIN_WORDS_PER_SENTENCE = 4  # Common sentences have subject and predicate
MIN_VERBS_PREPOSITIONS_PER_SENTENCE = 1
MIN_NLTK_STOPWORDS_PER_SENTENCE = 1

Beyond this, everything in the code may be adapted to specific uses.

## Optimization
An important factor may be balancing run-time optimization
with accuracy/precision. Two rules, number of raw space
split words and vowels per word, may sufficient.


## About Frequencies

Based on analysis of words and sentences from wiki articles, see more below:

Sentence_stats.py:
    per sentence

    word:
    Mean words: 24.76998
    Median words: 19.0
    Mode words: 12
    25th Percentile words: 13.0
    75th Percentile words: 30.0

    char:
    Mean characters: 164.1459750
    Median characters: 124.0
    Mode characters: 66
    25th Percentile characters: 86.0
    75th Percentile characters: 192.0


Note: While these 'steps' should be followed,
optimization to run in parallel on large batches of data
may/should influence how the processes are run.

# Steps/RUles
0. start with input string
1. add spaces after periods
2. remove extra/duplicate spaces
3. remove extra/duplicate dashes
4. remove extra/duplicate hyphons

# potential_words
5. split text on newlines and spaces (or convert those into spaces first)
6. remove potential words that are too long
7. remove potential words that are too short
8. use LEN_TO_N_VOWELS = {word_length: [possible vowel quantities]}
   to check if a word of length N has a possible number of vowels.

# potential_sentences
9. put the (remaining) potential-words back into 'potential_sentence's:
- split on sentence-ending symbols (ignoring abbrvitaions, etc.)
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

If tuple[0], potential_sentence_count is zero,
most of the time,
there is very likely no language in the input,
most of the time.

Pipeline can be made more strict or loose,
maybe with a parameter setting for strict or loose.

Use a body of unit-tests on groups of input to test
and calibrate:
- normal valid language
- terse valid language
- borderline cases
- invalid non-language

Note: some text-cleaning may be desirable first.

## On Stopwords
Note: There may be two opposite types of 'stop-words' here:
1. words or phrases that you need to remove before you use this tool
2. NLTK (or other) common 'stop-words' that you so much expect
   that you require N of them to be in any normal sentence.
"""

"""
Vowel Ratio Range Heuristic
----------------------------

Based on analysis of 215,784 Wikipedia article words:
these possibles numbers of vowels per word length
were empirically observed.
See: analyzer_slim_v5.py for analysis, results, and explanation.

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

"""

# import re
import unittest

# Sentence structure rules 1
MIN_VERBS_PREPOSITIONS_PER_SENTENCE = 1
MIN_NLTK_STOPWORDS_PER_SENTENCE = 1
MIN_WORDS_PER_SENTENCE = 4  # Common sentences have subject and predicate

# List of stop words
NLTK_STOPWORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
]
NLTK_STOPWORDS_SET = set(NLTK_STOPWORDS)

# Vowel count per word length (-s if over len 5]
# word_len_to_vowel_count_list_lookup
LEN_TO_N_VOWELS_BASE = {
    # # 2-4 Len
    # 1: [1]
    2: [1],
    3: [1, 2, 3],
    # 4+ len
    # if word is more than 3 letters long and ends in s, then remove the s
    4: [1, 2, 3],
    5: [1, 2, 3],
    6: [1, 2, 3, 4],
    7: [1, 2, 3, 4, 5],
    8: [2, 3, 4, 5],
    9: [2, 3, 4, 5, 6],
    10: [2, 3, 4, 5, 6],
    11: [3, 4, 5, 6],
    12: [3, 4, 5, 6, 7],
    13: [3, 4, 5, 6, 7],
    14: [4, 5, 6, 7],
    15: [5, 6, 7, 8],
    16: [6, 7],
    17: [6, 7, 8],
    18: [6, 7, 8],
}

# Convert list lookups to sets for O(1) lookup time
LEN_TO_N_VOWELS = {
    length: set(vowels) for length, vowels in LEN_TO_N_VOWELS_BASE.items()
}

# Character sets and linguistic rules
ENGLISH_VOWELS = set("aeiouAEIOUyY")
# Word formation rules
MAX_LENGTH_VOWELS_ONLY = 1  # Only "a", "I" as single vowel words
MAX_LENGTH_NO_VOWELS = 0  # Common short consonant clusters like "Mrs"
MAX_UPPERCASE_PER_WORD = 1  # Just first letter capitalization
MAX_DIGITS_PER_WORD = 1  # Standard words don't have numbers
MAX_HYPHENS_PER_WORD = 1  # Standard words don't have hyphens
MAX_UNDERSCORES_PER_WORD = 1  # Standard words don't have underscores
# Sentence structure rules 2
MAX_WORDS_PER_SENTENCE = 100  # Reasonable limit for normal text
SPLIT_SENTENCES_ON_N_WORDS = 30
# Character sets for validation
VALID_WORD_SYMBOLS = set("-'_")  # Symbols allowed within words

VERB_AND_PREPOS_TERMS_SET = set(
    [
        # prepositions
        "about",
        "above",
        "across",
        "after",
        "against",
        "among",
        "around",
        # "at",
        "before",
        "behind",
        "below",
        "beside",
        "between",
        "by",
        "down",
        "during",
        "for",
        # "from",
        # "in",
        "inside",
        "into",
        "near",
        # "of",
        "off",
        "on",
        "out",
        "over",
        "through",
        # "to",
        "toward",
        "under",
        "up",
        # "with",
        "aboard",
        "along",
        "amid",
        "as",
        "beneath",
        "beyond",
        "but",
        "concerning",
        "considering",
        "despite",
        "except",
        "following",
        "like",
        "minus",
        "next",
        "onto",
        "opposite",
        "outside",
        "past",
        "per",
        "plus",
        "regarding",
        "round",
        "save",
        "since",
        "than",
        "till",
        "underneath",
        "unlike",
        "until",
        "upon",
        "versus",
        "via",
        "within",
        "without",
        # common verbs
        "am",
        "is",
        "are",
        "was",
        "were",
        "been",
        "being",
        "be",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "done",
        "doing",
        "say",
        "says",
        "said",
        "saying",
        "go",
        "goes",
        "went",
        "gone",
        "going",
        "get",
        "gets",
        "got",
        "gotten",
        "getting",
        "make",
        "makes",
        "made",
        "making",
        "know",
        "knows",
        "knew",
        "known",
        "knowing",
        "think",
        "thinks",
        "thought",
        "thinking",
        "take",
        "takes",
        "took",
        "taken",
        "taking",
        "see",
        "sees",
        "saw",
        "seen",
        "seeing",
        "come",
        "comes",
        "came",
        "coming",
        "want",
        "wants",
        "wanted",
        "wanting",
        "look",
        "looks",
        "looked",
        "looking",
        "use",
        "uses",
        "used",
        "using",
        "find",
        "finds",
        "found",
        "finding",
        "give",
        "gives",
        "gave",
        "given",
        "giving",
        "tell",
        "tells",
        "told",
        "telling",
        "work",
        "works",
        "worked",
        "working",
        "call",
        "calls",
        "called",
        "calling",
        "try",
        "tries",
        "tried",
        "trying",
        "ask",
        "asks",
        "asked",
        "asking",
        "need",
        "needs",
        "needed",
        "needing",
        "feel",
        "feels",
        "felt",
        "feeling",
        "become",
        "becomes",
        "became",
        "become",
        "becoming",
        "leave",
        "leaves",
        "left",
        "leaving",
        "put",
        "puts",
        "putting",
    ]
)
SENTENCE_ENDINGS = set(".!?")  # Characters that CAN end sentences (not required)
ABBREVIATIONS_SET = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "st.",
    "dr.",
    "cir.",
    "inc.",
    # upper
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Vs.",
    "VS.",
    "Etc.",
    "E.g.",
    "I.e.",
    "St.",
    "Dr.",
    "Cir.",
    "Inc.",
}

INVALID_SYMBOLS = set("!@#$%^&*<>{}[]\\|")  # Symbols that invalidate words
# INVALID_SYMBOLS = set("!@#$%^&*()")  # Add any other prohibited symbols as needed


# add comments
def is_valid_english_word(word_candidate: str) -> bool:
    """
    The challenge here is to look for odd symbols
    'in' a word, but not bookending a word.

    ok! is fine.
    (fine) is fine

    but:
        o!#K is not ok,
        and f(n)e is not fine.
    """
    # Early returns for common cases
    if not word_candidate:
        return False

    if len(word_candidate) >= 3:
        # Trim first and last character
        inner_word = word_candidate[1:-1]

        # Check if any character in the word candidate is in the set of prohibited symbols
        if any(char in INVALID_SYMBOLS for char in inner_word):
            return False

    else:  ## if less then three characters long (if unlikely)
        # Check if any character in the word candidate is in the set of prohibited symbols
        if any(char in INVALID_SYMBOLS for char in word_candidate):
            return False

    # Remove punctuation check if not needed
    return check_vowel_count_for_length(word_candidate)


def split_over_max_onesentence_wordlist(
    input_sentence_wordlist: list[str],
) -> list[list[str]]:
    """
    Splits a sentence into smaller segments if it exceeds the maximum number of words per sentence.

    Args:
    input_sentence_wordlist (list[str]): A sentence represented as a list of words.

    Returns:
    list[list[str]]: List of sentence segments, where each segment is a list of words.
                     If the input sentence is within the limit, returns a list containing
                     the original sentence. Otherwise, returns multiple segments.

    Uses:
    SPLIT_SENTENCES_ON_N_WORDS (int): Number of words to split each sentence into smaller segments.
    MAX_WORDS_PER_SENTENCE (int): Maximum number of words allowed per sentence segment.
    """
    try:
        split_list: list[list[str]] = (
            []
        )  # Added type annotation here to help the type checker

        num_words = len(input_sentence_wordlist)

        if num_words <= MAX_WORDS_PER_SENTENCE:
            split_list.append(input_sentence_wordlist)
        else:
            # Split the sentence into smaller segments
            for i in range(0, num_words, SPLIT_SENTENCES_ON_N_WORDS):
                segment = input_sentence_wordlist[i : i + SPLIT_SENTENCES_ON_N_WORDS]
                split_list.append(segment)

        return split_list
    except Exception as e:
        print(str(e))
        return [input_sentence_wordlist]  # Fixed to return list[list[str]]


# works
def check_vowel_count_for_length(word: str) -> bool:
    """
    Validates if a word has an appropriate number of vowels for its length based on
    English language patterns defined in LEN_TO_N_VOWELS lookup table.

    This function implements validation rules for English word vowel patterns:
    1. Maps word length to allowed vowel counts using LEN_TO_N_VOWELS table
    2. Handles words longer than the maximum defined length by using the maximum length's rules
    3. Enforces minimum length requirements
    4. Counts both uppercase and lowercase vowels (including 'y' and 'Y')

    Args:
        word (str): The word to validate

    Returns:
        bool: True if the word has a valid vowel count for its length, False otherwise

    Example:
        >>> check_vowel_count_for_length("hello")  # len 5, 2 vowels
        True
        >>> check_vowel_count_for_length("rhythm")  # len 6, 0 vowels (y as consonant)
        False

    Note:
        - Uses the LEN_TO_N_VOWELS global dictionary which maps word lengths to valid
          vowel counts
        - Words shorter than minimum length in LEN_TO_N_VOWELS return False
        - For words longer than maximum defined length, uses rules for maximum length
    """
    # Cap word length to maximum defined in lookup table
    word_len = min(len(word), max(LEN_TO_N_VOWELS.keys()))

    # Fail if word is shorter than minimum defined length
    if word_len < min(LEN_TO_N_VOWELS.keys()):
        return False

    # Count vowels in word (case insensitive)
    vowel_count = sum(1 for char in word.lower() if char in ENGLISH_VOWELS)

    # Get valid vowel counts for this word length
    valid_vowel_counts = LEN_TO_N_VOWELS.get(word_len, [])

    # Check if vowel count is valid for word length
    return vowel_count in valid_vowel_counts


def remove_duplicate_chars(
    text: str,
    chars_to_dedupe: set[str] | None = None,
) -> str:
    """
    Alternative implementation using iterative approach instead of regex.

    Remove duplicate consecutive occurrences of specified characters from a string.
    This version processes the string character by character for potentially
    better performance on smaller strings.

    Args:
        text (str): The input string to process
        chars_to_dedupe (set, optional): Set of characters to remove duplicates for.
                                       Defaults to {' ', '-', '–', '—'}

    Returns:
        str: The processed string with duplicate characters removed
    """

    # Default characters to remove duplicates for
    if chars_to_dedupe is None:
        chars_to_dedupe = {" ", "-", "–", "—"}  # Using set for O(1) lookup

    # # Input validation
    # if not isinstance(text, str):
    #     raise TypeError("Input must be a string")

    if not text:  # Handle empty string
        return text

    result: list[str] = []  # Type annotation for result
    prev_char: str | None = None  # Type annotation for prev_char

    # Iterate through each character
    for char in text:
        # If current char is in our target set and same as previous, skip it
        if char in chars_to_dedupe and char == prev_char:
            continue
        else:
            result.append(char)
            prev_char = char

    final_result = "".join(result)

    return final_result


def sanitize_and_split_text(raw_text: str) -> list[str]:
    """
    Sanitizes and splits input text into a list of potential words.

    This function performs basic text normalization optimized for speed:
    1. Replaces all newlines and tabs with single spaces
    2. Ensures periods have a following space for sentence separation
    3. Splits text on whitespace into a list of strings

    Args:
        raw_text (str): The input text to be sanitized and split.
            Can be multi-line text containing any characters.

    Returns:
        list: A list of strings (potential words and punctuation).
            Empty list if input is empty/None.

    Design Notes:
        - Uses string.replace() instead of regex for performance
        - Single-pass replacement reduces string copying operations
        - Direct split() is faster than regex splitting for simple cases
        - Preserves internal punctuation and case for later analysis
        - Does not remove duplicate spaces (handled by split)

    Example:
        >>> sanitize_and_split_text("Hello\tworld.\nGoodbye!")
        ['Hello', 'world', '.', 'Goodbye!']
    """
    if not raw_text:
        return []

    # optional: remove duplicate chars
    """
    can input custom set chars_to_dedupe
    {' ', '-', '–', '—'}
    defaults to spaces, dashes
    """
    raw_text = remove_duplicate_chars(raw_text)

    # Single pass: replace newlines/tabs with space, add space after periods
    text = raw_text.replace("\n", " ").replace("\t", " ").replace(".", ". ")
    # Direct split - faster than regex for simple cases
    return text.split()


def split_wordlist_into_sentences_and_filter(
    words: list[str],
) -> list[list[str]]:
    """
    Split a list of words into sentences and filter them based on length and content criteria.

    This function processes a list of words to form valid sentences by:
    1. Detecting sentence boundaries using punctuation (., !, ?)
    2. Handling abbreviations (Mr., Dr., etc.) to avoid false sentence breaks
    3. Ensuring sentences meet minimum word count (MIN_WORDS_PER_SENTENCE)
    4. Ensuring sentences contain required grammatical elements (verbs/prepositions)
    5. For valid sentences exceeding MAX_WORDS_PER_SENTENCE:
       - Splits them using split_over_max_onesentence_wordlist()
       - Only keeps split segments that maintain validity criteria

    Args:
        words (list): List of strings representing individual words.
                     Words may include attached punctuation.
                     Empty list returns empty list.
                     None or invalid input raises ValueError.

    Returns:
        list[list[str]]: List of valid sentences where:
            - Each sentence is a list of words/punctuation
            - Each sentence has >= MIN_WORDS_PER_SENTENCE words
            - Each sentence contains required grammatical elements
            - No sentence exceeds MAX_WORDS_PER_SENTENCE words
            - Empty list if no valid sentences found

    Dependencies:
        - SENTENCE_ENDINGS (tuple): Valid sentence-ending punctuation
        - ABBREVIATIONS_SET (set): Known abbreviations to handle
        - MIN_WORDS_PER_SENTENCE (int): Minimum words required
        - MAX_WORDS_PER_SENTENCE (int): Maximum words allowed
        - VERB_AND_PREPOS_TERMS_SET (set): Required grammatical elements
        - split_over_max_onesentence_wordlist(): Handles oversized sentences

    Examples:
        >>> words = ["Hello", "world", "!"]
        >>> split_wordlist_into_sentences_and_filter(words)
        []  # Too few words, below MIN_WORDS_PER_SENTENCE

        >>> words = ["The", "cat", "sits", "on", "the", "mat", "."]
        >>> split_wordlist_into_sentences_and_filter(words)
        [['The', 'cat', 'sits', 'on', 'the', 'mat', '.']]  # Valid sentence

    Raises:
        ValueError: If words is None or contains invalid elements
        TypeError: If words is not a list

    Notes:
        - Does not require ending punctuation (will be added if missing)
        - Split segments must independently meet all validity criteria
    """
    sentences: list[list[str]] = []
    current_sentence: list[str] = []
    final_sentences: list[list[str]] = []

    # First: Split into actual sentences
    for word in words:
        if (
            any(word.endswith(end) for end in SENTENCE_ENDINGS)
            and word.lower() not in ABBREVIATIONS_SET
        ):
            word_part = word[:-1]
            punct_part = word[-1]

            if word_part:
                current_sentence.append(word_part)
            current_sentence.append(punct_part)
            sentences.append(current_sentence)
            current_sentence = []
        else:
            current_sentence.append(word)

    # Handle any remaining words in last sentence
    """
    adds a period at end if none,
    so that no trailing text is lost
    when splitting on SENTENCE_ENDINGS
    """
    if current_sentence:
        if not any(current_sentence[-1].endswith(end) for end in SENTENCE_ENDINGS):
            current_sentence.append(".")
        sentences.append(current_sentence)

    # Now filter the sentences and handle length
    for this_sentence in sentences:

        """
        remove punctuation from the end of the sentence
        to avoid mis-counting words number by +1
        """

        remove_list = [
            ".",
            "!",
            "?",
            # ";",
        ]
        remove_set = set(remove_list)

        # # inspection
        # print(this_sentence[-1])

        if this_sentence[-1] in remove_set:
            del this_sentence[-1]

        # # inspection
        # print(this_sentence[-1])

        # Calculate the count of verbs/prepositions in the current sentence
        verb_preposition_count = sum(
            1 for word in this_sentence if word.lower() in VERB_AND_PREPOS_TERMS_SET
        )

        nltk_stopword_count = sum(
            1 for word in this_sentence if word.lower() in NLTK_STOPWORDS
        )

        # First check if it's a valid sentence
        if (
            len(this_sentence) >= MIN_WORDS_PER_SENTENCE
            and verb_preposition_count >= MIN_VERBS_PREPOSITIONS_PER_SENTENCE
            and nltk_stopword_count >= MIN_NLTK_STOPWORDS_PER_SENTENCE
        ):  # Changed here

            # If valid and too long, split while preserving meaning
            if len(this_sentence) > MAX_WORDS_PER_SENTENCE:
                split_segments = split_over_max_onesentence_wordlist(this_sentence)

                # validate each split segment
                for segment in split_segments:
                    # Calculate count for segment
                    segment_verb_preposition_count = sum(
                        1
                        for word in segment
                        if word.lower() in VERB_AND_PREPOS_TERMS_SET
                    )
                    segment_nltk_stopword_count_count = sum(
                        1 for word in segment if word.lower() in NLTK_STOPWORDS
                    )

                    if (
                        len(segment) >= MIN_WORDS_PER_SENTENCE
                        and segment_verb_preposition_count
                        >= MIN_VERBS_PREPOSITIONS_PER_SENTENCE
                        and segment_nltk_stopword_count_count
                        > MIN_NLTK_STOPWORDS_PER_SENTENCE
                    ):
                        final_sentences.append(segment)
            else:
                final_sentences.append(this_sentence)

        # # inspection print
        # inspection_blurb = f"""
        # this_sentence               -> {this_sentence}
        # verb_preposition_count -> {verb_preposition_count}
        # nltk_stopword_count    -> {nltk_stopword_count}
        # """
        # print(inspection_blurb)

    return final_sentences


# #############################
# # Alternate Reporter version
# #############################
# def count_stop_words(sentence_list: list[str]) -> tuple[int, int]:
#     """
#     Count total words and stopwords in a sentence.

#     Args:
#         sentence_list (list[str]): List of words in the sentence

#     Returns:
#         tuple[int, int]: (total_word_count, stopword_count)
#     """
#     # Filter out punctuation for accurate word count
#     words = [word for word in sentence_list if not all(char in '.,!?' for char in word)]
#     word_count = len(words)
#     stop_word_count = sum(1 for word in words if word.lower() in NLTK_STOPWORDS)
#     return word_count, stop_word_count


# def log_stopword_report(sentence: list[str], score: int):
#     """
#     Log stopword analysis for rejected sentences.

#     Args:
#         sentence (list[str]): The sentence as a list of words
#         score (int): 0 for rejected, 1 for single sentence
#     """
#     word_count, stop_count = count_stop_words(sentence)
#     ratio = stop_count / word_count if word_count > 0 else 0

#     log_entry = (
#         f"Sentence: {' '.join(sentence)}\n"
#         f"Score: {score}\n"
#         f"Word count: {word_count}\n"
#         f"Stopword count: {stop_count}\n"
#         f"Stopword ratio: {ratio:.2f}\n\n"
#     )

#     with open('stopword_log.txt', 'a', encoding='utf-8') as f:
#         f.write(log_entry)


# def split_wordlist_into_sentences_and_filter(words: list) -> list[list[str]]:
#     # [Previous docstring remains the same]

#     sentences = []
#     current_sentence = []
#     final_sentences = []

#     # First: Split into actual sentences
#     for word in words:
#         if (any(word.endswith(end) for end in SENTENCE_ENDINGS) and
#                 word.lower() not in ABBREVIATIONS_SET):
#             word_part = word[:-1]
#             punct_part = word[-1]

#             if word_part:
#                 current_sentence.append(word_part)
#             current_sentence.append(punct_part)
#             sentences.append(current_sentence)
#             current_sentence = []
#         else:
#             current_sentence.append(word)

#     # Handle any remaining words in last sentence
#     if current_sentence:
#         if not any(current_sentence[-1].endswith(end) for end in SENTENCE_ENDINGS):
#             current_sentence.append('.')
#         sentences.append(current_sentence)

#     # Now filter the sentences and handle length
#     for sentence in sentences:
#         # Log invalid sentences (too short)
#         if len(sentence) < MIN_WORDS_PER_SENTENCE:
#             log_stopword_report(sentence, 0)
#             continue

#         # Log sentences missing required grammatical elements
#         if not any(word.lower() in VERB_AND_PREPOS_TERMS_SET for word in sentence):
#             log_stopword_report(sentence, 0)
#             continue

#         # If valid and too long, split while preserving meaning
#         if len(sentence) > MAX_WORDS_PER_SENTENCE:
#             split_segments = split_over_max_onesentence_wordlist(sentence)
#             # Now validate each split segment
#             for segment in split_segments:
#                 if (len(segment) >= MIN_WORDS_PER_SENTENCE and
#                     any(word.lower() in VERB_AND_PREPOS_TERMS_SET for word in segment)):
#                     final_sentences.append(segment)
#                 else:
#                     log_stopword_report(segment, 0)
#         else:
#             final_sentences.append(sentence)
#             # Log valid single sentences
#             log_stopword_report(sentence, 1)

#     return final_sentences

# # # Works!
# def split_wordlist_into_sentences_and_filter(words: list) -> list[list[str]]:
#     """
#     Split a list of words into sentences and filter them based on length criteria.

#     This function processes a list of words to form valid sentences by:
#     1. Detecting sentence boundaries using punctuation (., !, ?)
#     2. Handling abbreviations (Mr., Dr., etc.) to avoid false sentence breaks
#     3. Ensuring sentences meet minimum and maximum word count requirements
#     4. Adding missing sentence-ending punctuation where needed

#     Args:
#         words (list): List of strings representing individual words
#                      May include punctuation attached to words

#     Returns:
#         list[list[str]]: List of valid sentences, where each sentence is a list of words
#                          Including punctuation as separate elements
#                          Only returns sentences meeting length criteria

#     Example:
#         Input: ["Hello", "world", "This", "works."]
#         Output: [["Hello", "world", "."], ["This", "works", "."]]

#     Note:
#         - Uses global constants: SENTENCE_ENDINGS, ABBREVIATIONS_SET,
#           MIN_WORDS_PER_SENTENCE, MAX_WORDS_PER_SENTENCE
#         - Automatically adds period to sentences lacking ending punctuation
#         - Empty or invalid input returns empty list
#     """
#     sentences = []
#     current_sentence = []

#     # Process each word to build sentences
#     for word in words:
#         if (any(word.endswith(end) for end in SENTENCE_ENDINGS) and
#                 word.lower() not in ABBREVIATIONS_SET):
#             word_part = word[:-1]
#             punct_part = word[-1]

#             if word_part:
#                 current_sentence.append(word_part)
#             current_sentence.append(punct_part)

#             # check for prepositions/verbs here
#             if (MIN_WORDS_PER_SENTENCE <= len(current_sentence) <= MAX_WORDS_PER_SENTENCE and
#                 any(word.lower() in VERB_AND_PREPOS_TERMS_SET for word in current_sentence)):
#                 sentences.append(current_sentence)
#             current_sentence = []
#         else:
#             current_sentence.append(word)

#     # Handle any remaining words in last sentence
#     if current_sentence:
#         if not any(current_sentence[-1].endswith(end) for end in SENTENCE_ENDINGS):
#             current_sentence[-1] = current_sentence[-1] + '.'
#         # check for prepositions/verbs here too
#         if (MIN_WORDS_PER_SENTENCE <= len(current_sentence) <= MAX_WORDS_PER_SENTENCE and
#             any(word.lower() in VERB_AND_PREPOS_TERMS_SET for word in current_sentence)):
#             sentences.append(current_sentence)

#     # Special case handling with preposition/verb check
#     if not sentences and words:
#         if (MIN_WORDS_PER_SENTENCE <= len(words) <= MAX_WORDS_PER_SENTENCE and
#             any(word.lower() in VERB_AND_PREPOS_TERMS_SET for word in words)):
#             if not any(words[-1].endswith(end) for end in SENTENCE_ENDINGS):
#                 words.append('.')
#             sentences.append(words)

#     return sentences

# # Initialize containers for complete sentences and current working sentence
# sentences = []
# current_sentence = []

# # Process each word to build sentences
# for word in words:
#     # Check if word ends sentence (has ending punct. and isn't abbreviation)
#     if (any(word.endswith(end) for end in SENTENCE_ENDINGS) and
#             word.lower() not in ABBREVIATIONS_SET):
#         # Split word into content and punctuation
#         word_part = word[:-1]
#         punct_part = word[-1]

#         # Add word content if exists
#         if word_part:
#             current_sentence.append(word_part)
#         # Add punctuation as separate token
#         current_sentence.append(punct_part)

#         # If sentence meets length criteria, add to results
#         if MIN_WORDS_PER_SENTENCE <= len(current_sentence) <= MAX_WORDS_PER_SENTENCE:
#             sentences.append(current_sentence)
#         current_sentence = []
#     else:
#         # Not sentence end - add word to current sentence
#         current_sentence.append(word)

# # Handle any remaining words in last sentence
# if current_sentence:
#     # Add period if no ending punctuation
#     if not any(current_sentence[-1].endswith(end) for end in SENTENCE_ENDINGS):
#         current_sentence[-1] = current_sentence[-1] + '.'
#     # Add if meets length criteria
#     if MIN_WORDS_PER_SENTENCE <= len(current_sentence) <= MAX_WORDS_PER_SENTENCE:
#         sentences.append(current_sentence)


# # Special case: if no sentences found but have words meeting length criteria
# # If no sentences were formed at all, try to treat entire input as one sentence
# if not sentences and words:
#     if MIN_WORDS_PER_SENTENCE <= len(words) <= MAX_WORDS_PER_SENTENCE:
#         # Ensure ends with period
#         if not any(words[-1].endswith(end) for end in SENTENCE_ENDINGS):
#             words.append('.')
#         sentences.append(words)

# return sentences


def lang_detect_word_sentence_counter(input_text: str) -> tuple[int, int]:
    """
    Analyzes input text to count valid English words and complete sentences.

    This function implements a multi-step filtering process:
    1. Normalizes spacing in the input text
    2. Splits and sanitizes text into potential words
    3. Validates words based on English language rules
    4. Identifies and validates sentences
    5. Returns counts of valid words and sentences found

    Args:
        input_text (str): Raw input text to analyze. Can contain multiple sentences,
            spacing variations, and punctuation.

    Returns:
        tuple[int, int]: A tuple containing:
            - First element: Count of valid English words found
            - Second element: Count of valid complete sentences found

    Valid words must meet criteria including:
        - Appropriate vowel-to-length ratios
        - Limited special characters
        - Conformity to English word patterns

    Valid sentences must:
        - Contain MIN_WORDS_PER_SENTENCE

    Example:
        >>> text = "Please reply to my request about weather."
        >>> lang_detect_word_sentence_counter(text)
        (7, 1)
    """
    # Normalize spaces and join split words back together
    input_text = " ".join(word for word in input_text.split())

    # Split text into sanitized potential words
    words_list: list[str] = sanitize_and_split_text(input_text)

    # Filter for valid English words based on linguistic rules
    valid_wordslist: list[str] = [
        word for word in words_list if is_valid_english_word(word)
    ]

    # Group words into potential sentences
    # including further splitting overly long potential sentences
    sentences: list[list[str]] = split_wordlist_into_sentences_and_filter(
        valid_wordslist
    )

    # Filter for valid sentences meeting length and punctuation criteria
    valid_sentences = [
        sentence for sentence in sentences if MIN_WORDS_PER_SENTENCE <= len(sentence)
    ]

    # # Inspection
    #  print(f"""

    # input_text {input_text}
    # valid_words_list{valid_words_list}
    # valid_sentences{valid_sentences}

    # """)

    # Return tuple of counts (valid_words_list, valid_sentences)
    return (len(valid_wordslist), len(valid_sentences))


# # Example usage and testing
# test_text = "please reply to my request about weather, tom"
# word_count, sentence_count = lang_detect_word_sentence_counter(test_text)
# print(f"Valid words found: {word_count}")
# print(f"Complete sentences identified: {sentence_count}")


# empty/incomplete emails
invalid_incomplete_test_cases_2 = [
    """
"buy $$$"
""",
    """
    "BUY SPAM BUY SPAM!",  # not valid
""",
]

# short/borderline examples
valid_short_test_cases_4 = [
    """
He had a great time there.
""",
    """
This is sentence one.
""",
    """
This is really a sentence.
""",
]

valid_sample_cases = [
    "Mr. Smith went to Washington. He had a great time!",
    "This is sentence one. This is sentence two! What about three?",
    "Short. Too short. This is a proper sentence.",
    "Dr. Jones, Prof. Smith and Mrs. Brown attended the meeting on Downing St. Inc.",
    "This sentence is not incomplete.",
]


# # Valid Emails
valid_borderline_test_cases_3 = [
    """
This is a proper sentence.
""",
    """
Short. Too short. This is a proper sentence.
""",
]

# probably_invalid edge cases
edge_case_probably_invalid = [
    """
buy $$$
""",
    """
SPAM SPAM!
""",
    "fraud@crypto is the B!!!est slimball.",
]

###########
# Unittest
###########
"""
NOTE: In Python's unittest framework,
test methods must start with the word "test_"
to be automatically discovered and executed by the test runner

use:
    python3 -m unittest gofai_language_detect.py
"""


# short_test_cases_4
class ValidShortCasesTestLanguageDetection(unittest.TestCase):
    def test_short_valid(self):
        for test_case in valid_short_test_cases_4:
            with self.subTest(test_case=test_case):
                result = lang_detect_word_sentence_counter(test_case)
                self.assertGreater(result[1], 0)


class EmptyInvalidTestLanguageDetection(unittest.TestCase):
    def test_empty_invalid(self):
        for test_case in invalid_incomplete_test_cases_2:
            with self.subTest(test_case=test_case):
                result = lang_detect_word_sentence_counter(test_case)
                self.assertEqual(result[1], 0)


class EdgeCaseInvalidTestLanguageDetection(unittest.TestCase):
    def test_edgecase_empty_invalid(self):
        for test_case in edge_case_probably_invalid:
            with self.subTest(test_case=test_case):
                result = lang_detect_word_sentence_counter(test_case)
                self.assertEqual(result[1], 0)


class ValidBorderlineTestLanguageDetection(unittest.TestCase):
    def test_valid_borderline(self):
        for test_case in valid_borderline_test_cases_3:
            with self.subTest(test_case=test_case):
                result = lang_detect_word_sentence_counter(test_case)
                self.assertGreater(result[1], 0)


class ValidSampleTestLanguageDetection(unittest.TestCase):
    def test_valid_samples(self):
        for test_case in valid_sample_cases:
            with self.subTest(test_case=test_case):
                result = lang_detect_word_sentence_counter(test_case)
                self.assertGreater(result[1], 0)


# Unittest N
class ShortCasesTestLanguageDetection(unittest.TestCase):
    def test_short_valid(self):
        short_test_cases_4 = [
            "please reply to my request about weather, tom",
            "This is sentence one. This is sentence two. Here is!",
        ]
        for test_case in short_test_cases_4:
            with self.subTest(test_case=test_case):
                result = lang_detect_word_sentence_counter(test_case)
                self.assertGreater(result[1], 0)


if __name__ == "__main__":
    result = unittest.main()
    print(result)

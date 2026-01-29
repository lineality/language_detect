//! # Language Detection Module
//!
//! ## Project Context
//! This module implements a heuristic-based English language detection system
//! using negative definition: instead of defining what IS language (hard),
//! we define what is NOT language (easier) and count failures to remove
//! non-language constructs.
//!
//! ## Design Philosophy
//! - Positively defining meaningful language is impossible
//! - Negatively defining not-language is tractable most of the time
//! - Count failures to remove not-language words and sentences
//! - Return word and sentence counts as language presence indicators
//!
//! ## Key Configurable Thresholds
//! - `MIN_WORDS_PER_SENTENCE`: Minimum words for valid sentence (default: 4)
//! - `MIN_VERBS_PREPOSITIONS_PER_SENTENCE`: Required grammatical elements (default: 1)
//! - `MIN_NLTK_STOPWORDS_PER_SENTENCE`: Required common words (default: 1)
//!
//! ## Performance Optimizations
//! - O(1) HashSet lookups via LazyLock initialization
//! - ASCII fast-path for common English text
//! - Pre-allocated vectors to minimize reallocations
//! - Early exits on validation failures
//! - Compiler-optimized match statements (jump tables)
//!
//! ## Production Safety
//! - No panics in production code paths
//! - All errors return Result types for graceful handling
//! - No heap allocation in error messages
//! - Unique error prefixes for traceability

use std::collections::HashSet;
use std::sync::LazyLock;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Minimum words required for a valid sentence.
/// Common sentences have subject and predicate, rarely fewer than 4 words.
const MIN_WORDS_PER_SENTENCE: usize = 4;

/// Minimum verbs or prepositions required per sentence.
/// Validates grammatical structure presence.
const MIN_VERBS_PREPOSITIONS_PER_SENTENCE: usize = 1;

/// Minimum NLTK-style stopwords required per sentence.
/// Common words expected in natural language.
const MIN_NLTK_STOPWORDS_PER_SENTENCE: usize = 1;

/// Maximum words allowed in a single sentence before splitting.
/// Prevents run-on sentence false positives.
const MAX_WORDS_PER_SENTENCE: usize = 100;

/// Word count threshold for splitting oversized sentences.
/// Based on empirical analysis: median sentence ~19 words, 75th percentile ~30.
const SPLIT_SENTENCES_ON_N_WORDS: usize = 30;

// ============================================================================
// STATIC CHARACTER SETS - Compile-time arrays for O(1) lookups
// ============================================================================

// /// Vowels including Y for English word validation.
// /// Both cases included for case-insensitive matching without conversion.
// const VOWELS: [char; 12] = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y'];

/// Symbols that invalidate words when found in interior positions.
/// Exterior punctuation (word boundaries) is handled separately.
const INVALID_INTERIOR_SYMBOLS: [char; 16] = [
    '!', '@', '#', '$', '%', '^', '&', '*', '<', '>', '{', '}', '[', ']', '\\', '|',
];

/// Characters that can terminate sentences.
/// Note: Abbreviations are handled separately to avoid false splits.
const SENTENCE_ENDINGS: [char; 3] = ['.', '!', '?'];

// ============================================================================
// LAZY-INITIALIZED HASHSETS - O(1) runtime lookups
// ============================================================================

/// NLTK-derived stopwords for sentence validation.
/// Presence of common words indicates natural language.
///
/// # Performance
/// - LazyLock ensures single initialization
/// - HashSet provides O(1) contains checks
/// - Static lifetime avoids repeated allocation
static STOPWORDS_SET: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
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
    ])
});

/// Common English verbs and prepositions for grammatical validation.
/// Sentences lacking these elements are likely not natural language.
static VERBS_PREPOSITIONS_SET: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // Prepositions
        "about",
        "above",
        "across",
        "after",
        "against",
        "among",
        "around",
        "before",
        "behind",
        "below",
        "beside",
        "between",
        "by",
        "down",
        "during",
        "for",
        "inside",
        "into",
        "near",
        "off",
        "on",
        "out",
        "over",
        "through",
        "toward",
        "under",
        "up",
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
        // Common verbs (all forms)
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
        "becoming",
        "leave",
        "leaves",
        "left",
        "leaving",
        "put",
        "puts",
        "putting",
    ])
});

/// Common abbreviations that should not trigger sentence splits.
/// Includes both lowercase and title-case variants.
static ABBREVIATIONS_SET: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // Lowercase
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "st.",
        "cir.", "inc.", // Title case
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Vs.", "Etc.", "E.g.", "I.e.", "St.",
        "Cir.", "Inc.", // Uppercase
        "VS.", "DR.", "MR.", "MRS.", "MS.", "PROF.", "SR.", "JR.", "ETC.", "E.G.", "I.E.", "ST.",
        "CIR.", "INC.",
    ])
});

// ============================================================================
// INLINE LOOKUP FUNCTIONS - Hot path optimizations
// ============================================================================

// /// O(1) vowel check supporting English and Western European languages.
// ///
// /// # Performance
// /// - matches! macro compiles to efficient jump table
// /// - Single expression, no branching logic
// ///
// /// # Supported Languages
// /// English, French, German, Spanish, Portuguese, Italian, Nordic languages
// #[inline(always)]
// fn is_vowel(ch: char) -> bool {
//     matches!(
//         ch,
//         // ASCII vowels
//         'a' | 'e' | 'i' | 'o' | 'u' | 'y' |
//         'A' | 'E' | 'I' | 'O' | 'U' | 'Y' |
//         // Lowercase accented
//         'à' | 'â' | 'ä' | 'á' | 'ã' |
//         'è' | 'ê' | 'ë' | 'é' |
//         'ì' | 'î' | 'ï' | 'í' |
//         'ò' | 'ô' | 'ö' | 'ó' | 'õ' |
//         'ù' | 'û' | 'ü' | 'ú' |
//         'ÿ' | 'ý' |
//         // Uppercase accented
//         'À' | 'Â' | 'Ä' | 'Á' | 'Ã' |
//         'È' | 'Ê' | 'Ë' | 'É' |
//         'Ì' | 'Î' | 'Ï' | 'Í' |
//         'Ò' | 'Ô' | 'Ö' | 'Ó' | 'Õ' |
//         'Ù' | 'Û' | 'Ü' | 'Ú' |
//         'Ÿ' | 'Ý' |
//         // Nordic
//         'æ' | 'Æ' | 'ø' | 'Ø' | 'å' | 'Å'
//     )
// }

/// O(1) vowel check with ASCII fast-path.
///
/// # Performance
/// - ASCII bytes checked via matches! macro (compiles to efficient jump table)
/// - Non-ASCII characters return false (not English vowels)
///
/// # Design Note
/// English vowels are ASCII-only. Non-ASCII characters (accented vowels like é, ü)
/// are not considered vowels for this English language detection heuristic.
#[inline(always)]
fn is_vowel(ch: char) -> bool {
    ch.is_ascii()
        && matches!(
            ch as u8,
            b'a' | b'e' | b'i' | b'o' | b'u' | b'y' | b'A' | b'E' | b'I' | b'O' | b'U' | b'Y'
        )
}

/// O(1) invalid interior symbol check.
///
/// # Context
/// These symbols are invalid INSIDE words but may be valid at boundaries.
/// Example: "ok!" is fine, but "o!k" is not.
#[inline(always)]
fn is_invalid_interior_symbol(ch: char) -> bool {
    INVALID_INTERIOR_SYMBOLS.contains(&ch)
}

/// O(1) sentence ending punctuation check.
#[inline(always)]
fn is_sentence_ending(ch: char) -> bool {
    SENTENCE_ENDINGS.contains(&ch)
}

/// O(1) HashSet lookup for stopwords with case normalization.
///
/// # Performance Strategy
/// - Fast path: Check original case first (most text is lowercase)
/// - Slow path: Convert to lowercase only if fast path fails
#[inline]
fn is_stopword(word: &str) -> bool {
    // Fast path: check original case
    if STOPWORDS_SET.contains(word) {
        return true;
    }
    // Slow path: lowercase conversion
    let lower = word.to_lowercase();
    STOPWORDS_SET.contains(lower.as_str())
}

/// O(1) HashSet lookup for verbs and prepositions.
#[inline]
fn is_verb_or_preposition(word: &str) -> bool {
    if VERBS_PREPOSITIONS_SET.contains(word) {
        return true;
    }
    let lower = word.to_lowercase();
    VERBS_PREPOSITIONS_SET.contains(lower.as_str())
}

/// O(1) HashSet lookup for abbreviations.
#[inline]
fn is_abbreviation(word: &str) -> bool {
    ABBREVIATIONS_SET.contains(word)
}

// ============================================================================
// VOWEL COUNT VALIDATION
// ============================================================================

/// Validates vowel count against empirically-derived length-based ranges.
///
/// # Linguistic Basis
/// Based on analysis of 215,784 Wikipedia article words:
/// - Words have predictable vowel-to-length ratios
/// - Deviations indicate non-words (codes, gibberish, etc.)
///
/// # Implementation
/// Uses match statement that compiler optimizes to O(1) jump table.
///
/// # Arguments
/// * `word` - The word to validate
///
/// # Returns
/// * `true` if vowel count is valid for word length
/// * `false` if word length out of range (2-18) or invalid vowel count
#[inline]
fn check_vowel_count_for_length(word: &str) -> bool {
    let word_len = word.chars().count(); // Use char count for Unicode safety

    // Words outside 2-18 character range fail validation
    if word_len < 2 || word_len > 18 {
        return false;
    }

    let vowel_count = word.chars().filter(|&ch| is_vowel(ch)).count();

    // Compiler optimizes to jump table - O(1) lookup
    match word_len {
        2 => vowel_count == 1,
        3 | 4 | 5 => (1..=3).contains(&vowel_count),
        6 => (1..=4).contains(&vowel_count),
        7 => (1..=5).contains(&vowel_count),
        8 => (2..=5).contains(&vowel_count),
        9 | 10 => (2..=6).contains(&vowel_count),
        11 => (3..=6).contains(&vowel_count),
        12 | 13 => (3..=7).contains(&vowel_count),
        14 => (4..=7).contains(&vowel_count),
        15 => (5..=8).contains(&vowel_count),
        16 => (6..=7).contains(&vowel_count),
        17 | 18 => (6..=8).contains(&vowel_count),
        _ => false,
    }
}

// ============================================================================
// WORD VALIDATION
// ============================================================================

/// Validates a word candidate against English word patterns.
///
/// # Project Context
/// This is the core word-level filter in the language detection pipeline.
/// Words that pass validation are counted; sentences containing enough
/// valid words with proper grammatical markers indicate language presence.
///
/// # Validation Rules
/// 1. Non-empty and not all whitespace
/// 2. Interior characters (excluding first/last) contain no invalid symbols
/// 3. Vowel count appropriate for word length
///
/// # Performance
/// - ASCII fast path for common English text (~95% of input)
/// - Early exits on validation failures
/// - Minimal allocations
///
/// # Arguments
/// * `word` - Candidate word to validate
///
/// # Returns
/// * `true` if word passes all validation checks
/// * `false` otherwise
#[inline]
pub fn is_valid_english_word(word: &str) -> bool {
    // Early exit: empty
    if word.is_empty() {
        return false;
    }

    // Early exit: all whitespace
    let trimmed = word.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Collect chars for safe Unicode handling
    let chars: Vec<char> = word.chars().collect();
    let char_count = chars.len();

    // Check interior characters for invalid symbols
    if char_count >= 3 {
        // Check chars[1..char_count-1] (interior only)
        for i in 1..(char_count - 1) {
            if is_invalid_interior_symbol(chars[i]) {
                return false;
            }
        }
    } else {
        // Short words: check all characters
        for &ch in &chars {
            if is_invalid_interior_symbol(ch) {
                return false;
            }
        }
    }

    // Final validation: vowel count appropriate for length
    check_vowel_count_for_length(word)
}

// ============================================================================
// TEXT PREPROCESSING
// ============================================================================
/// Removes consecutive duplicate characters from specified set.
///
/// # Project Context
/// Preprocessing step to normalize spacing and punctuation before
/// word/sentence splitting. Handles common text artifacts like
/// multiple spaces, repeated dashes, etc.
///
/// # Arguments
/// * `text` - Input text to process
/// * `chars_to_dedupe` - Optional custom character set; defaults to spaces/dashes
///
/// # Returns
/// String with consecutive duplicates of target characters collapsed to single
///
/// # Performance
/// - Pre-allocated capacity to minimize reallocations
/// - Single pass through string
/// - Combinator pattern avoids match branching overhead
#[inline]
pub fn remove_duplicate_chars(text: &str, chars_to_dedupe: Option<&[char]>) -> String {
    const DEFAULT_DEDUPE_CHARS: [char; 4] = [' ', '-', '–', '—'];
    let target_chars = chars_to_dedupe.unwrap_or(&DEFAULT_DEDUPE_CHARS);

    if text.is_empty() {
        return String::new();
    }

    // Pre-allocate with input capacity (likely need most of it)
    let mut result = String::with_capacity(text.len());
    let mut prev_char: Option<char> = None;

    for ch in text.chars() {
        // Combinator pattern: avoids rust-analyzer false positive on None pattern
        // Also more idiomatic Rust for Option handling
        let should_skip = prev_char
            .map(|prev| target_chars.contains(&ch) && ch == prev)
            .unwrap_or(false);

        if !should_skip {
            result.push(ch);
            prev_char = Some(ch);
        }
    }

    result
}

/// Sanitizes and splits text into potential word candidates.
///
/// # Project Context
/// First stage of the language detection pipeline:
/// 1. Normalize whitespace (newlines, tabs -> spaces)
/// 2. Remove duplicate characters
/// 3. Ensure periods have following space (sentence separation)
/// 4. Split on whitespace
///
/// # Arguments
/// * `raw_text` - Raw input text, may contain multiple lines
///
/// # Returns
/// Vector of potential word strings (may include punctuation)
///
/// # Performance
/// - Minimizes string allocations through replace chaining
/// - Pre-split capacity estimation
pub fn sanitize_and_split_text(raw_text: &str) -> Vec<String> {
    if raw_text.is_empty() {
        return Vec::new();
    }

    // Check if all whitespace without allocation
    if raw_text.chars().all(|ch| ch.is_whitespace()) {
        return Vec::new();
    }

    // Step 1: Remove duplicate characters
    let text = remove_duplicate_chars(raw_text, None);

    // Step 2-3: Normalize whitespace and ensure period spacing
    let normalized = text
        .replace('\n', " ")
        .replace('\t', " ")
        .replace(".", ". ");

    // Step 4: Split and collect
    normalized.split_whitespace().map(String::from).collect()
}

// ============================================================================
// SENTENCE PROCESSING
// ============================================================================

/// Splits oversized sentences into manageable segments.
///
/// # Project Context
/// Sentences exceeding MAX_WORDS_PER_SENTENCE are split at
/// SPLIT_SENTENCES_ON_N_WORDS intervals. Each segment is then
/// independently validated for grammatical markers.
///
/// # Arguments
/// * `sentence_words` - Slice of words comprising the sentence
///
/// # Returns
/// Vector of word vectors, each representing a sentence segment
fn split_over_max_sentence(sentence_words: &[String]) -> Vec<Vec<String>> {
    let num_words = sentence_words.len();

    if num_words <= MAX_WORDS_PER_SENTENCE {
        return vec![sentence_words.to_vec()];
    }

    // Pre-calculate segment count for allocation
    let segment_count = (num_words + SPLIT_SENTENCES_ON_N_WORDS - 1) / SPLIT_SENTENCES_ON_N_WORDS;
    let mut segments = Vec::with_capacity(segment_count);

    let mut i = 0;
    // Bounded loop: maximum iterations = segment_count
    let max_iterations = segment_count + 1; // Safety bound
    let mut iteration = 0;

    while i < num_words && iteration < max_iterations {
        let end = std::cmp::min(i + SPLIT_SENTENCES_ON_N_WORDS, num_words);
        segments.push(sentence_words[i..end].to_vec());
        i = end;
        iteration += 1;
    }

    segments
}

/// Counts grammatical markers in a word slice.
///
/// # Returns
/// Tuple of (verb_preposition_count, stopword_count)
#[inline]
fn count_grammatical_markers(words: &[String]) -> (usize, usize) {
    let mut verb_count = 0;
    let mut stopword_count = 0;

    for word in words {
        if is_verb_or_preposition(word) {
            verb_count += 1;
        }
        if is_stopword(word) {
            stopword_count += 1;
        }
    }

    (verb_count, stopword_count)
}

/// Splits word list into validated sentences.
///
/// # Project Context
/// Core sentence extraction and validation:
/// 1. Split on sentence-ending punctuation (handling abbreviations)
/// 2. Filter by minimum word count
/// 3. Validate grammatical marker presence
/// 4. Split oversized sentences and re-validate segments
///
/// # Validation Criteria
/// - At least MIN_WORDS_PER_SENTENCE words
/// - At least MIN_VERBS_PREPOSITIONS_PER_SENTENCE verbs/prepositions
/// - At least MIN_NLTK_STOPWORDS_PER_SENTENCE stopwords
///
/// # Arguments
/// * `words` - Slice of validated word strings
///
/// # Returns
/// Vector of valid sentences, each as a vector of words
pub fn split_wordlist_into_sentences_and_filter(words: &[String]) -> Vec<Vec<String>> {
    if words.is_empty() {
        return Vec::new();
    }

    // Pre-allocate with reasonable capacity
    let estimated_sentences = std::cmp::max(1, words.len() / 10);
    let mut sentences: Vec<Vec<String>> = Vec::with_capacity(estimated_sentences);
    let mut current_sentence: Vec<String> = Vec::with_capacity(20);

    // PASS 1: Split on sentence-ending punctuation
    for word in words {
        let last_char = word.chars().last();
        let ends_with_punct = last_char.map(is_sentence_ending).unwrap_or(false);

        if ends_with_punct && !is_abbreviation(word) {
            // Safe: we know there's at least one char
            if let Some(punct_char) = last_char {
                let word_without_punct: String =
                    word.chars().take(word.chars().count() - 1).collect();

                if !word_without_punct.is_empty() {
                    current_sentence.push(word_without_punct);
                }
                current_sentence.push(punct_char.to_string());

                if !current_sentence.is_empty() {
                    sentences.push(std::mem::replace(
                        &mut current_sentence,
                        Vec::with_capacity(20),
                    ));
                }
            }
        } else {
            current_sentence.push(word.clone());
        }
    }

    // Handle remaining words (sentence without ending punctuation)
    if !current_sentence.is_empty() {
        let needs_period = current_sentence
            .last()
            .and_then(|w| w.chars().last())
            .map(|ch| !is_sentence_ending(ch))
            .unwrap_or(true);

        if needs_period {
            current_sentence.push(".".to_string());
        }
        sentences.push(current_sentence);
    }

    // PASS 2: Validate and filter sentences
    let mut valid_sentences: Vec<Vec<String>> = Vec::with_capacity(sentences.len());

    for mut sentence in sentences {
        // Remove trailing punctuation for word counting
        if let Some(last) = sentence.last() {
            if last == "." || last == "!" || last == "?" {
                sentence.pop();
            }
        }

        if sentence.is_empty() {
            continue;
        }

        // Early exit: minimum word count (cheapest check)
        if sentence.len() < MIN_WORDS_PER_SENTENCE {
            continue;
        }

        // Count grammatical markers
        let (verb_count, stopword_count) = count_grammatical_markers(&sentence);

        // Validate grammatical requirements
        if verb_count < MIN_VERBS_PREPOSITIONS_PER_SENTENCE
            || stopword_count < MIN_NLTK_STOPWORDS_PER_SENTENCE
        {
            continue;
        }

        // Handle oversized sentences
        if sentence.len() > MAX_WORDS_PER_SENTENCE {
            let segments = split_over_max_sentence(&sentence);

            for segment in segments {
                if segment.len() < MIN_WORDS_PER_SENTENCE {
                    continue;
                }

                let (seg_verb_count, seg_stopword_count) = count_grammatical_markers(&segment);

                if seg_verb_count >= MIN_VERBS_PREPOSITIONS_PER_SENTENCE
                    && seg_stopword_count >= MIN_NLTK_STOPWORDS_PER_SENTENCE
                {
                    valid_sentences.push(segment);
                }
            }
        } else {
            valid_sentences.push(sentence);
        }
    }

    valid_sentences
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

/// Language detection errors.
///
/// # Design
/// - No heap allocation in error variants
/// - Unique prefix "LDWSC" for traceability
/// - Copy/Clone for efficient passing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangDetectError {
    /// Input was empty or whitespace-only
    EmptyInput,
    /// No valid words found after filtering
    NoValidWords,
    /// Processing encountered unexpected state
    ProcessingError,
}

impl std::fmt::Display for LangDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LangDetectError::EmptyInput => write!(f, "LDWSC: empty input"),
            LangDetectError::NoValidWords => write!(f, "LDWSC: no valid words"),
            LangDetectError::ProcessingError => write!(f, "LDWSC: processing error"),
        }
    }
}

impl std::error::Error for LangDetectError {}

/// Result type alias for language detection operations.
pub type LangDetectResult<T> = Result<T, LangDetectError>;

// ============================================================================
// PUBLIC API - Main detection functions
// ============================================================================

/// Analyzes text and returns both word and sentence counts.
///
/// # Project Context
/// Main entry point for language detection. A return of (_, 0) sentences
/// strongly indicates non-language input. Higher counts indicate
/// higher confidence of valid English text.
///
/// # Arguments
/// * `input_text` - Raw text to analyze
///
/// # Returns
/// * `Ok((word_count, sentence_count))` - Counts of valid words and sentences
/// * `Err(LangDetectError)` - If input is empty or contains no valid words
///
/// # Example
/// ```
/// let result = lang_detect_word_sentence_counter("This is a test sentence.");
/// match result {
///     Ok((words, sentences)) => println!("Words: {}, Sentences: {}", words, sentences),
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
pub fn lang_detect_word_sentence_counter(input_text: &str) -> LangDetectResult<(usize, usize)> {
    // Validate input
    if input_text.is_empty() {
        return Err(LangDetectError::EmptyInput);
    }

    if input_text.chars().all(|ch| ch.is_whitespace()) {
        return Err(LangDetectError::EmptyInput);
    }

    // Normalize and split
    let words = sanitize_and_split_text(input_text);

    if words.is_empty() {
        return Err(LangDetectError::EmptyInput);
    }

    // Filter valid words
    let valid_words: Vec<String> = words
        .into_iter()
        .filter(|word| is_valid_english_word(word))
        .collect();

    if valid_words.is_empty() {
        return Err(LangDetectError::NoValidWords);
    }

    // Extract and validate sentences
    let valid_sentences = split_wordlist_into_sentences_and_filter(&valid_words);

    Ok((valid_words.len(), valid_sentences.len()))
}

/// Returns only the valid word count (optimized path).
///
/// # Use Case
/// When only word count is needed, avoids sentence processing overhead.
///
/// # Arguments
/// * `input_text` - Raw text to analyze
///
/// # Returns
/// * `Ok(word_count)` - Count of valid English words
/// * `Err(LangDetectError)` - If input is empty or contains no valid words
pub fn lang_detect_word_count_only(input_text: &str) -> LangDetectResult<usize> {
    if input_text.is_empty() || input_text.chars().all(|ch| ch.is_whitespace()) {
        return Err(LangDetectError::EmptyInput);
    }

    let words = sanitize_and_split_text(input_text);

    if words.is_empty() {
        return Err(LangDetectError::EmptyInput);
    }

    let valid_word_count = words
        .iter()
        .filter(|word| is_valid_english_word(word))
        .count();

    if valid_word_count == 0 {
        return Err(LangDetectError::NoValidWords);
    }

    Ok(valid_word_count)
}

/// Returns only the valid sentence count (requires full processing).
///
/// # Use Case
/// When only sentence count is needed for quick language presence check.
///
/// # Arguments
/// * `input_text` - Raw text to analyze
///
/// # Returns
/// * `Ok(sentence_count)` - Count of valid sentences
/// * `Err(LangDetectError)` - If input is empty or contains no valid words
pub fn lang_detect_sentence_count_only(input_text: &str) -> LangDetectResult<usize> {
    let (_, sentence_count) = lang_detect_word_sentence_counter(input_text)?;
    Ok(sentence_count)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Test Data - Using const arrays avoids slice coercion issues
    // ------------------------------------------------------------------------

    /// Invalid/incomplete test cases - should have 0 sentences
    const INVALID_INCOMPLETE_TEST_CASES: [&str; 2] = ["\"buy $$$\"", "\"BUY SPAM BUY SPAM!\""];

    /// Valid short test cases - should have >= 1 sentence
    const VALID_SHORT_TEST_CASES: [&str; 3] = [
        "He had a great time there.",
        "This is sentence one.",
        "This is really a sentence.",
    ];

    /// Valid sample cases - should have >= 1 sentence
    const VALID_SAMPLE_CASES: [&str; 5] = [
        "Mr. Smith went to Washington. He had a great time!",
        "This is sentence one. This is sentence two! What about three?",
        "Short. Too short. This is a proper sentence.",
        "Dr. Jones, Prof. Smith and Mrs. Brown attended the meeting on Downing St. Inc.",
        "This sentence is not incomplete.",
    ];

    /// Valid borderline test cases - should have >= 1 sentence
    const VALID_BORDERLINE_TEST_CASES: [&str; 2] = [
        "This is a proper sentence.",
        "Short. Too short. This is a proper sentence.",
    ];

    /// Edge cases probably invalid - should have 0 sentences
    const EDGE_CASE_PROBABLY_INVALID: [&str; 3] = [
        "buy $$$",
        "SPAM SPAM!",
        "fraud@crypto is the B!!!est slimball.",
    ];

    // ========================================================================
    // Unit Tests for Helper Functions
    // ========================================================================

    #[test]
    fn test_isvowel() {
        assert!(is_vowel('a'));
        assert!(is_vowel('E'));
        assert!(is_vowel('y'));
        assert!(is_vowel('Y'));
        assert!(!is_vowel('b'));
        assert!(!is_vowel('Z'));
    }

    #[test]
    fn test_check_vowel_count_for_length() {
        assert!(check_vowel_count_for_length("hello"));
        assert!(check_vowel_count_for_length("the"));
        assert!(check_vowel_count_for_length("is"));
        assert!(!check_vowel_count_for_length("a"));
        assert!(!check_vowel_count_for_length(
            "supercalifragilisticexpialidocious"
        ));
    }

    #[test]
    fn test_is_valid_english_word() {
        assert!(is_valid_english_word("hello"));
        assert!(is_valid_english_word("world"));
        assert!(is_valid_english_word("the"));
        assert!(!is_valid_english_word(""));
        assert!(!is_valid_english_word("he#llo"));
        assert!(!is_valid_english_word("wo@rld"));
        assert!(is_valid_english_word("hello!"));
        assert!(is_valid_english_word("(hello)"));
    }

    #[test]
    fn test_remove_duplicate_chars() {
        assert_eq!(remove_duplicate_chars("hello  world", None), "hello world");
        assert_eq!(remove_duplicate_chars("a--b", None), "a-b");
        assert_eq!(remove_duplicate_chars("", None), "");
        assert_eq!(
            remove_duplicate_chars("no duplicates", None),
            "no duplicates"
        );
    }

    #[test]
    fn test_sanitize_and_split_text() {
        let result = sanitize_and_split_text("Hello world");
        assert_eq!(result, vec!["Hello", "world"]);

        let result = sanitize_and_split_text("Hello\nworld");
        assert_eq!(result, vec!["Hello", "world"]);

        let result = sanitize_and_split_text("");
        assert!(result.is_empty());

        let result = sanitize_and_split_text("   ");
        assert!(result.is_empty());
    }

    // ========================================================================
    // Integration Tests - Original Python Test Cases
    // ========================================================================

    #[test]
    fn test_valid_short_cases() {
        for test_case in VALID_SHORT_TEST_CASES.iter() {
            let result = lang_detect_word_sentence_counter(test_case);
            match result {
                Ok((_, sentences)) => {
                    assert!(
                        sentences > 0,
                        "Expected sentences > 0 for: '{}', got: {}",
                        test_case,
                        sentences
                    );
                }
                Err(e) => {
                    panic!("Unexpected error for '{}': {}", test_case, e);
                }
            }
        }
    }

    #[test]
    fn test_empty_invalid() {
        for test_case in INVALID_INCOMPLETE_TEST_CASES.iter() {
            let result = lang_detect_word_sentence_counter(test_case);
            match result {
                Ok((_, sentences)) => {
                    assert_eq!(
                        sentences, 0,
                        "Expected 0 sentences for invalid: '{}', got: {}",
                        test_case, sentences
                    );
                }
                Err(_) => {}
            }
        }
    }

    #[test]
    fn test_edge_case_invalid() {
        for test_case in EDGE_CASE_PROBABLY_INVALID.iter() {
            let result = lang_detect_word_sentence_counter(test_case);
            match result {
                Ok((_, sentences)) => {
                    assert_eq!(
                        sentences, 0,
                        "Expected 0 sentences for edge case: '{}', got: {}",
                        test_case, sentences
                    );
                }
                Err(_) => {}
            }
        }
    }

    #[test]
    fn test_valid_borderline() {
        for test_case in VALID_BORDERLINE_TEST_CASES.iter() {
            let result = lang_detect_word_sentence_counter(test_case);
            match result {
                Ok((_, sentences)) => {
                    assert!(
                        sentences > 0,
                        "Expected sentences > 0 for borderline: '{}', got: {}",
                        test_case,
                        sentences
                    );
                }
                Err(e) => {
                    panic!("Unexpected error for '{}': {}", test_case, e);
                }
            }
        }
    }

    #[test]
    fn test_valid_samples() {
        for test_case in VALID_SAMPLE_CASES.iter() {
            let result = lang_detect_word_sentence_counter(test_case);
            match result {
                Ok((_, sentences)) => {
                    assert!(
                        sentences > 0,
                        "Expected sentences > 0 for sample: '{}', got: {}",
                        test_case,
                        sentences
                    );
                }
                Err(e) => {
                    panic!("Unexpected error for '{}': {}", test_case, e);
                }
            }
        }
    }

    #[test]
    fn test_short_inline_cases() {
        let short_test_cases = [
            "please reply to my request about weather, tom",
            "This is sentence one. This is sentence two. Here is!",
        ];

        for test_case in short_test_cases.iter() {
            let result = lang_detect_word_sentence_counter(test_case);
            match result {
                Ok((_, sentences)) => {
                    assert!(
                        sentences > 0,
                        "Expected sentences > 0 for: '{}', got: {}",
                        test_case,
                        sentences
                    );
                }
                Err(e) => {
                    panic!("Unexpected error for '{}': {}", test_case, e);
                }
            }
        }
    }

    #[test]
    fn test_word_count_only_api() {
        let result = lang_detect_word_count_only("Hello world this is a test.");
        assert!(result.is_ok());
        assert!(result.unwrap_or(0) > 0);
    }

    #[test]
    fn test_sentence_count_only_api() {
        let result = lang_detect_sentence_count_only("This is a valid sentence.");
        assert!(result.is_ok());
        assert!(result.unwrap_or(0) > 0);
    }

    #[test]
    fn test_empty_input_handling() {
        assert!(matches!(
            lang_detect_word_sentence_counter(""),
            Err(LangDetectError::EmptyInput)
        ));

        assert!(matches!(
            lang_detect_word_sentence_counter("   "),
            Err(LangDetectError::EmptyInput)
        ));

        assert!(matches!(
            lang_detect_word_count_only(""),
            Err(LangDetectError::EmptyInput)
        ));
    }

    #[test]
    fn test_lazy_lock_initialization() {
        assert!(!STOPWORDS_SET.is_empty());
        assert!(!VERBS_PREPOSITIONS_SET.is_empty());
        assert!(!ABBREVIATIONS_SET.is_empty());
    }

    #[test]
    fn test_hashset_lookups() {
        assert!(is_stopword("the"));
        assert!(is_stopword("THE"));
        assert!(!is_stopword("xyzzy"));

        assert!(is_verb_or_preposition("is"));
        assert!(is_verb_or_preposition("through"));
        assert!(!is_verb_or_preposition("xyzzy"));

        assert!(is_abbreviation("Mr."));
        assert!(is_abbreviation("dr."));
        assert!(!is_abbreviation("hello"));
    }

    // ========================================================================
    // ADDITIONAL TESTS - Based on Python Specifications
    // ========================================================================

    // ------------------------------------------------------------------------
    // Vowel Count Validation Tests (LEN_TO_N_VOWELS rules)
    // Based on empirical Wikipedia word analysis
    // ------------------------------------------------------------------------

    #[test]
    fn test_vowel_count_length_2() {
        // Length 2: must have exactly 1 vowel
        assert!(check_vowel_count_for_length("is")); // 1 vowel
        assert!(check_vowel_count_for_length("at")); // 1 vowel
        assert!(check_vowel_count_for_length("on")); // 1 vowel
        assert!(check_vowel_count_for_length("it")); // 1 vowel
        assert!(!check_vowel_count_for_length("bb")); // 0 vowels - invalid
    }

    #[test]
    fn test_vowel_count_length_3_to_5() {
        // Length 3-5: must have 1-3 vowels
        assert!(check_vowel_count_for_length("the")); // 3 chars, 1 vowel
        assert!(check_vowel_count_for_length("cat")); // 3 chars, 1 vowel
        assert!(check_vowel_count_for_length("see")); // 3 chars, 2 vowels
        assert!(check_vowel_count_for_length("area")); // 4 chars, 3 vowels
        assert!(check_vowel_count_for_length("hello")); // 5 chars, 2 vowels
        assert!(!check_vowel_count_for_length("brrr")); // 4 chars, 0 vowels - invalid
    }

    #[test]
    fn test_vowel_count_length_6_to_7() {
        // Length 6: 1-4 vowels; Length 7: 1-5 vowels
        assert!(check_vowel_count_for_length("simple")); // 6 chars, 2 vowels
        assert!(check_vowel_count_for_length("people")); // 6 chars, 3 vowels
        assert!(check_vowel_count_for_length("example")); // 7 chars, 3 vowels
        assert!(check_vowel_count_for_length("receive")); // 7 chars, 4 vowels
    }

    #[test]
    fn test_vowel_count_length_8_to_10() {
        // Length 8: 2-5 vowels; Length 9-10: 2-6 vowels
        assert!(check_vowel_count_for_length("sentence")); // 8 chars, 3 vowels
        assert!(check_vowel_count_for_length("beautiful")); // 9 chars, 5 vowels
        assert!(check_vowel_count_for_length("experience")); // 10 chars, 5 vowels
    }

    #[test]
    fn test_vowel_count_length_11_to_13() {
        // Length 11: 3-6 vowels; Length 12-13: 3-7 vowels
        assert!(check_vowel_count_for_length("information")); // 11 chars, 5 vowels
        assert!(check_vowel_count_for_length("professional")); // 12 chars, 5 vowels
        assert!(check_vowel_count_for_length("understanding")); // 13 chars, 5 vowels
    }

    #[test]
    fn test_vowel_count_length_14_to_18() {
        // Length 14: 4-7 vowels; Length 15: 5-8; Length 16: 6-7; Length 17-18: 6-8
        assert!(check_vowel_count_for_length("transportation")); // 14 chars, 5 vowels
        assert!(check_vowel_count_for_length("representatives")); // 15 chars, 6 vowels
    }

    #[test]
    fn test_vowel_count_out_of_range() {
        // Too short (< 2 chars)
        assert!(!check_vowel_count_for_length("a"));
        assert!(!check_vowel_count_for_length("I"));

        // Too long (> 18 chars)
        assert!(!check_vowel_count_for_length("internationalization"));
        assert!(!check_vowel_count_for_length(
            "supercalifragilisticexpialidocious"
        ));
    }

    // ------------------------------------------------------------------------
    // Interior Symbol Detection Tests
    // Symbols OK at boundaries, invalid inside words
    // ------------------------------------------------------------------------

    #[test]
    fn test_interior_symbols_invalid() {
        // Invalid: symbols inside the word
        assert!(!is_valid_english_word("he@llo"));
        assert!(!is_valid_english_word("wo#rld"));
        assert!(!is_valid_english_word("te$st"));
        assert!(!is_valid_english_word("ex%ample"));
        assert!(!is_valid_english_word("he^llo"));
        assert!(!is_valid_english_word("wo&rld"));
        assert!(!is_valid_english_word("te*st"));
        assert!(!is_valid_english_word("he<llo"));
        assert!(!is_valid_english_word("wo>rld"));
        assert!(!is_valid_english_word("te{st"));
        assert!(!is_valid_english_word("he}llo"));
        assert!(!is_valid_english_word("wo[rld"));
        assert!(!is_valid_english_word("te]st"));
        assert!(!is_valid_english_word("he\\llo"));
        assert!(!is_valid_english_word("wo|rld"));
    }

    #[test]
    fn test_boundary_symbols_valid() {
        // Valid: symbols at word boundaries (first or last char)
        assert!(is_valid_english_word("hello!"));
        assert!(is_valid_english_word("hello?"));
        assert!(is_valid_english_word("(hello)"));
        assert!(is_valid_english_word("\"hello\""));
        assert!(is_valid_english_word("$hello")); // $ at start is OK
        assert!(is_valid_english_word("hello$")); // $ at end is OK
    }

    #[test]
    fn test_short_words_with_symbols() {
        // For words < 3 chars, all chars are checked
        assert!(!is_valid_english_word("@b"));
        assert!(!is_valid_english_word("a#"));
        assert!(!is_valid_english_word("#$"));
    }

    // ------------------------------------------------------------------------
    // Abbreviation Handling Tests
    // Abbreviations should NOT trigger sentence splits
    // ------------------------------------------------------------------------

    #[test]
    fn test_abbreviation_detection_lowercase() {
        assert!(is_abbreviation("mr."));
        assert!(is_abbreviation("mrs."));
        assert!(is_abbreviation("ms."));
        assert!(is_abbreviation("dr."));
        assert!(is_abbreviation("prof."));
        assert!(is_abbreviation("sr."));
        assert!(is_abbreviation("jr."));
        assert!(is_abbreviation("vs."));
        assert!(is_abbreviation("etc."));
        assert!(is_abbreviation("e.g."));
        assert!(is_abbreviation("i.e."));
        assert!(is_abbreviation("st."));
        assert!(is_abbreviation("inc."));
    }

    #[test]
    fn test_abbreviation_detection_titlecase() {
        assert!(is_abbreviation("Mr."));
        assert!(is_abbreviation("Mrs."));
        assert!(is_abbreviation("Ms."));
        assert!(is_abbreviation("Dr."));
        assert!(is_abbreviation("Prof."));
        assert!(is_abbreviation("Sr."));
        assert!(is_abbreviation("Jr."));
        assert!(is_abbreviation("Vs."));
        assert!(is_abbreviation("Etc."));
        assert!(is_abbreviation("St."));
        assert!(is_abbreviation("Inc."));
    }

    #[test]
    fn test_abbreviation_in_sentence() {
        // Sentences with abbreviations should NOT split at the abbreviation
        let result = lang_detect_word_sentence_counter(
            "Dr. Smith and Mrs. Jones went to St. Louis together.",
        );
        match result {
            Ok((_, sentences)) => {
                // Should be 1 sentence, not split at Dr. Mrs. St.
                assert_eq!(sentences, 1, "Abbreviations should not split sentences");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_abbreviation_not_false_positive() {
        // Regular words ending in period should NOT be treated as abbreviations
        assert!(!is_abbreviation("hello."));
        assert!(!is_abbreviation("world."));
        assert!(!is_abbreviation("test."));
    }

    // ------------------------------------------------------------------------
    // Sentence Structure Tests
    // MIN_WORDS_PER_SENTENCE = 4
    // MIN_VERBS_PREPOSITIONS_PER_SENTENCE = 1
    // MIN_NLTK_STOPWORDS_PER_SENTENCE = 1
    // ------------------------------------------------------------------------

    #[test]
    fn test_minimum_words_per_sentence() {
        // Exactly 4 words with required elements = valid
        let result = lang_detect_word_sentence_counter("This really is a test.");
        assert!(matches!(result, Ok((_, sentences)) if sentences == 1));

        // 3 words = invalid (below minimum)
        let result = lang_detect_word_sentence_counter("This is test.");
        match result {
            Ok((_, sentences)) => assert_eq!(sentences, 0, "3 words should be invalid"),
            Err(_) => {} // Error also acceptable
        }
    }

    #[test]
    fn test_sentence_requires_verb_or_preposition() {
        // Sentence with verb = valid
        let result = lang_detect_word_sentence_counter("The cat is sleeping.");
        assert!(matches!(result, Ok((_, sentences)) if sentences == 1));

        // Sentence with preposition = valid
        let result = lang_detect_word_sentence_counter("The cat on the mat.");
        assert!(matches!(result, Ok((_, sentences)) if sentences == 1));
    }

    #[test]
    fn test_sentence_requires_stopword() {
        // Sentence with stopwords = valid
        let result = lang_detect_word_sentence_counter("The quick brown fox jumps over.");
        assert!(matches!(result, Ok((_, sentences)) if sentences == 1));
    }

    #[test]
    fn test_multiple_sentences() {
        // Multiple valid sentences
        let result = lang_detect_word_sentence_counter(
            "This is sentence one. This is sentence two. This is sentence three.",
        );
        match result {
            Ok((_, sentences)) => {
                assert!(
                    sentences >= 2,
                    "Expected multiple sentences, got {}",
                    sentences
                );
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_sentence_with_exclamation() {
        let result = lang_detect_word_sentence_counter("This is really exciting!");
        assert!(matches!(result, Ok((_, sentences)) if sentences == 1));
    }

    #[test]
    fn test_sentence_with_question() {
        let result = lang_detect_word_sentence_counter("Is this a valid question?");
        assert!(matches!(result, Ok((_, sentences)) if sentences == 1));
    }

    // ------------------------------------------------------------------------
    // Duplicate Character Removal Tests
    // Handles: spaces, dashes (-, –, —)
    // ------------------------------------------------------------------------

    #[test]
    fn test_duplicate_spaces_removed() {
        let result = remove_duplicate_chars("hello    world", None);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_duplicate_dashes_removed() {
        let result = remove_duplicate_chars("hello---world", None);
        assert_eq!(result, "hello-world");

        let result = remove_duplicate_chars("hello–––world", None); // en-dash
        assert_eq!(result, "hello–world");

        let result = remove_duplicate_chars("hello———world", None); // em-dash
        assert_eq!(result, "hello—world");
    }

    #[test]
    fn test_mixed_duplicates_removed() {
        let result = remove_duplicate_chars("hello  --  world", None);
        assert_eq!(result, "hello - world");
    }

    #[test]
    fn test_custom_dedupe_chars() {
        let result = remove_duplicate_chars("aaa bbb ccc", Some(&['a', 'b']));
        assert_eq!(result, "a b ccc");
    }

    #[test]
    fn test_no_duplicates_unchanged() {
        let result = remove_duplicate_chars("hello world", None);
        assert_eq!(result, "hello world");
    }

    // ------------------------------------------------------------------------
    // Text Sanitization Tests
    // Newlines, tabs converted to spaces; periods get trailing space
    // ------------------------------------------------------------------------

    #[test]
    fn test_newlines_converted_to_spaces() {
        let result = sanitize_and_split_text("hello\nworld");
        assert_eq!(result.len(), 2, "Should split into 2 words");
        assert_eq!(result[0], "hello");
        assert_eq!(result[1], "world");
    }

    #[test]
    fn test_tabs_converted_to_spaces() {
        let result = sanitize_and_split_text("hello\tworld");
        assert_eq!(result.len(), 2, "Should split into 2 words");
        assert_eq!(result[0], "hello");
        assert_eq!(result[1], "world");
    }

    #[test]
    fn test_periods_get_spacing() {
        let result = sanitize_and_split_text("hello.world");
        // Should become "hello. world" then split to ["hello", ".", "world"] or similar
        assert!(result.len() >= 2, "Period should cause split");
    }

    #[test]
    fn test_whitespace_only_returns_empty() {
        assert!(sanitize_and_split_text("").is_empty());
        assert!(sanitize_and_split_text("   ").is_empty());
        assert!(sanitize_and_split_text("\n\n\n").is_empty());
        assert!(sanitize_and_split_text("\t\t\t").is_empty());
        assert!(sanitize_and_split_text("  \n  \t  ").is_empty());
    }

    // ------------------------------------------------------------------------
    // Stopword Detection Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_stopword_detection_common_words() {
        // Common NLTK stopwords
        assert!(is_stopword("the"));
        assert!(is_stopword("a"));
        assert!(is_stopword("an"));
        assert!(is_stopword("and"));
        assert!(is_stopword("or"));
        assert!(is_stopword("but"));
        assert!(is_stopword("is"));
        assert!(is_stopword("are"));
        assert!(is_stopword("was"));
        assert!(is_stopword("were"));
        assert!(is_stopword("have"));
        assert!(is_stopword("has"));
        assert!(is_stopword("had"));
        assert!(is_stopword("do"));
        assert!(is_stopword("does"));
        assert!(is_stopword("did"));
    }

    #[test]
    fn test_stopword_detection_pronouns() {
        assert!(is_stopword("i"));
        assert!(is_stopword("I")); // Case insensitive
        assert!(is_stopword("me"));
        assert!(is_stopword("my"));
        assert!(is_stopword("you"));
        assert!(is_stopword("your"));
        assert!(is_stopword("he"));
        assert!(is_stopword("she"));
        assert!(is_stopword("it"));
        assert!(is_stopword("they"));
        assert!(is_stopword("we"));
    }

    #[test]
    fn test_stopword_case_insensitive() {
        assert!(is_stopword("THE"));
        assert!(is_stopword("The"));
        assert!(is_stopword("AND"));
        assert!(is_stopword("And"));
    }

    #[test]
    fn test_non_stopwords() {
        assert!(!is_stopword("hello"));
        assert!(!is_stopword("world"));
        assert!(!is_stopword("computer"));
        assert!(!is_stopword("language"));
        assert!(!is_stopword("xyzzy"));
    }

    // ------------------------------------------------------------------------
    // Verb and Preposition Detection Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_common_verbs() {
        assert!(is_verb_or_preposition("is"));
        assert!(is_verb_or_preposition("are"));
        assert!(is_verb_or_preposition("was"));
        assert!(is_verb_or_preposition("were"));
        assert!(is_verb_or_preposition("have"));
        assert!(is_verb_or_preposition("has"));
        assert!(is_verb_or_preposition("go"));
        assert!(is_verb_or_preposition("went"));
        assert!(is_verb_or_preposition("going"));
        assert!(is_verb_or_preposition("make"));
        assert!(is_verb_or_preposition("made"));
        assert!(is_verb_or_preposition("take"));
        assert!(is_verb_or_preposition("took"));
        assert!(is_verb_or_preposition("taken"));
    }

    #[test]
    fn test_common_prepositions() {
        assert!(is_verb_or_preposition("about"));
        assert!(is_verb_or_preposition("above"));
        assert!(is_verb_or_preposition("by"));
        assert!(is_verb_or_preposition("for"));
        // assert!(is_verb_or_preposition("with"));
        assert!(is_verb_or_preposition("about"));
        assert!(is_verb_or_preposition("through"));
        assert!(is_verb_or_preposition("between"));
        assert!(is_verb_or_preposition("under"));
        assert!(is_verb_or_preposition("over"));
        assert!(is_verb_or_preposition("after"));
        assert!(is_verb_or_preposition("before"));
    }

    #[test]
    fn test_verb_preposition_case_insensitive() {
        assert!(is_verb_or_preposition("IS"));
        assert!(is_verb_or_preposition("Is"));
        assert!(is_verb_or_preposition("THROUGH"));
        assert!(is_verb_or_preposition("Through"));
    }

    // ------------------------------------------------------------------------
    // Long Sentence Splitting Tests
    // MAX_WORDS_PER_SENTENCE = 100, SPLIT_SENTENCES_ON_N_WORDS = 30
    // ------------------------------------------------------------------------

    #[test]
    fn test_split_over_max_sentence_under_limit() {
        let words: Vec<String> = (0..20).map(|i| format!("word{}", i)).collect();
        let result = split_over_max_sentence(&words);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 20);
    }

    #[test]
    fn test_split_over_max_sentence_at_limit() {
        let words: Vec<String> = (0..100).map(|i| format!("word{}", i)).collect();
        let result = split_over_max_sentence(&words);
        assert_eq!(result.len(), 1); // Exactly at limit, no split
    }

    #[test]
    fn test_split_over_max_sentence_over_limit() {
        let words: Vec<String> = (0..101).map(|i| format!("word{}", i)).collect();
        let result = split_over_max_sentence(&words);
        // 101 words split at 30: ceil(101/30) = 4 segments
        assert!(result.len() > 1, "Should split when over limit");

        // First segments should be 30 words
        assert_eq!(result[0].len(), 30);
    }

    #[test]
    fn test_split_over_max_sentence_large() {
        let words: Vec<String> = (0..150).map(|i| format!("word{}", i)).collect();
        let result = split_over_max_sentence(&words);
        // 150 words split at 30: 5 segments
        assert_eq!(result.len(), 5);

        // Verify all words accounted for
        let total_words: usize = result.iter().map(|s| s.len()).sum();
        assert_eq!(total_words, 150);
    }

    // ------------------------------------------------------------------------
    // Non-Language Detection Tests (Invalid Input)
    // Based on Python edge_case_probably_invalid
    // ------------------------------------------------------------------------

    #[test]
    fn test_gibberish_rejected() {
        let gibberish_cases = [
            "asdf jkl; qwer tyui",
            "xxx yyy zzz www",
            "123 456 789 000",
            "!@#$ %^&* ()_+",
        ];

        for case in gibberish_cases.iter() {
            let result = lang_detect_word_sentence_counter(case);
            match result {
                Ok((_, sentences)) => {
                    assert_eq!(
                        sentences, 0,
                        "Gibberish should have 0 sentences: '{}'",
                        case
                    );
                }
                Err(_) => {} // Error also acceptable
            }
        }
    }

    #[test]
    fn test_spam_rejected() {
        let spam_cases = [
            "BUY NOW BUY NOW",
            "FREE $$$ FREE $$$",
            "CLICK HERE CLICK HERE",
            "WIN WIN WIN WIN",
        ];

        for case in spam_cases.iter() {
            let result = lang_detect_word_sentence_counter(case);
            match result {
                Ok((_, sentences)) => {
                    assert_eq!(sentences, 0, "Spam should have 0 sentences: '{}'", case);
                }
                Err(_) => {} // Error also acceptable
            }
        }
    }

    #[test]
    fn test_code_rejected() {
        let code_cases = [
            "var x = 42;",
            "function() { return; }",
            "if (x > 0) { y++ }",
            "SELECT * FROM users",
        ];

        for case in code_cases.iter() {
            let result = lang_detect_word_sentence_counter(case);
            match result {
                Ok((_, sentences)) => {
                    assert_eq!(sentences, 0, "Code should have 0 sentences: '{}'", case);
                }
                Err(_) => {} // Error also acceptable
            }
        }
    }

    // ------------------------------------------------------------------------
    // Valid Language Detection Tests (Positive Cases)
    // Based on Python valid_sample_cases
    // ------------------------------------------------------------------------

    #[test]
    fn test_formal_english_accepted() {
        let formal_cases = [
            "The committee has decided to postpone the meeting.",
            "According to the report, sales increased by twenty percent.",
            "Please submit your application before the deadline.",
            "The project was completed ahead of schedule.",
        ];

        for case in formal_cases.iter() {
            let result = lang_detect_word_sentence_counter(case);
            match result {
                Ok((_, sentences)) => {
                    assert!(sentences > 0, "Formal English should be valid: '{}'", case);
                }
                Err(e) => panic!("Unexpected error for '{}': {}", case, e),
            }
        }
    }

    #[test]
    fn test_informal_english_accepted() {
        let informal_cases = [
            "Hey, how are you doing today?",
            "I think we should go to the park.",
            "That movie was really great, wasn't it?",
            "Let me know if you need any help.",
        ];

        for case in informal_cases.iter() {
            let result = lang_detect_word_sentence_counter(case);
            match result {
                Ok((_, sentences)) => {
                    assert!(
                        sentences > 0,
                        "Informal English should be valid: '{}'",
                        case
                    );
                }
                Err(e) => panic!("Unexpected error for '{}': {}", case, e),
            }
        }
    }

    #[test]
    fn test_complex_sentences_accepted() {
        let complex_cases = [
            "Although it was raining, we decided to go for a walk in the park.",
            "The book that I borrowed from the library was very interesting.",
            "She said that she would come to the party if she finished her work.",
        ];

        for case in complex_cases.iter() {
            let result = lang_detect_word_sentence_counter(case);
            match result {
                Ok((_, sentences)) => {
                    assert!(
                        sentences > 0,
                        "Complex sentences should be valid: '{}'",
                        case
                    );
                }
                Err(e) => panic!("Unexpected error for '{}': {}", case, e),
            }
        }
    }

    // ------------------------------------------------------------------------
    // Error Handling Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_error_types_distinct() {
        // EmptyInput
        let err1 = lang_detect_word_sentence_counter("");
        assert!(matches!(err1, Err(LangDetectError::EmptyInput)));

        // NoValidWords - input with only invalid "words"
        let err2 = lang_detect_word_sentence_counter("#### $$$$");
        assert!(matches!(err2, Err(LangDetectError::NoValidWords)));
    }

    #[test]
    fn test_error_display_unique_prefix() {
        let err = LangDetectError::EmptyInput;
        let display = format!("{}", err);
        assert!(
            display.starts_with("LDWSC:"),
            "Error should have unique prefix"
        );

        let err = LangDetectError::NoValidWords;
        let display = format!("{}", err);
        assert!(
            display.starts_with("LDWSC:"),
            "Error should have unique prefix"
        );
    }

    // ------------------------------------------------------------------------
    // Unicode Handling Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_unicode_safety_no_panic() {
        // These should not panic, even if they fail validation
        let unicode_cases = [
            "café résumé naïve",
            "北京 上海 广州",
            "Ελληνικά κείμενο",
            "日本語テスト",
            "🎉 party 🎊 time",
        ];

        for case in unicode_cases.iter() {
            // Should not panic - may return error or low counts
            let _ = lang_detect_word_sentence_counter(case);
        }
    }

    #[test]
    fn test_mixed_unicode_ascii() {
        // Mixed content with some valid English
        let result = lang_detect_word_sentence_counter("The café is open today for business.");
        // Should handle gracefully
        match result {
            Ok((words, _)) => assert!(words > 0),
            Err(_) => {} // Also acceptable
        }
    }

    // ------------------------------------------------------------------------
    // Word Count Only Function Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_word_count_only_basic() {
        let result = lang_detect_word_count_only("Hello world this is a test.");
        assert!(result.is_ok());
        let count = result.unwrap_or(0);
        assert!(count > 0);
    }

    #[test]
    fn test_word_count_only_empty() {
        let result = lang_detect_word_count_only("");
        assert!(matches!(result, Err(LangDetectError::EmptyInput)));
    }

    #[test]
    fn test_word_count_only_whitespace() {
        let result = lang_detect_word_count_only("   \n\t   ");
        assert!(matches!(result, Err(LangDetectError::EmptyInput)));
    }

    // ------------------------------------------------------------------------
    // Sentence Count Only Function Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_sentence_count_only_basic() {
        let result = lang_detect_sentence_count_only("This is a valid sentence.");
        assert!(result.is_ok());
        let count = result.unwrap_or(0);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_sentence_count_only_multiple() {
        let result = lang_detect_sentence_count_only(
            "First sentence is here. Second sentence is here. Third sentence is here.",
        );
        assert!(result.is_ok());
        let count = result.unwrap_or(0);
        print!("test_sentence_count_only_multiple ->{}", count);
        assert!(count == 3);
    }

    #[test]
    fn test_sentence_count_only_empty() {
        let result = lang_detect_sentence_count_only("");
        assert!(matches!(result, Err(LangDetectError::EmptyInput)));
    }

    // ------------------------------------------------------------------------
    // Regression Tests - Specific Edge Cases
    // ------------------------------------------------------------------------

    #[test]
    fn test_sentence_without_ending_punctuation() {
        // Should add period and still validate
        let result = lang_detect_word_sentence_counter("This is a sentence without punctuation");
        match result {
            Ok((_, sentences)) => {
                assert!(
                    sentences > 0,
                    "Sentence without punctuation should be valid"
                );
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_multiple_spaces_between_words() {
        let result = lang_detect_word_sentence_counter("This    is    a    test    sentence.");
        match result {
            Ok((_, sentences)) => {
                assert!(sentences > 0, "Multiple spaces should be handled");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_newlines_between_sentences() {
        let result =
            lang_detect_word_sentence_counter("This is sentence one.\nThis is sentence two.");
        match result {
            Ok((_, sentences)) => {
                assert!(sentences >= 1, "Newlines should be handled");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_tabs_in_text() {
        let result = lang_detect_word_sentence_counter("This\tis\ta\ttest\tsentence.");
        match result {
            Ok((_, sentences)) => {
                assert!(sentences > 0, "Tabs should be handled");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    // ------------------------------------------------------------------------
    // Boundary Condition Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_exactly_min_words_valid() {
        // Exactly MIN_WORDS_PER_SENTENCE (4) with required elements
        let result = lang_detect_word_sentence_counter("Cat is on mat.");
        match result {
            Ok((_, sentences)) => {
                assert_eq!(sentences, 1, "Exactly minimum words should be valid");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_one_below_min_words_invalid() {
        // MIN_WORDS_PER_SENTENCE - 1 = 3 words
        let result = lang_detect_word_sentence_counter("Cat on mat.");
        match result {
            Ok((_, sentences)) => {
                assert_eq!(sentences, 0, "Below minimum words should be invalid");
            }
            Err(_) => {} // Error also acceptable
        }
    }

    #[test]
    fn test_single_word_rejected() {
        let result = lang_detect_word_sentence_counter("Hello.");
        match result {
            Ok((_, sentences)) => {
                assert_eq!(sentences, 0, "Single word should not form sentence");
            }
            Err(_) => {} // Error also acceptable
        }
    }

    #[test]
    fn test_two_words_rejected() {
        let result = lang_detect_word_sentence_counter("Hello world.");
        match result {
            Ok((_, sentences)) => {
                assert_eq!(sentences, 0, "Two words should not form sentence");
            }
            Err(_) => {} // Error also acceptable
        }
    }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

fn main() {
    let test_texts = [
        "Please reply to my request about weather.",
        "This is a test sentence with proper structure.",
        "buy $$$ spam",
        "",
    ];

    for text in test_texts.iter() {
        println!("\nAnalyzing: '{}'", text);

        match lang_detect_word_sentence_counter(text) {
            Ok((words, sentences)) => {
                println!("  Valid words: {}", words);
                println!("  Valid sentences: {}", sentences);

                if sentences > 0 {
                    println!("  Result: Appears to be valid English");
                } else {
                    println!("  Result: No valid sentences detected");
                }
            }
            Err(e) => {
                // Handle gracefully - no panic
                println!("  Result: {}", e);
            }
        }
    }

    // Demonstrate optimized API functions
    println!("\n--- Optimized API Demo ---");

    if let Ok(count) = lang_detect_word_count_only("Quick word count test here.") {
        println!("Word count only: {}", count);
    }

    if let Ok(count) = lang_detect_sentence_count_only("Sentence count test. Another one here.") {
        println!("Sentence count only: {}", count);
    }
}

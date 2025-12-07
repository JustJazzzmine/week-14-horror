#!/usr/bin/env python3
"""
Horror Literature Text Analysis
Performs comprehensive analysis of classic horror texts including:
- Content extraction
- Sentiment analysis (VADER)
- Lexical diversity and vocabulary richness
- Topic modeling
- Bag of words generation
"""

import re
import json
from collections import Counter
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Text files to analyze
TEXT_FILES = [
    "Carmilla.txt",
    "Dracula.txt",
    "Frankenstein.txt",
    "The Strange Case of Dr. Jekyll and Mr. Hyde.txt",
    "Turning of the Screw.txt"
]

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def extract_literary_content(self, file_path):
        """Extract only the literary content, skipping Project Gutenberg boilerplate"""
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        # Find the start marker
        start_pattern = r'\*\*\* START OF (?:THE |THIS )PROJECT GUTENBERG EBOOK.*?\*\*\*'
        start_match = re.search(start_pattern, content, re.IGNORECASE)

        if start_match:
            content = content[start_match.end():]

        # Find the end marker
        end_pattern = r'\*\*\* END OF (?:THE |THIS )PROJECT GUTENBERG EBOOK.*?\*\*\*'
        end_match = re.search(end_pattern, content, re.IGNORECASE)

        if end_match:
            content = content[:end_match.start()]

        return content.strip()

    def preprocess_text(self, text):
        """Tokenize and clean text"""
        # Convert to lowercase
        text = text.lower()

        # Tokenize into words
        tokens = word_tokenize(text)

        # Remove punctuation and non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]

        return tokens

    def get_filtered_tokens(self, tokens):
        """Remove stopwords from tokens"""
        return [word for word in tokens if word not in self.stop_words]

    def build_bag_of_words(self, tokens, top_n=100):
        """Create frequency distribution of words"""
        word_counts = Counter(tokens)
        return dict(word_counts.most_common(top_n))

    def calculate_lexical_diversity(self, tokens):
        """Calculate various lexical diversity metrics"""
        total_words = len(tokens)
        unique_words = len(set(tokens))

        # Type-Token Ratio
        ttr = unique_words / total_words if total_words > 0 else 0

        # Average word length
        avg_word_length = sum(len(word) for word in tokens) / total_words if total_words > 0 else 0

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": round(ttr, 4),
            "average_word_length": round(avg_word_length, 2),
            "lexical_density": round(ttr * 100, 2)  # As percentage
        }

    def analyze_sentiment(self, text):
        """Perform sentiment analysis using VADER"""
        # Split into sentences for granular analysis
        sentences = sent_tokenize(text)

        sentence_sentiments = []
        compound_scores = []

        for sent in sentences:
            scores = self.sentiment_analyzer.polarity_scores(sent)
            sentence_sentiments.append(scores)
            compound_scores.append(scores['compound'])

        # Calculate overall statistics
        overall = {
            "mean_compound": round(np.mean(compound_scores), 4),
            "median_compound": round(np.median(compound_scores), 4),
            "std_compound": round(np.std(compound_scores), 4),
            "positive_ratio": round(sum(1 for s in compound_scores if s > 0.05) / len(compound_scores), 4),
            "negative_ratio": round(sum(1 for s in compound_scores if s < -0.05) / len(compound_scores), 4),
            "neutral_ratio": round(sum(1 for s in compound_scores if -0.05 <= s <= 0.05) / len(compound_scores), 4),
            "total_sentences": len(sentences)
        }

        return overall

    def topic_modeling(self, text, n_topics=5, n_words=10):
        """Perform topic modeling using LDA"""
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )

        # Split text into chunks for topic modeling
        sentences = sent_tokenize(text)
        # Group sentences into larger chunks (every 20 sentences)
        chunk_size = 20
        chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

        if len(chunks) < n_topics:
            n_topics = max(2, len(chunks) // 2)

        try:
            doc_term_matrix = vectorizer.fit_transform(chunks)

            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            lda.fit(doc_term_matrix)

            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-n_words:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    "topic_id": topic_idx + 1,
                    "words": top_words,
                    "weights": [round(float(topic[i]), 4) for i in top_words_idx]
                })

            return topics
        except Exception as e:
            print(f"Topic modeling error: {e}")
            return []

    def analyze_text_file(self, file_path):
        """Comprehensive analysis of a single text file"""
        print(f"Analyzing {file_path}...")

        # Extract literary content
        raw_content = self.extract_literary_content(file_path)

        # Preprocess
        all_tokens = self.preprocess_text(raw_content)
        filtered_tokens = self.get_filtered_tokens(all_tokens)

        # Build bag of words
        bow = self.build_bag_of_words(filtered_tokens, top_n=150)

        # Calculate lexical diversity
        lexical_metrics = self.calculate_lexical_diversity(filtered_tokens)

        # Sentiment analysis
        sentiment = self.analyze_sentiment(raw_content)

        # Topic modeling
        topics = self.topic_modeling(raw_content)

        # Get title from filename
        title = Path(file_path).stem

        return {
            "title": title,
            "bag_of_words": bow,
            "lexical_diversity": lexical_metrics,
            "sentiment": sentiment,
            "topics": topics
        }

def main():
    """Main execution function"""
    analyzer = TextAnalyzer()
    results = {}

    print("Starting text analysis...\n")

    for text_file in TEXT_FILES:
        file_path = Path(__file__).parent.parent / text_file

        if not file_path.exists():
            print(f"Warning: {text_file} not found, skipping...")
            continue

        analysis = analyzer.analyze_text_file(str(file_path))
        results[analysis['title']] = analysis

    # Save results to JSON
    output_path = Path(__file__).parent.parent / 'data' / 'analysis_results.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Analysis complete! Results saved to {output_path}")

    # Print summary
    print("\nSummary:")
    for title, data in results.items():
        print(f"\n{title}:")
        print(f"  - Total words: {data['lexical_diversity']['total_words']:,}")
        print(f"  - Unique words: {data['lexical_diversity']['unique_words']:,}")
        print(f"  - Lexical density: {data['lexical_diversity']['lexical_density']}%")
        print(f"  - Mean sentiment: {data['sentiment']['mean_compound']}")
        print(f"  - Topics identified: {len(data['topics'])}")

if __name__ == "__main__":
    main()

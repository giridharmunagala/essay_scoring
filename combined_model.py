# Combine the statistics based model & the word & sentence embedding model.
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from utils import ExtractStatisticsBasedFeatures, TextEmbeddingDs, DataCollatorForSequences
import spacy



def prepare_data(texts, labels, word_embed_model_name, sent_embed_model_name):
    # Prepare embeddings
    tokenizer = AutoTokenizer.from_pretrained(word_embed_model_name)
    word_embed_model = AutoModel.from_pretrained(word_embed_model_name)
    sent_embed_model = SentenceTransformer(sent_embed_model_name)
    
    embedding_dataset = TextEmbeddingDs(texts, labels, tokenizer, word_embed_model, sent_embed_model)
    
    # Prepare statistics features
    nlp = spacy.load("en_core_web_sm")
    statistics_extractor = ExtractStatisticsBasedFeatures(texts, nlp)
    statistics_features = statistics_extractor.get_features()
    
    # Change spacy model to en_core_web_lg
    nlp = spacy.load("en_core_web_lg")
    statistics_extractor = ExtractStatisticsBasedFeatures(texts, nlp)
    statistics_features = statistics_extractor.get_features()

    # Generate additional features from statistics_based_model.ipynb
    additional_features = []
    for doc in nlp.pipe(texts, batch_size=100, n_process=10):
        features = {
            "average_words_per_sentence": statistics_extractor.get_average_words_per_sentence(doc),
            "percent_passive_sentences": statistics_extractor.get_percent_passive_sentences(doc),
            "percent_simple_sentences": statistics_extractor.get_percent_simple_sentences(doc),
            "percent_compound_sentences": statistics_extractor.get_percent_compound_sentences(doc),
            "percent_complex_sentences": statistics_extractor.get_percent_complex_sentences(doc)
        }
        additional_features.append(features)

    # Combine all statistics features
    combined_statistics = []
    for base_stats, add_stats in zip(statistics_features, additional_features):
        combined_stats = {**base_stats, **add_stats}
        combined_statistics.append(combined_stats)

    return embedding_dataset, combined_statistics

# Usage example:
# model = CombinedModel("bert-base-uncased", "all-MiniLM-L6-v2", num_statistics_features=4)
# embedding_dataset, statistics_features = prepare_data(texts, labels, "bert-base-uncased", "all-MiniLM-L6-v2")
# data_collator = DataCollatorForSequences()
# ... (continue with DataLoader, training loop, etc.)

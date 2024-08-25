# Author: Munagala Giridhar
# Utils used in the process of Essay scoring.
# Majorly towards data modelling
from typing import List, Dict
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import evaluate
import numpy as np
import spacy
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import multiprocessing


# Loading Accuracy Eval metric
accuracy = evaluate.load("accuracy")

# Loading Spacy model for sentence extraction only
nlp = spacy.load("en_core_web_sm", disable=["ner"])


def compute_accuracy_metric(eval_pred):
    """
    Computes Accuracy based on the Labels & Predictions
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def compute_quadratic_weighted_kappa(predictions, labels):
    """
    Computes Quadratic Weighted Kappa based on the Labels & Predictions
    """
    return cohen_kappa_score(predictions, labels, weights="quadratic")


def preprocess_text(text: str) -> str:
    """
    Removes special characters & lower cases text.
    """
    if text:
        return re.sub("[^a-z0-9,.\s]", "", text.lower())
    return ""


class TextClassifDs(Dataset):
    """
    Text dataset for Classification.
    Outputs tokenized inforation.
    Args:
        text: List of texts
        labels: List of labels
        tokenizer: Transformers tokenizer that follows Autotokenizer structures
        transformations: TBD
    returns:
        Dictionary with keys
            input_ids : Tokenized ids
            attention_mask: attention mask of the text
            label: Labels provided
    """

    def __init__(
        self, text: List, labels: List, tokenizer: AutoTokenizer, transformation=None
    ) -> Dict:
        super().__init__()
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index) -> Dict:
        """
        Tokenizes with the given tokenizer & returns the data in a dictionary with the labels.
        """
        tokenized_text = self.tokenizer(
            self.text[index],
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        return {
            "input_ids": tokenized_text["input_ids"].squeeze(),
            "attention_mask": tokenized_text["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }

class ExtractStatisticsBasedFeatures:
    """
    Extracts statistics based features from the text.
    """
    def __init__(self, docs: List) -> Dict:
        super().__init__()
        self.doc_df = pd.DataFrame([{'doc' : doc}  for doc in docs])

    def _get_number_of_sentences(self, doc):
        return len(list(doc.sents))

    def _get_number_of_unique_pos_tags(self, doc):
        return len(set([token.pos_ for token in doc]))

    def _get_number_of_unique_dep_tags(self, doc):
        return len(set([token.dep_ for token in doc]))

    def _get_number_of_unique_ents(self, doc):
        return len(set([ent.label_ for ent in doc.ents]))

    def _get_average_words_per_sentence(self, doc):
        return len(doc) / self._get_number_of_sentences(doc)

    def _get_percent_passive_sentences(self, doc):
        passive_sentences = [
            sent for sent in doc.sents if any(tok.dep_ == "nsubjpass" for tok in sent)
        ]
        return len(passive_sentences) / len(list(doc.sents)) * 100

    def _get_percent_simple_sentences(self, doc):
        simple_sentences = [sent for sent in doc.sents if len(list(sent.subtree)) <= 10]
        return len(simple_sentences) / len(list(doc.sents)) * 100

    def _get_percent_compound_sentences(self, doc):
        compound_sentences = [
            sent
            for sent in doc.sents
            if len(list(sent.subtree)) > 10 and len(list(sent.subtree)) <= 20
        ]
        return len(compound_sentences) / len(list(doc.sents)) * 100

    def _get_percent_complex_sentences(self, doc):
        complex_sentences = [sent for sent in doc.sents if len(list(sent.subtree)) > 20]
        return len(complex_sentences) / len(list(doc.sents)) * 100

    def _apply_statistics_features(self, df):
       
        return df

    def extract_stats_features(self) -> Dict:
        """
        Processes the text & returns the statistics based features.
        """
        print("Extracting number_of_sentences")
        self.doc_df["number_of_sentences"] = self.doc_df["doc"].progress_apply(self._get_number_of_sentences)
        print("Extracting number_of_unique_pos_tags")
        self.doc_df["number_of_unique_pos_tags"] = self.doc_df["doc"].progress_apply(
            self._get_number_of_unique_pos_tags
        )
        print("Extracting number_of_unique_dep_tags")
        self.doc_df["number_of_unique_dep_tags"] = self.doc_df["doc"].progress_apply(
            self._get_number_of_unique_dep_tags
        )
        print("Extracting number_of_unique_ents")
        self.doc_df["number_of_unique_ents"] = self.doc_df["doc"].progress_apply(self._get_number_of_unique_ents)
        print("Extracting average_words_per_sentence")
        self.doc_df["average_words_per_sentence"] = self.doc_df["doc"].progress_apply(
            self._get_average_words_per_sentence
        )
        print("Extracting percent_passive_sentences")
        self.doc_df["percent_passive_sentences"] = self.doc_df["doc"].progress_apply(
            self._get_percent_passive_sentences
        )
        print("Extracting percent_simple_sentences")
        self.doc_df["percent_simple_sentences"] = self.doc_df["doc"].progress_apply(
            self._get_percent_simple_sentences
        )
        print("Extracting percent_compound_sentences")
        self.doc_df["percent_compound_sentences"] = self.doc_df["doc"].progress_apply(
            self._get_percent_compound_sentences
        )
        print("Extracting percent_complex_sentences")
        self.doc_df["percent_complex_sentences"] = self.doc_df["doc"].progress_apply(
            self._get_percent_complex_sentences
        )
        return self.doc_df.drop(columns=['doc']).apply(np.array, axis=1).values.tolist()


class MultiInputTextDs(Dataset):
    """
    Text dataset for classificaiton, that provides word embeddings as well as sentence embeddings for each text.
    Converts the text to sentences using Spacy sentencizer
    Outputs token embeddings & sentence embeddings.
    Args:
        text: List of texts
        labels: List of labels
        word_tokenizer: Transformers tokenizer that follows Autotokenizer structures
        word_embed_model: Any Bert Transformer model
        sent_embed_model: Sentence Transformer model
        transformations: TBD
    returns:
        Dictionary with keys
            word_embeddings: Word embeddings of the text
            sentence_embeddings: Sentence embeddings of the text
            label: Labels provided
    """

    def __init__(
        self,
        text: List,
        labels: List,
        word_tokenizer: AutoTokenizer,
        word_embed_model: AutoModel,
        sent_embed_model: SentenceTransformer,
        spacy_model: spacy.language.Language,
        add_stats_features: bool = False,
        device: str = "cuda",
        transformation=None,
    ) -> Dict:
        super().__init__()
        self.text = text
        print("Extracting Sentences")
        self.spacy_model = spacy_model
        self.docs = self._extract_docs()
        self.add_stats_features = add_stats_features
        if self.add_stats_features:
            self.stats_features = self._get_stats_features()
        self.device = device
        self.word_embed_model = word_embed_model.eval().to(device)
        self.word_tokenizer = word_tokenizer
        self.sent_embed_model = sent_embed_model.eval().to(device)
        self.labels = labels
        self.transformations = transformation

    def _get_stats_features(self) -> List:
        """
        Extracts statistics based features from the text.
        """
        statistics_extractor = ExtractStatisticsBasedFeatures(self.docs)
        return statistics_extractor.extract_stats_features()

    def _extract_docs(self) -> List:
        """
        Extracts sentences from the given text.
        Use multiprocessing to speed up the Spacy doc creation
        """
        docs = []
        # Adding batching as higher number of text causing memory issues
        # & slowing down multiprocessing throughput
        for doc in tqdm(
            self.spacy_model.pipe(self.text, batch_size=100, n_process=10),
            total=len(self.text),
        ):
            docs.append(doc)
        return docs

    def _get_word_embeddings(self, text: str) -> torch.Tensor:
        """
        Returns the word embeddings of the given text.
        """
        tokenized_text = self.word_tokenizer(
            text,
            return_tensors="pt",
            max_length=self.word_tokenizer.model_max_length,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.word_embed_model(**tokenized_text.to(self.device))
            return outputs.last_hidden_state

    def __len__(self):
        return len(self.text)

    def _add_batch_dimension(self, torch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Adds batch dimension to the tensor if it is of batch size 1
        """
        if torch_tensor.dim() == 1:
            torch_tensor = torch_tensor.unsqueeze(0)
        return torch_tensor

    def __getitem__(self, index) -> Dict:
        """
        Generates word embeddings & sentence embeddings of the text.
        """
        with torch.no_grad():
            word_embeddings = self._get_word_embeddings(
                preprocess_text(self.text[index])
            ).squeeze()
            sentence_embeddings = self.sent_embed_model.encode(
                [sent.text for sent in self.docs[index].sents], convert_to_tensor=True
            ).squeeze()
        word_embeddings = self._add_batch_dimension(word_embeddings)
        sentence_embeddings = self._add_batch_dimension(sentence_embeddings)
        stats_features = self.stats_features[index] if self.add_stats_features  else None
        return {
            "word_embeddings": word_embeddings.to("cuda"),
            "sentence_embeddings": sentence_embeddings.to("cuda"),
            "label": torch.tensor(self.labels[index], dtype=torch.float).to("cuda"),
            "stats_features": torch.from_numpy(stats_features).float().to("cuda")
        }


class DataCollatorForSequences:
    """
    Data collator for the multi-input text dataset.
    """
    def __init__(self,add_stats_feat=False) -> None:
        self.add_stats_feat = add_stats_feat


    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract word embeddings, sentence embeddings, and labels from the batch
        word_embeddings = [item["word_embeddings"] for item in batch]
        sentence_embeddings = [item["sentence_embeddings"] for item in batch]
        labels = [item["label"] for item in batch]
        stats_features = None

        if self.add_stats_feat:
            stats_features = torch.stack([item["stats_features"] for item in batch])
        # Pad the sequences
        padded_word_embeddings = pad_sequence(
            word_embeddings, batch_first=True, padding_value=0.0
        )
        padded_sentence_embeddings = pad_sequence(
            sentence_embeddings, batch_first=True, padding_value=0.0
        )

        # Stack the labels into a tensor
        labels = torch.stack(labels)

        return {
            "word_embeddings": padded_word_embeddings,
            "sentence_embeddings": padded_sentence_embeddings,
            "labels": labels,
            "stats_features": stats_features
        }


def apply_parallel(df, func):
    """
    Applies the function in parallel to the dataframe.
    Add progress bar to the parallel processing.
    """
    num_cores = multiprocessing.cpu_count()
    df_split = np.array_split(df, num_cores)
    with multiprocessing.Pool(num_cores) as pool:
        results = []
        for result in tqdm(pool.imap(func, df_split), total=num_cores):
            results.append(result)
    df = pd.concat(results)
    return df

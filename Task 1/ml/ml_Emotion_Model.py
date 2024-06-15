import ml_imports as ml


class EmotionalAnalysisCommentsTrainer:
    def __init__(self, path_to_df: str) -> None:
        self._df = ml.pd.read_csv(path_to_df, sep=',')
        self._vectorizer = ml.TfidfVectorizer()
        self._model = ml.MultiOutputClassifier(ml.SGDClassifier(random_state=0, max_iter=1000, n_jobs=-1))
        self._snowball = ml.SnowballStemmer(language="russian")
        self._ru_stop_words = ml.stopwords.words("russian")

    def _del_punctuation(self, tokens: ml.List[str]) -> ml.List[str]:
        return [token for token in tokens if token not in ml.string.punctuation]

    def _del_stop_words(self, tokens: ml.List[str], stop_words: ml.List[str]) -> ml.List[str]:
        return [token for token in tokens if token not in stop_words]

    def _tokenize_text(self, text: str) -> str:
        tokens = ml.word_tokenize(text, language="russian")
        tokens = self._del_punctuation(tokens)
        tokens = self._del_stop_words(tokens, self._ru_stop_words)
        stemmed_tokens = [self._snowball.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def _tokenize_df(self) -> None:
        self._df['tokenized_comments'] = ml.Parallel(n_jobs=-1)(
            ml.delayed(self._tokenize_text)(text) for text in self._df['text']
        )

    def train_and_save_model(self, model_path: str, vectorizer_path: str) -> None:
        self._tokenize_df()
        target_labels = ['happy', 'neutral', 'sad', 'aggressive']
        emotions = self._df[target_labels].values

        features = self._vectorizer.fit_transform(self._df['tokenized_comments'])
        self._model.fit(features, emotions)

        ml.dump(self._model, model_path)
        ml.dump(self._vectorizer, vectorizer_path)


def main():
    trainer = EmotionalAnalysisCommentsTrainer('../dataset/dataset.csv')
    trainer.train_and_save_model('trainer/emotion_model.joblib', 'trainer/vectorizer.joblib')


if __name__ == "__main__":
    main()
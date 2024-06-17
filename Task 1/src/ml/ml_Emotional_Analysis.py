from . import ml_imports as ml

class EmotionalAnalysisComments:
    def __init__(self, model_path: str, vectorizer_path: str, comments: ml.List[str]) -> None:
        self._model_path = model_path
        self._vectorizer_path = vectorizer_path
        self._user_comments = comments
        self._snowball = ml.SnowballStemmer(language="russian")
        self._ru_stop_words = ml.stopwords.words("russian")
        self._user_tokenize_comments = []
        self._model = None
        self._vectorizer = None

    def _load_model_and_vectorizer(self) -> None:
        if not self._model:
            self._model = ml.load(self._model_path)
        if not self._vectorizer:
            self._vectorizer = ml.load(self._vectorizer_path)

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

    def _tokenize_user_comments(self) -> None:
        self._user_tokenize_comments = [
            self._tokenize_text(comment) for comment in self._user_comments
        ]

    def _print_emotion_comments(self, predicted_emotions: ml.np.array, labels: ml.List[str]) -> None:
        for i, comment in enumerate(self._user_comments):
            print(f"Comment: {comment}")
            predicted_emotion_dict = {labels[j]: predicted_emotions[i][j] for j in range(len(labels)) if predicted_emotions[i][j] == 1}
            print("Predicted emotions:", predicted_emotion_dict, '\n')

    def analysis_comments(self) -> None:
        self._load_model_and_vectorizer()
        self._tokenize_user_comments()

        user_comment_vector = self._vectorizer.transform(self._user_tokenize_comments)
        predicted_emotions = self._model.predict(user_comment_vector)

        target_labels = ['happy', 'neutral', 'sad', 'aggressive']
        self._print_emotion_comments(predicted_emotions, target_labels)
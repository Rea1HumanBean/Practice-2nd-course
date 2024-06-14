import string
import ml_imports as ml


class EmotionalAnalysisComments:
    def __init__(self, path_to_df: str, comments: list[str]) -> None:
        self._df = ml.pd.read_csv(path_to_df, sep=',')
        self._user_comments = comments
        self._user_tokenize_comments = []

    def _del_punctuation(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in string.punctuation]

    def _del_stop_words(self, tokens: list[str], stop_words: list[str]) -> list[str]:
        return [token for token in tokens if token not in stop_words]

    def _tokenize_df(self) -> None:
        self._df['tokenized_comments'] = self.__tokenize_and_stem(self._df['text'])

    def _tokenize_user_comments(self) -> None:
        self._user_tokenize_comments = self.__tokenize_and_stem(self._user_comments)

    def __tokenize_and_stem(self, texts: list[str]) -> list[str]:
        snowball = ml.SnowballStemmer(language="russian")
        ru_stop_words = ml.stopwords.words("russian")

        tokenized_texts = []
        for text in texts:
            tokens = ml.word_tokenize(text, language="russian")
            tokens = self._del_punctuation(tokens)
            tokens = self._del_stop_words(tokens, ru_stop_words)

            stemmed_tokens = [snowball.stem(token) for token in tokens]
            tokenized_texts.append(' '.join(stemmed_tokens))

        return tokenized_texts

    def create_and_predict_model(self, emotions: list[int]) -> ml.np.array:
        vectorizer = ml.TfidfVectorizer()
        features = vectorizer.fit_transform(self._df['tokenized_comments'])

        model = ml.MultiOutputClassifier(ml.LogisticRegression(random_state=0, max_iter=1000))
        model.fit(features, emotions)

        user_comment_vector = vectorizer.transform(self._user_tokenize_comments)
        predicted_emotions = model.predict(user_comment_vector)

        return predicted_emotions

    def _print_emotion_comments(self, predicted_emotions: ml.np.array, labels: list) -> None:
        for i, comment in enumerate(self._user_comments):
            print(f"Comment: {comment}")
            print("Predicted emotions:",
                  {labels[j]: predicted_emotions[i][j] for j in range(len(labels))})

    def analysis_comments(self) -> None:
        self._tokenize_df()
        self._tokenize_user_comments()

        target_labels = ['happy', 'neutral', 'sad', 'aggressive']
        emotions = self._df[target_labels].values

        predicted_emotions = self.create_and_predict_model(emotions)

        self._print_emotion_comments(predicted_emotions, target_labels)


def main():
    comments = [
        "Ты классная!",
        "Я не уверен в этом...",
        "Это ненавижу это!",
        "Я расстроен результатом."
    ]

    EmotionMessage = EmotionalAnalysisComments('../dataset/dataset.csv', comments)
    EmotionMessage.analysis_comments()


if __name__ == "__main__":
    main()

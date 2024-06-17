import os
import sys
from src.VKFetcher.VKCommentsFetcher import VKCommentsFetcher
from src.ml.ml_Emotional_Analysis import EmotionalAnalysisComments


def main():
    current_dir = os.path.dirname(sys.executable)
    credentials_path = os.path.abspath(os.path.join(current_dir, '..', 'credentials', 'key'))

    with open(credentials_path, 'r') as file:
        service_token = file.read().strip()

    user_id = int(input("Enter the user ID: "))

    fetcher = VKCommentsFetcher(service_token, user_id)
    comments = fetcher.get_user_comments()

    model_path = os.path.abspath(os.path.join(current_dir, '..', 'src', 'ml', 'trainer', 'emotion_model.joblib'))
    vectorizer_path = os.path.abspath(os.path.join(current_dir, '..', 'src', 'ml', 'trainer', 'vectorizer.joblib'))

    EmotionMessage = EmotionalAnalysisComments(model_path, vectorizer_path, comments)
    EmotionMessage.analysis_comments()

    input("Press any button to exit...")


if __name__ == "__main__":
    main()

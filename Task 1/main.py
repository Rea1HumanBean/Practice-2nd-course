from typing import List
import vk_api


class VKCommentsFetcher:
    def __init__(self, service_token: str, user_id: int) -> None:
        self._service_token = service_token
        self._user_id = user_id
        self._vk = vk_api.VkApi(token=self._service_token).get_api()
        self._user_comments = []

    def _get_mentions(self) -> dict:
        return self._vk.newsfeed.getMentions(owner_id=self._user_id, access_token=self._service_token)

    def _get_comments_response(self, community_id: int, post_id: int) -> List[dict]:
        return self._vk.wall.getComments(owner_id=community_id, post_id=post_id)


    def _generate_user_comments(self) -> None:
        mention = self._get_mentions()
        for post in mention['items']:
            comments_response = self._get_comments_response(post['to_id'], post['id'])
            comments = [comment['text'] for comment in comments_response.get('items', []) if comment['from_id'] == self._user_id]
            self._user_comments.extend(comments)

    def get_user_comments(self) -> List[str]:
        try:
            self._generate_user_comments()

        except vk_api.exceptions.ApiError as error:
            print(f"Error: {error}")

        return self._user_comments


def main():
    with open('credentials/key', 'r') as file:
        service_token = file.read()

    user_id = int(input("Enter the user ID: "))

    fetcher = VKCommentsFetcher(service_token, user_id)
    comments = fetcher.get_user_comments()

    for comment in comments:
        print(comment)


if __name__ == "__main__":
    main()

from typing import TypedDict


class MentionInput(TypedDict):
    mention: str
    context_left: str
    context_right: str


class EntityInput(TypedDict):
    text: str
    title: str
from typing import Optional, TypedDict


class MentionInput(TypedDict):
    mention: str
    context_left: str
    context_right: str


class EntityInput(TypedDict):
    text: str
    title: str


class ProcessInput(MentionInput, EntityInput):
    label_id: int
    world: Optional[str]
import json

from openai_harmony import Message, Role


def test_message_model_dump_json_preserves_text_content():
    message = Message.from_role_and_content(Role.ASSISTANT, "hello")

    payload = json.loads(message.model_dump_json())

    assert payload["content"] == [{"text": "hello"}]

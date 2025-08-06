"""Python wrapper around the Rust implementation of *harmony*.

The heavy lifting (tokenisation, rendering, parsing, …) is implemented in
Rust.  The thin bindings are available through the private ``openai_harmony``
extension module which is compiled via *maturin* / *PyO3*.

This package provides a small, typed convenience layer that mirrors the public
API of the Rust crate so that it can be used from Python code in an
idiomatic way (``dataclasses``, ``Enum``s, …).
"""

from __future__ import annotations

import functools
import json
from enum import Enum
from typing import (
    AbstractSet,
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Pattern,
    Sequence,
    TypeVar,
    Union,
)

import re
from pydantic import BaseModel, Field

# Re-export the low-level Rust bindings under a private name so that we can
# keep the *public* namespace clean and purely Pythonic.
try:
    from .openai_harmony import (
        HarmonyError as HarmonyError,  # expose the actual Rust error directly
        PyHarmonyEncoding as _PyHarmonyEncoding,  # type: ignore
        PyStreamableParser as _PyStreamableParser,  # type: ignore
        load_harmony_encoding as _load_harmony_encoding,  # type: ignore
        load_harmony_encoding_from_file as _load_harmony_encoding_from_file,  # type: ignore
    )

except ModuleNotFoundError:  # pragma: no cover – raised during type-checking
    # When running *mypy* without the compiled extension in place we still want
    # to succeed.  Therefore we create dummy stubs that satisfy the type
    # checker.  They will, however, raise at **runtime** if accessed.

    class _Stub:  # pylint: disable=too-few-public-methods
        def __getattr__(self, name: str) -> None:  # noqa: D401
            raise RuntimeError(
                "The compiled harmony bindings are not available. Make sure to "
                "build the project with `maturin develop` before running this "
                "code."
            )

    _load_harmony_encoding = _Stub()  # type: ignore
    _PyHarmonyEncoding = _Stub()  # type: ignore
    _PyStreamableParser = _Stub()  # type: ignore
    _HarmonyError = RuntimeError


def _special_token_regex(tokens: frozenset[str]) -> Pattern[str]:
    inner = "|".join(re.escape(token) for token in tokens)
    return re.compile(f"({inner})")


def raise_disallowed_special_token(token: str) -> None:
    raise HarmonyError(
        "Encountered text corresponding to disallowed special token "
        f"{token!r}.\n"
        "If you want this text to be encoded as a special token, "
        f"pass it to `allowed_special`, e.g. `allowed_special={{'{token}', ...}}`.\n"
        "If you want this text to be encoded as normal text, disable the check for this token "
        f"by passing `disallowed_special=(enc.special_tokens_set - {{'{token}'}})`.\n"
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
    )


# ---------------------------------------------------------------------------
# Chat-related data-structures (mirroring ``src/chat.rs``)
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """The role of a message author (mirrors ``chat::Role``)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"

    @classmethod
    def _missing_(cls, value: object) -> "Role":  # type: ignore[override]
        raise ValueError(f"Unknown role: {value!r}")


class Author(BaseModel):
    role: Role
    name: Optional[str] = None

    @classmethod
    def new(cls, role: Role, name: str) -> "Author":  # noqa: D401 – keep parity with Rust API
        return cls(role=role, name=name)


# Content hierarchy ---------------------------------------------------------


T = TypeVar("T")


class Content(BaseModel):  # noqa: D101 – simple wrapper
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class TextContent(Content):
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "text", "text": self.text}


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    @classmethod
    def new(
        cls, name: str, description: str, parameters: Optional[dict] = None
    ) -> "ToolDescription":  # noqa: D401
        return cls(name=name, description=description, parameters=parameters)


class ReasoningEffort(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ChannelConfig(BaseModel):
    valid_channels: List[str]
    channel_required: bool

    @classmethod
    def require_channels(cls, channels: List[str]) -> "ChannelConfig":  # noqa: D401
        return cls(valid_channels=channels, channel_required=True)


class ToolNamespaceConfig(BaseModel):
    name: str
    description: Optional[str] = None
    tools: List[ToolDescription]

    @staticmethod
    def browser() -> "ToolNamespaceConfig":
        from .openai_harmony import (
            get_tool_namespace_config as _get_tool_namespace_config,
        )

        cfg = _get_tool_namespace_config("browser")
        return ToolNamespaceConfig(**cfg)

    @staticmethod
    def python() -> "ToolNamespaceConfig":
        from .openai_harmony import (
            get_tool_namespace_config as _get_tool_namespace_config,
        )

        cfg = _get_tool_namespace_config("python")
        return ToolNamespaceConfig(**cfg)


class SystemContent(Content):
    model_identity: Optional[str] = (
        "You are ChatGPT, a large language model trained by OpenAI."
    )
    reasoning_effort: Optional[ReasoningEffort] = ReasoningEffort.MEDIUM
    conversation_start_date: Optional[str] = None
    knowledge_cutoff: Optional[str] = "2024-06"
    channel_config: Optional[ChannelConfig] = Field(
        default_factory=lambda: ChannelConfig.require_channels(
            ["analysis", "commentary", "final"]
        )
    )
    tools: Optional[dict[str, ToolNamespaceConfig]] = None

    @classmethod
    def new(cls) -> "SystemContent":
        return cls()

    # Fluent interface ------------------------------------------------------

    def with_model_identity(self, model_identity: str) -> "SystemContent":
        self.model_identity = model_identity
        return self

    def with_reasoning_effort(
        self, reasoning_effort: ReasoningEffort
    ) -> "SystemContent":
        self.reasoning_effort = reasoning_effort
        return self

    def with_conversation_start_date(
        self, conversation_start_date: str
    ) -> "SystemContent":
        self.conversation_start_date = conversation_start_date
        return self

    def with_knowledge_cutoff(self, knowledge_cutoff: str) -> "SystemContent":
        self.knowledge_cutoff = knowledge_cutoff
        return self

    def with_channel_config(self, channel_config: ChannelConfig) -> "SystemContent":
        self.channel_config = channel_config
        return self

    def with_required_channels(self, channels: list[str]) -> "SystemContent":
        self.channel_config = ChannelConfig.require_channels(channels)
        return self

    def with_tools(self, ns_config: ToolNamespaceConfig) -> "SystemContent":
        if self.tools is None:
            self.tools = {}
        self.tools[ns_config.name] = ns_config
        return self

    def with_browser_tool(self) -> "SystemContent":
        return self.with_tools(ToolNamespaceConfig.browser())

    def with_python_tool(self) -> "SystemContent":
        return self.with_tools(ToolNamespaceConfig.python())

    def to_dict(self) -> dict:
        out = self.model_dump(exclude_none=True)
        out["type"] = "system_content"
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> "SystemContent":
        return cls(**raw)


class DeveloperContent(Content):
    instructions: Optional[str] = None
    tools: Optional[dict[str, ToolNamespaceConfig]] = None

    @classmethod
    def new(cls) -> "DeveloperContent":
        return cls()

    def with_instructions(self, instructions: str) -> "DeveloperContent":
        self.instructions = instructions
        return self

    def with_tools(self, ns_config: ToolNamespaceConfig) -> "DeveloperContent":
        if self.tools is None:
            self.tools = {}
        self.tools[ns_config.name] = ns_config
        return self

    def with_function_tools(
        self, tools: Sequence[ToolDescription]
    ) -> "DeveloperContent":
        return self.with_tools(
            ToolNamespaceConfig(name="functions", description=None, tools=list(tools))
        )

    def to_dict(self) -> dict:
        out = self.model_dump(exclude_none=True)
        out["type"] = "developer_content"
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> "DeveloperContent":
        return cls(**raw)


# Message & Conversation -----------------------------------------------------


class Message(BaseModel):
    author: Author
    content: List[Content] = Field(default_factory=list)
    channel: Optional[str] = None
    recipient: Optional[str] = None
    content_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience constructors (mirroring the Rust API)
    # ------------------------------------------------------------------

    @classmethod
    def from_author_and_content(
        cls, author: Author, content: Union[str, Content]
    ) -> "Message":
        if isinstance(content, str):
            content = TextContent(text=content)
        return cls(author=author, content=[content])

    @classmethod
    def from_role_and_content(
        cls, role: Role, content: Union[str, Content]
    ) -> "Message":  # noqa: D401 – parity with Rust API
        return cls.from_author_and_content(Author(role=role), content)

    @classmethod
    def from_role_and_contents(
        cls, role: Role, contents: Sequence[Content]
    ) -> "Message":
        return cls(author=Author(role=role), content=list(contents))

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def adding_content(self, content: Union[str, Content]) -> "Message":
        if isinstance(content, str):
            content = TextContent(text=content)
        self.content.append(content)
        return self

    def with_channel(self, channel: str) -> "Message":
        self.channel = channel
        return self

    def with_recipient(self, recipient: str) -> "Message":
        self.recipient = recipient
        return self

    def with_content_type(self, content_type: str) -> "Message":
        self.content_type = content_type
        return self

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:  # noqa: D401 – simple mapper
        out: Dict[str, Any] = {
            **self.author.model_dump(),
            "content": [c.to_dict() for c in self.content],
        }
        if self.channel is not None:
            out["channel"] = self.channel
        if self.recipient is not None:
            out["recipient"] = self.recipient
        if self.content_type is not None:
            out["content_type"] = self.content_type
        return out

    def to_json(self) -> str:  # noqa: D401
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        # Simple, sufficient implementation for test-roundtrip purposes.
        role = Role(data["role"])
        author = Author(role=role, name=data.get("name"))

        contents: List[Content] = []

        raw_content = data["content"]

        # The Rust side serialises *single* text contents as a **plain string**
        # for convenience.  Detect this shortcut and normalise it to the list
        # representation that the rest of the Python code expects.
        if isinstance(raw_content, str):
            raw_content = [{"type": "text", "text": raw_content}]

        for raw in raw_content:
            if raw.get("type") == "text":
                contents.append(TextContent(**raw))
            elif raw.get("type") == "system_content":
                contents.append(SystemContent(**raw))
            elif raw.get("type") == "developer_content":
                contents.append(DeveloperContent(**raw))
            else:  # pragma: no cover – unknown variant
                raise ValueError(f"Unknown content variant: {raw}")

        msg = cls(author=author, content=contents)
        msg.channel = data.get("channel")
        msg.recipient = data.get("recipient")
        msg.content_type = data.get("content_type")
        return msg


class Conversation(BaseModel):
    messages: List[Message] = Field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: Sequence[Message]) -> "Conversation":  # noqa: D401
        return cls(messages=list(messages))

    def __iter__(self):
        return iter(self.messages)

    # Serialisation helpers -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:  # noqa: D401
        return {"messages": [m.to_dict() for m in self.messages]}

    def to_json(self) -> str:  # noqa: D401
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "Conversation":  # noqa: D401
        data = json.loads(payload)
        return cls(messages=[Message.from_dict(m) for m in data["messages"]])


# ---------------------------------------------------------------------------
# Encoding interaction (thin wrappers around the Rust bindings)
# ---------------------------------------------------------------------------


class RenderConversationConfig(BaseModel):
    auto_drop_analysis: bool = True


class RenderOptions(BaseModel):
    conversation_has_function_tools: bool = False


class HarmonyEncoding:
    """High-level wrapper around the Rust ``PyHarmonyEncoding`` class."""

    def __init__(self, inner: _PyHarmonyEncoding):
        self._inner = inner

    # ------------------------------------------------------------------
    # Delegated helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        return self._inner.name  # type: ignore[attr-defined]

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        return set(self._inner.special_tokens())

    # -- Rendering -----------------------------------------------------

    def render_conversation_for_completion(
        self,
        conversation: Conversation,
        next_turn_role: Role,
        config: Optional[RenderConversationConfig] = None,
    ) -> List[int]:
        """
        Render a conversation for completion.
        Args:
            conversation: Conversation object
            next_turn_role: Role for the next turn
            config: Optional RenderConversationConfig (default auto_drop_analysis=True)
        """
        if config is None:
            config_dict = {"auto_drop_analysis": True}
        else:
            config_dict = {"auto_drop_analysis": config.auto_drop_analysis}
        return self._inner.render_conversation_for_completion(
            conversation_json=conversation.to_json(),
            next_turn_role=str(next_turn_role.value),
            config=config_dict,
        )

    def render_conversation(
        self,
        conversation: Conversation,
        config: Optional[RenderConversationConfig] = None,
    ) -> List[int]:
        """Render a conversation without appending a new role."""
        if config is None:
            config_dict = {"auto_drop_analysis": True}
        else:
            config_dict = {"auto_drop_analysis": config.auto_drop_analysis}
        return self._inner.render_conversation(
            conversation_json=conversation.to_json(),
            config=config_dict,
        )

    def render_conversation_for_training(
        self,
        conversation: Conversation,
        config: Optional[RenderConversationConfig] = None,
    ) -> List[int]:
        """Render a conversation for training."""
        if config is None:
            config_dict = {"auto_drop_analysis": True}
        else:
            config_dict = {"auto_drop_analysis": config.auto_drop_analysis}
        return self._inner.render_conversation_for_training(
            conversation_json=conversation.to_json(),
            config=config_dict,
        )

    def render(
        self, message: Message, render_options: Optional[RenderOptions] = None
    ) -> List[int]:
        """Render a single message into tokens."""
        if render_options is None:
            render_options_dict = {"conversation_has_function_tools": False}
        else:
            render_options_dict = {
                "conversation_has_function_tools": render_options.conversation_has_function_tools
            }

        return self._inner.render(
            message_json=message.to_json(), render_options=render_options_dict
        )

    # -- Parsing -------------------------------------------------------

    def parse_messages_from_completion_tokens(
        self, tokens: Sequence[int], role: Optional[Role] | None = None
    ) -> List[Message]:
        raw_json: str = self._inner.parse_messages_from_completion_tokens(
            list(tokens), None if role is None else str(role.value)
        )
        return [Message.from_dict(m) for m in json.loads(raw_json)]

    # -- Token decoding ------------------------------------------------

    def decode_utf8(self, tokens: Sequence[int]) -> str:
        """Decode a list of tokens into a UTF-8 string. Will raise an error if the tokens result in invalid UTF-8. Use decode if you want to replace invalid UTF-8 with the unicode replacement character."""
        return self._inner.decode_utf8(list(tokens))

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - set(allowed_special)
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        try:
            return self._inner.encode(text, list(allowed_special))
        except UnicodeEncodeError:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._inner.encode(text, list(allowed_special))

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Decodes a list of tokens into a string.

        WARNING: the default behaviour of this function is lossy, since decoded bytes are not
        guaranteed to be valid UTF-8. You can use `decode_utf8` if you want to raise an error on invalid UTF-8.

        ```
        >>> enc.decode([31373, 995])
        'hello world'
        ```
        """
        data = bytes(self._inner.decode_bytes(list(tokens)))
        return data.decode("utf-8", errors=errors)

    def is_special_token(self, token: int) -> bool:
        """Returns if an individual token is a special token"""
        return self._inner.is_special_token(token)

    # -- Stop tokens --------------------------------------------------

    def stop_tokens(self) -> List[int]:
        return self._inner.stop_tokens()

    def stop_tokens_for_assistant_actions(self) -> List[int]:
        return self._inner.stop_tokens_for_assistant_actions()


class StreamState(Enum):
    EXPECT_START = "ExpectStart"
    HEADER = "Header"
    CONTENT = "Content"


class StreamableParser:
    """Incremental parser over completion tokens."""

    def __init__(self, encoding: HarmonyEncoding, role: Role | None):
        role_str = str(role.value) if role is not None else None
        self._inner = _PyStreamableParser(encoding._inner, role_str)

    def process(self, token: int) -> "StreamableParser":
        self._inner.process(token)
        return self

    def process_eos(self) -> "StreamableParser":
        self._inner.process_eos()
        return self

    @property
    def current_content(self) -> str:
        return self._inner.current_content

    @property
    def current_role(self) -> Optional[Role]:
        raw = self._inner.current_role
        return Role(raw) if raw is not None else None

    @property
    def current_content_type(self) -> Optional[str]:
        return self._inner.current_content_type

    @property
    def last_content_delta(self) -> Optional[str]:
        return self._inner.last_content_delta

    @property
    def messages(self) -> List[Message]:
        raw = self._inner.messages
        return [Message.from_dict(m) for m in json.loads(raw)]

    @property
    def tokens(self) -> List[int]:
        return self._inner.tokens

    @property
    def state_data(self) -> Dict[str, Any]:
        """Return a JSON string representing the parser's internal state."""
        return json.loads(self._inner.state)

    @property
    def state(self) -> StreamState:
        data = self.state_data
        return StreamState(data["state"])

    @property
    def current_recipient(self) -> Optional[str]:
        return self._inner.current_recipient

    @property
    def current_channel(self) -> Optional[str]:
        return self._inner.current_channel


# Public helper --------------------------------------------------------------


def load_harmony_encoding(name: str | "HarmonyEncodingName") -> HarmonyEncoding:  # type: ignore[name-defined]
    """Load an encoding by *name* (delegates to the Rust implementation)."""

    # Allow both strings and enum values.
    if not isinstance(name, str):
        name = str(name)

    inner: _PyHarmonyEncoding = _load_harmony_encoding(name)
    return HarmonyEncoding(inner)


def load_harmony_encoding_from_file(
    name: str,
    vocab_file: str,
    special_tokens: list[tuple[str, int]],
    pattern: str,
    n_ctx: int,
    max_message_tokens: int,
    max_action_length: int,
    expected_hash: str | None = None,
) -> HarmonyEncoding:
    """Load a HarmonyEncoding from a local vocab file (offline usage).
    Use this when network access is restricted or for reproducible builds where you want to avoid remote downloads.
    """
    inner: _PyHarmonyEncoding = _load_harmony_encoding_from_file(
        name,
        vocab_file,
        special_tokens,
        pattern,
        n_ctx,
        max_message_tokens,
        max_action_length,
        expected_hash,
    )
    return HarmonyEncoding(inner)


# For *mypy* we expose a minimal stub of the `HarmonyEncodingName` enum.  At
# **runtime** the user is expected to pass the *string* names because the Rust
# side only operates on strings anyway.


class HarmonyEncodingName(str, Enum):  # noqa: D101 – simple enum stub
    HARMONY_GPT_OSS = "HarmonyGptOss"

    def __str__(self) -> str:  # noqa: D401
        return str(self.value)


# What should be re-exported when the user does ``from harmony import *``?
__all__ = [
    "Role",
    "Author",
    "Content",
    "TextContent",
    "ToolDescription",
    "SystemContent",
    "Message",
    "Conversation",
    "HarmonyEncoding",
    "HarmonyEncodingName",
    "load_harmony_encoding",
    "StreamableParser",
    "StreamState",
    "HarmonyError",
    "load_harmony_encoding_from_file",
]

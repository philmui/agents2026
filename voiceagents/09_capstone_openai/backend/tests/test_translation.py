import asyncio
import base64
import json
import unittest
from array import array
from unittest.mock import patch

from fastapi.testclient import TestClient

from src import main


def pcm16_base64(value: int, samples: int) -> str:
    return base64.b64encode(array("h", [value]) * samples).decode()


class FakeOpenAI:
    """Base in-memory stand-in for one authenticated OpenAI WebSocket."""

    def __init__(self, configuration_event: dict[str, object] | None = None):
        self.configuration_event = configuration_event or {"type": "session.updated"}
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []

    async def recv(self) -> str:
        return await self.events.get()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self.events.get()


class FakeTranslation(FakeOpenAI):
    def __init__(self, configuration_event: dict[str, object] | None = None):
        super().__init__(configuration_event)
        self.sent_output = False

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "session.update":
            await self.events.put(json.dumps({"type": "session.created"}))
            await self.events.put(json.dumps(self.configuration_event))
        elif (
            event["type"] == "session.input_audio_buffer.append"
            and not self.sent_output
        ):
            self.sent_output = True
            # Deliberately emit no session.input_transcript.delta. This reproduces
            # the live service behavior that left "You said" blank.
            await self.events.put(
                json.dumps({"type": "session.output_transcript.delta", "delta": "hola"})
            )
            await self.events.put(
                json.dumps({"type": "session.output_audio.delta", "delta": "AA=="})
            )
        elif event["type"] == "session.close":
            await self.events.put(json.dumps({"type": "session.closed"}))


class FakeSourceTranscriber(FakeOpenAI):
    def __init__(self, configuration_event: dict[str, object] | None = None):
        super().__init__(configuration_event)
        self.item_number = 0

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "session.update":
            await self.events.put(json.dumps({"type": "session.created"}))
            await self.events.put(json.dumps(self.configuration_event))
        elif event["type"] == "input_audio_buffer.commit":
            self.item_number += 1
            item_id = f"source-{self.item_number}"
            await self.events.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": item_id,
                        "delta": "hello",
                    }
                )
            )
            await self.events.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "transcript": "Hello.",
                    }
                )
            )


class FakeConnect:
    def __init__(
        self,
        translation: FakeTranslation,
        source_transcriber: FakeSourceTranscriber,
    ):
        self.translation = translation
        self.source_transcriber = source_transcriber

    def __call__(self, url: str, *args, **kwargs):
        socket = (
            self.source_transcriber
            if url == main.SOURCE_TRANSCRIBE_URL
            else self.translation
        )

        class Context:
            async def __aenter__(self):
                return socket

            async def __aexit__(self, *exc):
                return None

        return Context()


class TranslationTests(unittest.TestCase):
    def test_session_updates_keep_translation_and_transcription_schemas_separate(self):
        self.assertEqual(
            main.build_translation_session_update("es"),
            {
                "type": "session.update",
                "session": {"audio": {"output": {"language": "es"}}},
            },
        )
        self.assertEqual(
            main.build_source_transcription_session_update(),
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {
                                "model": "gpt-realtime-whisper",
                                "delay": "low",
                            },
                        }
                    },
                },
            },
        )

    def test_phrase_detector_commits_after_speech_then_silence(self):
        detector = main.SourcePhraseDetector()
        self.assertFalse(detector.add_chunk(pcm16_base64(1000, 6000)))
        self.assertTrue(detector.add_chunk(pcm16_base64(0, 18000)))
        self.assertTrue(detector.has_uncommitted_speech())
        detector.reset()
        self.assertFalse(detector.has_uncommitted_speech())

    def test_phrase_detector_does_not_count_leading_silence_as_phrase_duration(self):
        detector = main.SourcePhraseDetector()
        self.assertFalse(detector.add_chunk(pcm16_base64(0, main.SAMPLE_RATE * 5)))
        self.assertFalse(detector.add_chunk(pcm16_base64(1000, 6000)))
        self.assertTrue(detector.add_chunk(pcm16_base64(0, 18000)))

    def test_proxy_uses_sidecar_for_source_and_drains_before_close(self):
        translation = FakeTranslation()
        source_transcriber = FakeSourceTranscriber()
        connect = FakeConnect(translation, source_transcriber)

        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(main.websockets, "connect", connect),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect("/translate") as socket:
                socket.send_json({"type": "start", "language": "es"})
                self.assertEqual(socket.receive_json(), {"type": "ready"})

                loud_audio = pcm16_base64(1000, 6000)
                silent_audio = pcm16_base64(0, 18000)
                socket.send_json({"type": "audio", "audio": loud_audio})
                socket.send_json({"type": "audio", "audio": silent_audio})

                messages = [socket.receive_json() for _ in range(4)]
                self.assertIn({"type": "target", "delta": "hola"}, messages)
                self.assertIn({"type": "audio", "delta": "AA=="}, messages)
                self.assertIn(
                    {
                        "type": "source",
                        "item_id": "source-1",
                        "delta": "hello",
                    },
                    messages,
                )
                self.assertIn(
                    {
                        "type": "source",
                        "item_id": "source-1",
                        "transcript": "Hello.",
                        "completed": True,
                    },
                    messages,
                )

                socket.send_json({"type": "stop"})
                self.assertEqual(socket.receive_json(), {"type": "closed"})

        self.assertEqual(
            translation.sent[0],
            {
                "type": "session.update",
                "session": {"audio": {"output": {"language": "es"}}},
            },
        )
        self.assertEqual(
            source_transcriber.sent[0],
            main.build_source_transcription_session_update(),
        )
        self.assertEqual(
            [
                event["audio"]
                for event in source_transcriber.sent
                if event["type"] == "input_audio_buffer.append"
            ],
            [loud_audio, silent_audio],
        )
        self.assertEqual(
            sum(
                event["type"] == "input_audio_buffer.commit"
                for event in source_transcriber.sent
            ),
            1,
        )

    def test_proxy_forwards_configuration_error_without_ready(self):
        translation = FakeTranslation(
            {
                "type": "error",
                "error": {"message": "configuration rejected"},
            }
        )
        source_transcriber = FakeSourceTranscriber()
        with (
            patch.object(main, "OPENAI_API_KEY", "sk-test-only"),
            patch.object(
                main.websockets,
                "connect",
                FakeConnect(translation, source_transcriber),
            ),
            TestClient(main.app) as client,
        ):
            with client.websocket_connect("/translate") as socket:
                socket.send_json({"type": "start", "language": "es"})
                self.assertEqual(
                    socket.receive_json(),
                    {"type": "error", "message": "configuration rejected"},
                )


if __name__ == "__main__":
    unittest.main()

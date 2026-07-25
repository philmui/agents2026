// =============================================================================
// lib/rawWebrtc.ts  —  SECOND PATH: "How it really works" (no SDK).
// =============================================================================
//
// The SDK path (lib/useRealtime.ts) is what you should ship. THIS file exists
// so you can see exactly what the SDK does for you: a real WebRTC handshake
// with OpenAI, done by hand with the browser's built-in RTCPeerConnection.
//
// Nothing here imports @openai/agents. It is plain browser APIs plus one fetch.
// Read it top to bottom; every line is commented. It is wired to the same UI
// via a small toggle in app/page.tsx.
//
// WHY WEBRTC (NOT WEBSOCKET) IN THE BROWSER?  (a required teaching point)
//   - WebRTC is built for live MEDIA: it carries audio over UDP, so a lost
//     packet is skipped instead of stalling the stream (low latency).
//   - The browser's WebRTC stack does echo cancellation and noise suppression
//     on the mic for free, so the assistant does not hear itself and loop.
//   - WebRTC handles NAT traversal, so it works from behind home routers.
//   A server, by contrast, has none of those needs and uses a WebSocket
//   (see modules 02-05). "WebRTC for browsers, WebSocket for servers."
// =============================================================================

"use client"; // getUserMedia, RTCPeerConnection, and <audio> are browser-only.

// The realtime "calls" endpoint that accepts our SDP offer. From API_FACTS.md.
const CALLS_URL = "https://api.openai.com/v1/realtime/calls";

// The voice-assistant model id. From API_FACTS.md (older name: "gpt-realtime-2").
const MODEL = "gpt-realtime-2.1";

// A callback the UI passes in so we can push transcript updates back to React.
// role tells the UI who spoke; text is the (partial) words; done marks the end.
export type RawTranscriptHandler = (line: {
  id: string;
  role: "user" | "assistant";
  text: string;
  done: boolean;
}) => void;

// Everything we need to later tear the connection down cleanly.
export type RawConnection = {
  pc: RTCPeerConnection; // the peer connection (media + data)
  dc: RTCDataChannel; // the JSON event channel named "oai-events"
  micStream: MediaStream; // the microphone tracks, so we can stop them
  close: () => void; // one call to hang up and free the mic
};

/**
 * Open a raw WebRTC session to gpt-realtime-2.1.
 *
 * @param ephemeralKey  the short-lived "ek_..." token from OUR backend
 * @param audioEl       an <audio> element to play the assistant's voice into
 * @param onTranscript  called as user/assistant transcripts stream in
 */
export async function connectRawWebrtc(
  ephemeralKey: string,
  audioEl: HTMLAudioElement,
  onTranscript: RawTranscriptHandler
): Promise<RawConnection> {
  // ---------------------------------------------------------------------------
  // STEP 1 — Create the peer connection.
  // An RTCPeerConnection is the browser object that speaks WebRTC: it will
  // negotiate audio and a data channel with the far end (OpenAI).
  // ---------------------------------------------------------------------------
  const pc = new RTCPeerConnection();

  // ---------------------------------------------------------------------------
  // STEP 2 — Play whatever audio the far end sends us.
  // When OpenAI adds its audio track, `ontrack` fires; we point our <audio>
  // element at that incoming stream so you HEAR the assistant.
  // ---------------------------------------------------------------------------
  pc.ontrack = (event) => {
    audioEl.srcObject = event.streams[0];
  };

  // ---------------------------------------------------------------------------
  // STEP 3 — Capture the microphone and send it.
  // getUserMedia pops the browser's mic-permission prompt. We add the mic track
  // to the connection so our voice flows UP to the model.
  // ---------------------------------------------------------------------------
  const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  for (const track of micStream.getTracks()) {
    pc.addTrack(track, micStream);
  }

  // ---------------------------------------------------------------------------
  // STEP 4 — Open the JSON events data channel.
  // Media rides on its own audio channel; text EVENTS (transcripts, VAD marks,
  // session config) ride on a data channel that MUST be named "oai-events".
  // ---------------------------------------------------------------------------
  const dc = pc.createDataChannel("oai-events");

  // When an event arrives from the server, it is a JSON string. We parse it and
  // pick out the two transcript streams we care about. NOTE the exact names:
  //   - assistant words:  response.output_audio_transcript.delta / .done
  //   - your words:       conversation.item.input_audio_transcription.completed
  // (Audio BYTES would come as response.output_audio.delta, but with WebRTC the
  //  audio is played for us via ontrack above, so we never touch raw bytes here.)
  dc.addEventListener("message", (event) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(event.data);
    } catch {
      return; // ignore anything that is not JSON
    }

    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return;
    }

    const msg = parsed as Record<string, unknown>;
    const itemId = typeof msg.item_id === "string" ? msg.item_id : "";

    switch (msg.type) {
      // Assistant's speech, transcribed, arriving a few words at a time.
      case "response.output_audio_transcript.delta":
        onTranscript({
          id: itemId,
          role: "assistant",
          text: typeof msg.delta === "string" ? msg.delta : "",
          done: false,
        });
        break;

      // Assistant finished this turn: `transcript` holds the full final text.
      case "response.output_audio_transcript.done":
        onTranscript({
          id: itemId,
          role: "assistant",
          text: typeof msg.transcript === "string" ? msg.transcript : "",
          done: true,
        });
        break;

      // Your speech, transcribed once you stop talking.
      case "conversation.item.input_audio_transcription.completed":
        onTranscript({
          id: itemId,
          role: "user",
          text: typeof msg.transcript === "string" ? msg.transcript : "",
          done: true,
        });
        break;

      // (Many other event types exist; we ignore them in this minimal demo.)
      default:
        break;
    }
  });

  // ---------------------------------------------------------------------------
  // STEP 5 — Create an SDP OFFER and set it as our local description.
  // SDP ("Session Description Protocol") is a plain-text menu describing the
  // media we want to exchange (codecs, directions). createOffer() builds ours;
  // setLocalDescription() also starts gathering network candidates.
  // ---------------------------------------------------------------------------
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // ---------------------------------------------------------------------------
  // STEP 6 — POST the offer to OpenAI and get an SDP ANSWER back.
  // The body is the raw SDP TEXT (not JSON), so Content-Type is
  // "application/sdp". We authenticate with the EPHEMERAL key only. The model
  // is chosen via the ?model= query string.
  // ---------------------------------------------------------------------------
  const sdpResponse = await fetch(`${CALLS_URL}?model=${MODEL}`, {
    method: "POST",
    body: offer.sdp, // the offer text we just made
    headers: {
      Authorization: `Bearer ${ephemeralKey}`, // ek_..., never the real key
      "Content-Type": "application/sdp",
    },
  });

  if (!sdpResponse.ok) {
    const detail = await sdpResponse.text().catch(() => "");
    // Clean up the half-open connection before throwing.
    micStream.getTracks().forEach((t) => t.stop());
    pc.close();
    throw new Error(
      `WebRTC handshake failed: HTTP ${sdpResponse.status}. ${detail}`.trim()
    );
  }

  // ---------------------------------------------------------------------------
  // STEP 7 — Apply the server's answer.
  // The response body is the far end's SDP answer (again plain text). Setting it
  // as our remote description completes the handshake; media + the data channel
  // now connect, ontrack fires, and you can start talking.
  // ---------------------------------------------------------------------------
  const answerSdp = await sdpResponse.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

  // Hand back a tidy object with a one-call teardown.
  const close = () => {
    try {
      dc.close();
    } catch {
      /* already closed */
    }
    micStream.getTracks().forEach((t) => t.stop()); // turn the mic light off
    pc.close();
  };

  return { pc, dc, micStream, close };
}

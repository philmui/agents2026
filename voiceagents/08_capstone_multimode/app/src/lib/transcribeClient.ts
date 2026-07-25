// ---------------------------------------------------------------------------
// TRANSCRIBE MODE — live speech-to-text in the browser, over raw WebRTC.
//
// This deliberately uses the HAND-BUILT WebRTC handshake from Module 07 (not
// the SDK) so you can see exactly what the SDK hides. The only differences from
// an assistant session are:
//   1) the session type is "transcription" (we only want text back, no voice)
//   2) we listen for  conversation.item.input_audio_transcription.completed
//      which carries the text of what YOU said.
//
// The WebRTC recipe (RTCPeerConnection, mic track, "oai-events" data channel,
// POST the SDP offer to /v1/realtime/calls) is verified in API_FACTS §5.
// ---------------------------------------------------------------------------

// Callbacks the UI hands us so we can push updates back to React.
type TranscribeHandlers = {
  onPartial: (text: string) => void; // interim words as you speak
  onFinal: (text: string) => void; // a finished sentence/segment
  onStatus: (status: string) => void; // human-readable connection status
  onError: (message: string) => void;
};

export class TranscribeClient {
  private pc: RTCPeerConnection | null = null;
  private dc: RTCDataChannel | null = null;
  private mic: MediaStream | null = null;
  private handlers: TranscribeHandlers;

  constructor(handlers: TranscribeHandlers) {
    this.handlers = handlers;
  }

  // Open the mic, do the WebRTC handshake, and start transcribing.
  // `token` is the ephemeral "ek_..." key from our backend.
  async start(token: string) {
    this.handlers.onStatus("connecting");

    // 1) A peer connection is the WebRTC pipe to OpenAI.
    const pc = new RTCPeerConnection();
    this.pc = pc;
    pc.addEventListener("connectionstatechange", () => {
      if (pc.connectionState === "failed") {
        this.handlers.onError("The transcription WebRTC connection failed.");
      } else if (pc.connectionState === "disconnected") {
        this.handlers.onStatus("disconnected");
      }
    });

    // 2) Ask the browser for the microphone and add it as an outgoing track.
    //    getUserMedia pops the browser's "allow microphone?" permission prompt.
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mic = mic;
    pc.addTrack(mic.getTracks()[0], mic);

    // 3) The "oai-events" data channel is the two-way JSON event stream. This is
    //    the SAME channel name the SDK uses under the hood (API_FACTS §5).
    const dc = pc.createDataChannel("oai-events");
    this.dc = dc;

    // As soon as the channel opens, tell the server we want TRANSCRIPTION only.
    dc.addEventListener("open", () => {
      this.handlers.onStatus("listening");
      // session.update configures the live session. Note the GA nesting:
      // audio settings live under session.audio.input (API_FACTS §3).
      const configure = {
        type: "session.update",
        session: {
          type: "transcription",
          audio: {
            input: {
              // PCM16 @ 24 kHz mono is the Realtime audio format (API_FACTS §3).
              format: { type: "audio/pcm", rate: 24000 },
              // The dedicated realtime transcription model (API_FACTS §1).
              transcription: {
                model: "gpt-realtime-whisper",
                delay: "low",
              },
            },
          },
        },
      };
      dc.send(JSON.stringify(configure));
    });

    // 4) Every server event arrives here as a JSON string. We only care about a
    //    few event types; the exact strings are verified in API_FACTS §4.
    dc.addEventListener("message", (e) => this.onServerEvent(e.data));
    dc.addEventListener("error", () => {
      this.handlers.onError("The transcription event channel failed.");
    });

    // 5) Standard WebRTC "offer/answer" handshake.
    //    a) create our offer and set it as the local description
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    //    b) POST the offer's SDP to OpenAI, authorized with the ephemeral token.
    //       The Content-Type MUST be application/sdp (API_FACTS §5).
    const sdpRes = await fetch(
      "https://api.openai.com/v1/realtime/calls",
      {
        method: "POST",
        body: offer.sdp,
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/sdp",
        },
      },
    );

    if (!sdpRes.ok) {
      const detail = await sdpRes.text();
      this.stop();
      throw new Error(`Handshake failed (${sdpRes.status}): ${detail}`);
    }

    //    c) OpenAI's SDP "answer" completes the connection.
    const answer = { type: "answer" as const, sdp: await sdpRes.text() };
    await pc.setRemoteDescription(answer);
  }

  // Parse one server event and route the interesting ones to the UI.
  private onServerEvent(raw: string) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return; // ignore anything that is not JSON
    }

    if (typeof parsed !== "object" || parsed === null) return;
    const event = parsed as Record<string, unknown>;

    switch (event.type) {
      // Interim transcript of the user's speech, updated as you talk.
      case "conversation.item.input_audio_transcription.delta":
        if (typeof event.delta === "string") this.handlers.onPartial(event.delta);
        break;

      // A finished segment of the user's speech. This is the reliable one.
      case "conversation.item.input_audio_transcription.completed":
        if (typeof event.transcript === "string")
          this.handlers.onFinal(event.transcript);
        this.handlers.onStatus("listening");
        break;

      // The server tells us when it heard speech start/stop (nice for a mic UI).
      case "input_audio_buffer.speech_started":
        this.handlers.onStatus("hearing you...");
        break;
      case "input_audio_buffer.speech_stopped":
        this.handlers.onStatus("listening");
        break;

      // Surface server-side errors instead of failing silently.
      case "error": {
        const error = event.error;
        const message =
          typeof error === "object" &&
          error !== null &&
          "message" in error &&
          typeof error.message === "string"
            ? error.message
            : "Unknown server error";
        this.handlers.onError(message);
        break;
      }
    }
  }

  // gpt-realtime-whisper does not support server VAD. Committing marks the end
  // of the current phrase so OpenAI can emit a completed transcript item.
  commit() {
    if (this.dc?.readyState !== "open") return;
    this.dc.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
    this.handlers.onStatus("processing");
  }

  // Tear everything down: stop the mic, close the channel and the connection.
  stop() {
    this.handlers.onStatus("stopped");
    if (this.mic) this.mic.getTracks().forEach((t) => t.stop());
    if (this.dc) this.dc.close();
    if (this.pc) this.pc.close();
    this.mic = null;
    this.dc = null;
    this.pc = null;
  }
}

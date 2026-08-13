// ---------------------------------------------------------------------------
// ASSIST MODE — a speech-to-speech voice assistant that can CALL A TOOL.
//
// This is the headline of the capstone. It uses the official browser SDK,
// `@openai/agents/realtime`, which wraps the WebRTC plumbing you built by hand
// in Module 07. You describe an "agent" (its instructions + its tools), open a
// "session", connect with an ephemeral token, and then just talk.
//
// THE TOOLS and the ReAct loop
// ----------------------------
// We give the agent a local clock tool and a backend-powered web search tool.
// When you ask "what time is it in
// Tokyo?", the model does a small reason -> act -> observe -> respond loop:
//   reason  : "The user wants the current time; I have a get_time tool."
//   act     : it CALLS get_time({ timeZone: "Asia/Tokyo" })
//   observe : our execute() runs in the browser and returns the real clock time
//   respond : the model SPEAKS that result back to you in natural language
// The model never actually knows the time on its own; the tool is its senses.
// ---------------------------------------------------------------------------

import { RealtimeAgent, RealtimeSession, tool } from "@openai/agents/realtime";
import { z } from "zod";
import { BACKEND_URL } from "@/lib/backend";

// The canonical voice-assistant model id (see docs/API_FACTS.md §1).
const MODEL = "gpt-realtime-2.1";

// --- The tool ------------------------------------------------------------
// tool() turns a plain function into something the model is allowed to call.
//   name        : what the model refers to it by
//   description : plain English so the model knows WHEN to use it
//   parameters  : a zod schema describing the arguments (the SDK converts this
//                 to the JSON schema OpenAI needs, and validates the model's
//                 arguments before your code runs)
//   execute     : YOUR code. It runs in the browser, returns a value, and that
//                 value becomes the model's "observation".
export const getTimeTool = tool({
  name: "get_time",
  description:
    "Get the current wall-clock time. Optionally for a specific IANA time " +
    "zone such as 'Asia/Tokyo' or 'America/New_York'. Use this whenever the " +
    "user asks what time it is.",
  parameters: z.object({
    // z.nullable(...) lets the model omit the zone. Realtime tool schemas must
    // be strict (no truly optional fields), so we model "no zone given" as null
    // and default to the browser's own time zone below.
    timeZone: z
      .string()
      .nullable()
      .describe("IANA time zone id, e.g. 'Asia/Tokyo'. Null = user's local time."),
  }),
  execute: async ({ timeZone }) => {
    // Fall back to the browser's own time zone when the model passes null.
    const zone =
      timeZone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    // Format the CURRENT time in that zone. This is real data from the device,
    // which is the whole point: the model gets a fact it could not invent.
    const now = new Date().toLocaleTimeString("en-US", {
      timeZone: zone,
      hour: "2-digit",
      minute: "2-digit",
    });
    // Whatever we return becomes the model's observation. Keep it short and
    // literal so the model can read it back cleanly.
    return `The current time in ${zone} is ${now}.`;
  },
});

// Realtime sessions accept function tools, so web search is represented here as
// a normal function call. Its execute() delegates to our FastAPI backend, which
// safely uses the permanent API key to run the Responses API hosted web_search
// tool. The browser receives only the grounded answer, never the key.
export const webSearchTool = tool({
  name: "web_search",
  description:
    "Search the public web for current or factual information. Use this for " +
    "recent events, news, changing facts, or whenever the user asks you to " +
    "search online. Returns a concise plain-text answer with source names, or " +
    "a message beginning 'Web search failed:' if the lookup could not run.",
  parameters: z.object({
    query: z
      .string()
      .describe("A concise, self-contained web search query."),
  }),
  execute: async ({ query }) => {
    try {
      const response = await fetch(`${BACKEND_URL}/web-search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const raw = await response.text();

      if (!response.ok) {
        let detail = raw;
        try {
          const parsed: unknown = JSON.parse(raw);
          if (
            typeof parsed === "object" &&
            parsed !== null &&
            "detail" in parsed &&
            typeof parsed.detail === "string"
          ) {
            detail = parsed.detail;
          }
        } catch {
          // Keep the raw response when the backend did not return JSON.
        }
        throw new Error(`backend returned ${response.status}: ${detail}`);
      }

      const data: unknown = JSON.parse(raw);
      if (
        typeof data !== "object" ||
        data === null ||
        !("answer" in data) ||
        typeof data.answer !== "string"
      ) {
        throw new Error("backend response did not contain an answer");
      }
      return data.answer;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      // Return a normal observation so a temporary search failure does not tear
      // down the live voice session; the agent can explain it and keep talking.
      return `Web search failed: ${detail}`;
    }
  },
});

// --- The agent -----------------------------------------------------------
// A RealtimeAgent bundles the persona (instructions) with its tools. Adding
// function tools to a voice agent is identical to adding one to a text agent.
export function makeAssistant(): RealtimeAgent {
  return new RealtimeAgent({
    name: "Capstone Assistant",
    instructions:
      "You are a friendly, concise voice assistant for a coding class. " +
      "Speak in short, clear sentences. When the user asks for the time, " +
      "call the get_time tool and then say the result out loud. Do not guess " +
      "the time yourself. Use web_search for current information, recent " +
      "events, changing facts, or whenever the user asks you to search the web. " +
      "Never invent search results; summarize the tool's observation.",
    tools: [getTimeTool, webSearchTool],
  });
}

// --- The session ---------------------------------------------------------
// A RealtimeSession is one live conversation. We pass the agent plus the model.
// The SDK handles the microphone, the WebRTC connection, and playback for us.
export function makeSession(agent: RealtimeAgent): RealtimeSession {
  return new RealtimeSession(agent, {
    model: MODEL,
    config: {
      // Only speak (audio out); we read the assistant's words from transcript
      // events for the on-screen caption.
      outputModalities: ["audio"],
      // Low reasoning effort keeps voice latency snappy (API_FACTS §7).
      reasoning: { effort: "low" },
    },
  });
}

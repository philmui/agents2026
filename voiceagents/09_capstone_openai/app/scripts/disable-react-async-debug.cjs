// =============================================================================
// scripts/disable-react-async-debug.cjs  -  DEV-ONLY crash fix (preload).
// =============================================================================
//
// WHY THIS FILE EXISTS
// --------------------
// In development, the Next.js dev server sometimes dies after running a while:
//
//     RangeError: Map maximum size exceeded
//         at Map.set (<anonymous>)
//         at AsyncHook.init (.../next-server/app-page-turbo.runtime.dev.js)
//         ... at hot-reloader-turbopack.js  (the HMR watch loop)
//
// ROOT CAUSE (dev-only; production is unaffected):
//   React 19 ships a DEV feature, `enableAsyncDebugInfo`, that builds "async
//   owner stacks" for awaited promises in Server Components. It installs a Node
//   `async_hooks` hook whose `init` callback records EVERY promise the process
//   creates into a plain `Map` (`pendingOperations.set(asyncId, ...)`). The only
//   cleanup is a `destroy` callback, and for promises `destroy` runs only on GC.
//   Turbopack's HMR watch loop creates a steady stream of promises, so the Map
//   grows faster than GC prunes it and eventually hits V8's hard ceiling for a
//   single Map (2^24 = 16,777,216 entries). The next `Map.set` throws the
//   RangeError above, inside an async-hook callback that cannot be caught, so the
//   whole dev server exits. The flag is compiled-in with no env/config off-switch.
//
// WHAT THIS PRELOAD DOES:
//   It runs (via NODE_OPTIONS="--require ./scripts/disable-react-async-debug.cjs")
//   BEFORE any Next.js/React code loads, and makes that ONE debug async-hook a
//   no-op so the leaking Map is never populated. It does this narrowly:
//   `async_hooks.createHook(...)` still returns a real hook object, but its
//   `.enable()` becomes a no-op *only* for the React debug hook (identified by
//   its exact 4-callback shape: init + before + promiseResolve + destroy).
//   Everything else - including `AsyncLocalStorage`, which does NOT use
//   `createHook` - is untouched.
//
//   The only thing lost is React's DEV async owner-stack debug info, which this
//   client-heavy voice app does not rely on. Production builds never load this.
// =============================================================================

"use strict";

// Only ever meddle in development. In production the leaking hook does not exist,
// and we must not change async behavior of a real server.
if (process.env.NODE_ENV === "production") {
  return;
}

const asyncHooks = require("async_hooks");
const realCreateHook = asyncHooks.createHook;

if (typeof realCreateHook === "function" && !realCreateHook.__patchedForReactLeak) {
  const patched = function createHook(callbacks) {
    const hook = realCreateHook.call(this, callbacks);

    // React's async-debug tracker (`initAsyncDebugInfo`) is the ONLY hook that
    // registers all FOUR of these lifecycle callbacks together:
    //   init  -> records each async resource into the leaking Map
    //   before, promiseResolve, destroy -> walk / prune it
    // Requiring all four (not just three) makes the match precise: a generic
    // diagnostic hook that omits `before` (or any of them) is left running.
    //
    // This heuristic is deliberately FAIL-SAFE either way:
    //   - if React changes its shape and we MISS it, the app still runs (it just
    //     reverts to the old crash) - we never break a working setup;
    //   - if we ever OVER-match some unrelated hook, the only effect is that its
    //     async tracking is off in DEV - never a correctness change in prod.
    const isReactAsyncDebugHook =
      callbacks &&
      typeof callbacks.init === "function" &&
      typeof callbacks.before === "function" &&
      typeof callbacks.promiseResolve === "function" &&
      typeof callbacks.destroy === "function";

    if (isReactAsyncDebugHook) {
      // Neutralize it: enabling it is what registers the leaking init callback.
      hook.enable = function noopEnable() {
        return hook;
      };
    }
    return hook;
  };
  patched.__patchedForReactLeak = true;
  asyncHooks.createHook = patched;
}

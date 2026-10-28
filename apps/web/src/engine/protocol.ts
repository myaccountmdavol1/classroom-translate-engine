/**
 * What the pages, the engine host and the Service Worker say to each other
 * when the engine runs in this browser (NEXT_PUBLIC_ENGINE_URL=/engine).
 *
 * public/engine-sw.js is a plain file Next does not build, so it repeats
 * the few constants it needs; keep the two in step.
 */

/** Same-origin requests under this prefix are the engine's API. */
export const LOCAL_ENGINE_URL = "/engine";

/** True when this build runs the engine in the browser. Read once at build
 * time: with the production value (the Render address) it is false and no
 * other code in src/engine ever runs. */
export const ENGINE_IS_LOCAL =
  process.env.NEXT_PUBLIC_ENGINE_URL === LOCAL_ENGINE_URL;

/** Where Word, PowerPoint and Excel files go, and documents this browser
 * does not hold: the Render engine. Empty when there is none. */
export const REMOTE_ENGINE_URL = (
  process.env.NEXT_PUBLIC_REMOTE_ENGINE_URL ?? ""
).replace(/\/+$/, "");

/** BroadcastChannel shared by every tab: status from the host, and pages
 * asking it to drop a request they gave up on. */
export const ENGINE_CHANNEL = "ct-engine";

/** Names each engine request, so a page that gave up on one (its timeout)
 * can ask the host to drop it if Python has not started it. An aborted
 * fetch never reaches the Service Worker, so this is said in so many words. */
export const REQUEST_ID_HEADER = "X-CT-Request";

/** Answered by the host, in JS: whether she pressed Stop on a job that is
 * running. Asked by the worker from inside the job (ct_browser/cancel.py). */
export const STOP_PATH = "/api/__stop/";

/** Set by the Service Worker on every answer that came from Render, so a
 * page can tell Render waking up from the translator in this browser. */
export const ENGINE_FROM_HEADER = "X-CT-Engine";

/** The Web Lock the hosting tab holds while it is open. */
export const ENGINE_LOCK = "ct-engine";

/** BroadcastChannel for changes to the documents this browser keeps (a
 * backup taken, a document deleted from this computer), so every tab's
 * lists follow. */
export const DOCS_CHANNEL = "ct-docs";

/** How long a document made in this browser is kept here: everything about
 * it -- its files, its pictures, the uploaded file kept beside it -- goes
 * together once this has passed since it was uploaded. 24 hours, like the
 * online service (the engine's retention_hours), unless this build sets
 * NEXT_PUBLIC_LOCAL_RETENTION_HOURS. */
export const LOCAL_RETENTION_HOURS =
  Number(process.env.NEXT_PUBLIC_LOCAL_RETENTION_HOURS) > 0
    ? Number(process.env.NEXT_PUBLIC_LOCAL_RETENTION_HOURS)
    : 24;
export const LOCAL_RETENTION_MS = LOCAL_RETENTION_HOURS * 60 * 60 * 1000;
/** The same length as she would say it: "7 days" rather than "168 hours". */
export const LOCAL_RETENTION_WORDS =
  LOCAL_RETENTION_HOURS >= 48 && LOCAL_RETENTION_HOURS % 24 === 0
    ? `${LOCAL_RETENTION_HOURS / 24} days`
    : `${LOCAL_RETENTION_HOURS} hours`;

/** IndexedDB where the host records the documents this browser holds, read
 * by the Service Worker to route a document's requests here or to Render. */
export const INDEX_DB = "ct-engine";
export const INDEX_STORE = "docs";
export const META_STORE = "meta";

export type EngineState = "idle" | "booting" | "ready" | "failed";

/** What the host tells every tab about the engine. */
export interface EngineStatus {
  state: EngineState;
  /** "engine" while the translator downloads and starts, "reader" while the
   * text reader loads before a translation that needs it, "model" while
   * the language's translation model loads before a translation. */
  phase: "engine" | "reader" | "model";
  loaded: number;
  total: number;
  /** A job is running: Python is busy until it ends. */
  busy: boolean;
  /** The document whose job is running. */
  jobDocument: string | null;
  /** Why the engine could not start, in a teacher's words. */
  error: string | null;
  /** The technical reason behind `error`, for whoever looks into it; never
   * shown to her. */
  diagnostic: string | null;
  /** The engine build the hosting tab runs. */
  hash?: string;
  /** When the hosting tab's page was loaded (ms since 1970): of two builds,
   * the tab loaded later has the one the site serves now. */
  since?: number;
  /** Documents are kept in this browser's storage and outlive the tab. False
   * when the browser would not give the storage (they then live only as
   * long as the hosting tab, and the pages keep backups as before). */
  stored?: boolean;
  /** Documents whose latest changes the disk refused (full, or blocked):
   * kept in the translator's memory meanwhile, and backed up by the pages
   * as before, until they are written out whole again. */
  unsafe?: string[];
  /** Why the translator is starting again, while it does: "refresh" when
   * it starts afresh between jobs to free memory (nothing to tell her),
   * "memory" when a job ran out of memory and paused. */
  restart?: "refresh" | "memory" | null;
}

export type ChannelMessage =
  | { type: "status"; status: EngineStatus }
  | { type: "status?" }
  | { type: "abort"; id: string }
  /** A tab running a newer build than the host's: the host finishes the job
   * it is on, saves everything, and hands over. */
  | { type: "newer"; hash: string; since: number }
  /** For "Copy details for support": asked by any tab, answered by the host. */
  | { type: "diagnostics?"; id: string }
  | { type: "diagnostics"; id: string; data: Record<string, unknown> };

export type DocsMessage =
  | { type: "backups" }
  | { type: "recent" }
  | { type: "deleted"; id: string };

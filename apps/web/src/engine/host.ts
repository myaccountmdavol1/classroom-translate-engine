/**
 * The engine host: runs the translator in this browser for every tab.
 *
 * Only used when NEXT_PUBLIC_ENGINE_URL=/engine (see protocol.ts). Then
 * every engine request a page makes -- api.ts's fetches, the upload, the
 * pictures' <img src> -- goes to /engine/api/..., where the Service Worker
 * (public/engine-sw.js) picks it up and hands it to this host over a
 * MessageChannel. The host answers what it can without Python, and passes
 * the rest to the engine worker (engine.worker.js), which runs ct_engine.
 *
 * One tab per browser hosts, chosen with the Web Lock "ct-engine": the
 * browser releases it when that tab closes or crashes, and the next tab
 * waiting for it takes over -- it opens the same documents from this
 * browser's storage, and picks up any translation that was interrupted.
 * Every tab (host or not) registers the Service Worker and answers its pings.
 *
 * Answered here, in JS, so the document page and its progress never wait
 * behind a job that holds Python:
 *   GET /api/health           the engine's limits, plus whether a class code
 *                             is needed (asked of the translation relay)
 *   GET /api/languages, /api/output-modes   the engine's own lists
 *   GET /documents/{id}/job   the newest job, as the engine last saved it
 *   GET /documents/{id}       likewise, while a job runs
 *   POST /documents/{id}/cancel  while a job runs: remembered here (and in
 *                             IndexedDB, so a tab that takes over still
 *                             stops it), and the running job asks for it at
 *                             its next page (GET /api/__stop/{job}, from
 *                             ct_browser/cancel.py); a job waiting behind it
 *                             is also dropped by Python before it can start
 *   DELETE /documents/{id}    "Delete from this computer": every copy goes
 *   a document this browser no longer holds: the engine's own "could not be
 *                             found", which api.ts and backup.ts already know
 *                             how to recover from
 * and the class code is checked with the relay before a translation is
 * queued, so a wrong code is refused before any work starts.
 *
 * Documents are kept in OPFS (opfs-store.ts), written from the engine's own
 * store journal in the order the engine wrote, and listed in IndexedDB
 * (index-db.ts). A document is put into the worker's memory the first time
 * a request names it, and taken out again once it has been idle a while.
 * When the host starts, the documents with unfinished jobs go in first and
 * the engine's own resume picks their jobs up again. After a job, a worker
 * that has grown too big (or ran out of memory) is replaced by a fresh one,
 * which resumes the same way. Everything a browser keeps of a document --
 * its files, its pictures, the uploaded file kept beside it -- is deleted
 * together when it expires or she deletes it.
 *
 * A browser that will not give the storage (blocked, or full) runs as
 * before: documents live in the worker only, and the pages' backups put
 * them back after a reload.
 *
 * If the translator cannot be set up here (three failed starts), this
 * browser uses the Render engine instead when there is one; otherwise it
 * tries again a minute later, or at once when she uploads again.
 */

import {
  ENGINE_BASE,
  ENGINE_BOOT_BYTES,
  ENGINE_HASH,
  ENGINE_HEALTH,
  ENGINE_LANGUAGES,
  ENGINE_OUTPUT_MODES,
} from "./manifest.generated";
import {
  forgetDocument,
  getDocument,
  getMeta,
  isIndexed,
  listDocuments,
  rememberDocument,
  setMeta,
  type IndexedDocument,
} from "./index-db";
import {
  documentOf,
  expiredDocuments,
  isUnfinished,
  OpfsStore,
  type DirLike,
  type JournalEntry,
} from "./opfs-store";
import { chromeReady, chromeTranslate } from "./chrome-mt";
import {
  accessCode,
  fallBackToRemoteEngine,
  forgetRecent,
  parseRecent,
  recentSnapshot,
  rememberRecent,
} from "@/lib/api";
import { forgetKeptCopies } from "@/lib/backup";
import { announceDocs } from "@/lib/docsChannel";
import { pageIsHeld } from "@/lib/leaveGuard";
import {
  ENGINE_CHANNEL,
  ENGINE_LOCK,
  LOCAL_RETENTION_HOURS,
  LOCAL_RETENTION_MS,
  LOCAL_RETENTION_WORDS,
  REMOTE_ENGINE_URL,
  STOP_PATH,
  type ChannelMessage,
  type EngineStatus,
} from "./protocol";

/* --- Wire shapes ------------------------------------------------------- */

/** A request as the Service Worker sends it. `path` is the engine's own
 * ("/api/..."), with its query string. */
interface HostRequest {
  type: "ct-request";
  id: string;
  method: string;
  path: string;
  headers: Record<string, string>;
  body: ArrayBuffer | null;
}

interface HostResponse {
  status: number;
  headers: [string, string][];
  body: ArrayBuffer | null;
}

type Json = Record<string, unknown>;

/* --- Words ------------------------------------------------------------- */

const ACCESS_REFUSED = "This service needs the class access code.";
// Starts like the engine's own sentence, which api.ts recognises as a lost
// document, and says "restarted", which backup.ts reads as "put it back".
// Said only of a document this browser listed but no longer has the files
// of (one made before documents were kept, or storage the browser cleared).
const DOCUMENT_GONE =
  "That document could not be found. This browser no longer has its files (the browser may have cleared them to make space), so the translator on this computer restarted without it. Upload the file again.";
const NOT_FOUND =
  "That document could not be found. It may have been opened on another computer, or removed.";
// No "restarted": nothing is put back (every copy went with it).
const EXPIRED = `That document could not be found. It was removed from this computer because it was more than ${LOCAL_RETENTION_WORDS} old. Upload the file again to translate it.`;
// Why it could not start is kept for the diagnostic; she is told what to do.
const BOOT_FAILED =
  "The translator could not be set up on this computer. Check the internet connection, then try again in a minute.";
const BOOT_FAILED_REMOTE =
  "The translator could not be set up on this computer, so files are now translated online instead. Upload the file again.";
const WORKER_LOST =
  "The translator on this computer stopped while it was working. Please try again.";
const JOB_LOST =
  "The translator on this computer stopped while working on this document. Please translate it again.";
const STILL_TRANSLATING =
  "This document is still being translated. Wait until it finishes, or open it and press Stop, then delete it.";

/* --- Settings ------------------------------------------------------------ */

const MAX_BOOT_ATTEMPTS = 3;
// After the last failed start, how long until a request may try again.
const BOOT_COOL_DOWN_MS = 60000;
// A document nobody has asked about for this long leaves the worker's memory
// (it stays on disk, and comes back the next time a page asks for it).
const EVICT_IDLE_MS = 10 * 60000;
const EVICT_CHECK_MS = 60000;
// How often expired documents are looked for while the tab is open.
const SWEEP_MS = 15 * 60000;
// A document a page asked about this recently is not removed by the sweep
// (it is open in front of her); it goes at a later sweep.
const SWEEP_SPARES_USED_MS = 60 * 60000;
// After the disk refused a write: how soon the document is written out
// again (doubling each time it fails again, up to the second figure).
const RESYNC_FIRST_MS = 30000;
const RESYNC_MOST_MS = 10 * 60000;
// A newer build waits this long between looks for a quiet moment.
const HANDOVER_LOOK_MS = 2000;
// After a job, a worker bigger than this (Python's heap + the text reader's
// + the documents in memory) is replaced by a fresh one. Chosen well under
// the plan's 1.4 GB stop line for a 4 GB Chromebook; no Chromebook has been
// measured yet. A developer can change it: localStorage "ct.engine.recycleMB".
const DEFAULT_RECYCLE_MB = 1200;

function recycleBytes(): number {
  try {
    const mb = Number(window.localStorage.getItem("ct.engine.recycleMB"));
    if (Number.isFinite(mb) && mb > 0) return mb * 1e6;
  } catch {
    /* storage blocked: the default */
  }
  return DEFAULT_RECYCLE_MB * 1e6;
}

/** When this page was loaded: of two builds open at once, the tab loaded
 * later has the build the site serves now. */
const SINCE = Math.round(
  typeof performance !== "undefined" ? performance.timeOrigin : Date.now(),
);

/* --- State ------------------------------------------------------------- */

const status: EngineStatus = {
  state: "idle",
  phase: "engine",
  loaded: 0,
  total: ENGINE_BOOT_BYTES,
  busy: false,
  jobDocument: null,
  error: null,
  diagnostic: null,
  hash: ENGINE_HASH,
  since: SINCE,
};

let isHost = false;
let worker: Worker | null = null;
let bootAttempts = 0;
let lastBootFailure = 0;
let channel: BroadcastChannel | null = null;

// This browser's lasting store, once opened; null when the browser would
// not give it (documents then live in the worker only, as before).
let store: OpfsStore | null = null;
let storeOpened: Promise<void> | null = null;
let storeProblem: string | null = null;

// Requests handed to the worker, by id, until it answers.
const inFlight = new Map<string, { done: (r: HostResponse) => void; method: string; path: string }>();
// Requests that must wait for the engine to start (uploads).
const waitingForBoot: { req: HostRequest; done: (r: HostResponse) => void }[] = [];
// Aborted by their page before they reached the worker. The Service Worker
// names a request by the page's X-CT-Request header when it has one, so the
// page's "abort" names the same request.
const abortedIds = new Set<string>();

// Documents in the running worker's memory, and when a page last asked
// about each. A new worker holds none.
const hydrated = new Map<string, number>();
const hydrating = new Map<string, Promise<"ok" | "missing" | "stale">>();
// Documents this tab's earlier workers held, when there is no lasting store
// (a reload loses them in any case).
const lostDocs = new Set<string>();
// Documents being deleted: anything the worker still writes for them is
// not kept.
const dropped = new Set<string>();
// The documents listed in the index, as this tab knows them.
const indexedIds = new Set<string>();
// The engine's own snapshots, posted as it saves them (ct_browser/boot.py).
const jobs = new Map<string, Json>();
const documents = new Map<string, Json>();
let runningJob: { documentId: string; jobId: string; translation: boolean } | null = null;
// Jobs she asked to stop; the running one reads this (GET /api/__stop/{job}).
const stopRequested = new Set<string>();
// Documents she stopped while their "Translate" was still waiting for
// Python (behind another document's job): the job it makes is stopped as
// soon as it starts.
const stopDocs = new Set<string>();
// Set once this tab is handing the engine to a tab with a newer build.
let leaving = false;
// A newer build asked for the engine: handed over at the next quiet moment.
let handOverWanted = false;
let halted: (() => void) | null = null;
let releaseLock: (() => void) | null = null;
// Between a worker that stopped itself to be replaced and its successor:
// nothing may start another one meanwhile.
let recycling = false;
// Documents whose latest writes the disk refused (see opfs-store.ts): kept
// in the worker's memory, never evicted or recycled away, until written out
// whole again.
const unsafe = new Set<string>();
let resyncDelay = RESYNC_FIRST_MS;
let resyncTimer: ReturnType<typeof setTimeout> | null = null;
// The journal's last number from the running worker (each worker counts
// from 1): a gap means a write that never arrived.
let lastSeq = 0;
// Jobs picked up again only because the worker started afresh between jobs
// (not something she would call a pause), by how many times.
const quietResumes = new Map<string, number>();
// Documents removed because their time was up, so a page still showing one
// is told so (their ids only; the most recent few hundred).
let expiredIds: string[] = [];

// For "Copy details for support".
const stats = {
  boots: 0,
  lastBootSeconds: null as number | null,
  lastBootBytes: null as number | null,
  recycles: 0,
  lastRecycle: null as Json | null,
  crashes: 0,
  lastCrash: null as string | null,
  resumedJobs: 0,
  hydrations: 0,
  evictions: 0,
  lastMemory: null as Json | null,
  peakMemory: 0,
  journalProblems: 0,
  lastJournalProblem: null as string | null,
  // The last translation model that could not be loaded, and why.
  lastModelProblem: null as string | null,
  lastSweep: null as Json | null,
  persist: null as Json | null,
  becameHostAt: null as number | null,
  unsafeDocuments: 0,
  resyncs: 0,
  lastResync: null as Json | null,
  journalGaps: 0,
  lastStopChecks: null as Json | null,
  handOverWaits: 0,
};

/* --- Small helpers ------------------------------------------------------- */

const encoder = new TextEncoder();

function json(status: number, body: unknown, extra: [string, string][] = []): HostResponse {
  const bytes = encoder.encode(JSON.stringify(body));
  return {
    status,
    headers: [["content-type", "application/json"], ["cache-control", "no-store"], ...extra],
    body: bytes.buffer as ArrayBuffer,
  };
}

function errorBody(message: string, extra: Json = {}): Json {
  return { error: message, detail: message, diagnostic: null, ...extra };
}

/** The refusal api.ts reads as "the engine is not there yet, ask again":
 * 503 with no body and a Retry-After, never applied, safe to resend. */
function notYet(): HostResponse {
  return { status: 503, headers: [["retry-after", "2"]], body: null };
}

function header(req: HostRequest, name: string): string | null {
  const lower = name.toLowerCase();
  for (const [k, v] of Object.entries(req.headers)) if (k.toLowerCase() === lower) return v;
  return null;
}

function publishStatus(): void {
  status.stored = store !== null;
  status.unsafe = [...unsafe];
  const message: ChannelMessage = { type: "status", status: { ...status } };
  try {
    channel?.postMessage(message);
  } catch {
    /* a closed channel: the tab is going away */
  }
}

/** Requests the worker never started: refused, safe to send again. */
function refuseInFlight(response: () => HostResponse): void {
  for (const [, { done }] of inFlight) done(response());
  inFlight.clear();
}

/* --- The class code, checked with the translation relay ------------------- */

let accessKnown: { required: boolean; at: number } | null = null;
const codeChecks = new Map<string, { ok: boolean; at: number }>();
const ACCESS_TTL_MS = 60000;

/** POST /api/mt/health. `code` null asks only whether one is needed. */
async function relayHealth(code: string | null): Promise<{ status: number; body: Json | null }> {
  try {
    const res = await fetch("/api/mt/health", {
      method: "POST",
      headers: code ? { "X-CT-Access": code } : {},
      cache: "no-store",
    });
    let body: Json | null = null;
    try {
      body = (await res.json()) as Json;
    } catch {
      /* not the relay's own answer */
    }
    return { status: res.status, body };
  } catch {
    return { status: 0, body: null };
  }
}

async function accessRequired(): Promise<boolean> {
  if (accessKnown && Date.now() - accessKnown.at < ACCESS_TTL_MS) return accessKnown.required;
  const r = await relayHealth(null);
  // Not known (no relay, offline): say a code is needed, so the gate asks
  // rather than letting a first translation be refused.
  const required = r.status === 200 ? Boolean(r.body?.access_required) : (accessKnown?.required ?? true);
  if (r.status === 200) accessKnown = { required, at: Date.now() };
  return required;
}

/** null when the code is fine (or none is needed); else the refusal. */
async function refuseBadCode(req: HostRequest): Promise<HostResponse | null> {
  if (!(await accessRequired())) return null;
  const code = header(req, "X-CT-Access");
  if (!code) return json(401, errorBody(ACCESS_REFUSED, { code: "access_required" }));
  const known = codeChecks.get(code);
  if (known && Date.now() - known.at < ACCESS_TTL_MS) {
    return known.ok ? null : json(401, errorBody(ACCESS_REFUSED, { code: "access_required" }));
  }
  const r = await relayHealth(code);
  if (r.status === 401) {
    codeChecks.set(code, { ok: false, at: Date.now() });
    return json(401, errorBody(ACCESS_REFUSED, { code: "access_required" }));
  }
  // Anything but a clear "no" lets the request on: the relay checks the
  // code again on every batch, and a refusal there fails the job plainly.
  if (r.status === 200) codeChecks.set(code, { ok: true, at: Date.now() });
  return null;
}

/* --- The lasting store ---------------------------------------------------------- */

async function openStore(): Promise<void> {
  try {
    const nav = navigator as Navigator & { storage?: StorageManager & { getDirectory?: () => Promise<unknown> } };
    if (!nav.storage?.getDirectory) throw new Error("this browser has no private storage for sites (OPFS)");
    const root = (await nav.storage.getDirectory()) as FileSystemDirectoryHandle;
    const dir = (await root.getDirectoryHandle("ct", { create: true })) as unknown as DirLike;
    const s = new OpfsStore(dir);
    // A write and a read, so a browser that lists the folder but will not
    // keep anything (full, or blocked by policy) is found out now.
    s.enqueue({ seq: 0, op: "put", key: "probe/ok", data: encoder.encode("1").buffer as ArrayBuffer });
    await s.flushed();
    if (s.stats.failed || (await s.readText("probe/ok")) !== "1") {
      throw new Error(s.stats.lastError || "the private storage would not keep a file");
    }
    s.onApplied = (entry, outcome) => {
      if (outcome !== "failed") return;
      stats.journalProblems += 1;
      stats.lastJournalProblem = s.stats.lastError;
      // The store has blocked the document; keep the good copy in memory.
      const id = documentOf(entry.key);
      if (id) markUnsafe(id);
    };
    store = s;
    for (const doc of await listDocuments()) indexedIds.add(doc.id);
  } catch (e) {
    store = null;
    storeProblem = String((e as Error)?.message ?? e).slice(0, 300);
  }
  publishStatus();
}

// Index writes, one at a time, so two changes to one document never race.
let indexChain: Promise<unknown> = Promise.resolve();
function inIndex<T>(fn: () => Promise<T>): Promise<T> {
  const next = indexChain.then(fn, fn);
  indexChain = next.catch(() => {});
  return next;
}

function indexDocument(id: string, fields: Partial<IndexedDocument>): Promise<void> {
  if (dropped.has(id)) return Promise.resolve();
  indexedIds.add(id);
  return inIndex(async () => {
    const current = await getDocument(id);
    await rememberDocument({
      filename: "",
      createdAt: Date.now(),
      engine: ENGINE_HASH,
      ...(current ?? {}),
      ...Object.fromEntries(Object.entries(fields).filter(([, v]) => v !== undefined)),
      id,
      stored: store !== null || Boolean(current?.stored),
      updatedAt: Date.now(),
    });
  });
}

/** The journal's numbers must follow on: a number skipped is a write that
 * never reached the queue, of a document this cannot know, so every
 * document in memory stops being trusted on disk until written out again. */
function checkSeq(seq: number): void {
  if (!Number.isFinite(seq) || seq <= 0) return;
  if (lastSeq && seq !== lastSeq + 1) {
    stats.journalGaps += 1;
    stats.lastJournalProblem = `journal gap: ${lastSeq} then ${seq}`;
    for (const id of hydrated.keys()) {
      store?.block(id);
      markUnsafe(id);
    }
  }
  lastSeq = seq;
}

/** The disk no longer holds this document's latest writes (opfs-store.ts
 * has blocked it, or a write never reached it). Its good copy is the one in
 * the worker's memory: kept there, the pages back it up as before, and it
 * is written out whole again soon. */
function markUnsafe(id: string): void {
  if (dropped.has(id) || unsafe.has(id)) return;
  unsafe.add(id);
  stats.unsafeDocuments = unsafe.size;
  // Never recycled away while it is the only good copy.
  worker?.postMessage({ type: "limit", recycleBytes: 0 });
  publishStatus();
  scheduleResync();
}

function scheduleResync(): void {
  if (resyncTimer || !unsafe.size) return;
  resyncTimer = setTimeout(() => {
    resyncTimer = null;
    void resyncUnsafe();
  }, resyncDelay);
  resyncDelay = Math.min(resyncDelay * 2, RESYNC_MOST_MS);
}

/** Make room (expired documents go first), then have the worker tell every
 * file of each unsafe document again. */
async function resyncUnsafe(): Promise<void> {
  if (!store || !unsafe.size) return;
  await sweep();
  if (!worker || status.state !== "ready" || leaving) {
    scheduleResync();
    return;
  }
  for (const id of unsafe) {
    if (hydrated.has(id)) worker.postMessage({ type: "resync", id });
  }
  // Checked again later: anything still unsafe is tried again.
  scheduleResync();
}

const resyncFailedAt = new Map<string, number>();

function onResync(msg: Json): void {
  const id = String(msg.id);
  if (!store || dropped.has(id)) return;
  if (msg.type === "resync-start") {
    store.unblock(id);
    void store.after(() => {
      resyncFailedAt.set(id, store!.stats.failed);
    });
  } else if (msg.type === "resync-file") {
    store.enqueue({ seq: 0, op: "put", key: String(msg.key), data: (msg.data as ArrayBuffer) ?? null });
  } else if (msg.type === "resync-end") {
    const keys = new Set((msg.keys as string[]) ?? []);
    void store.after(async () => {
      const s = store!;
      if (s.stats.failed !== resyncFailedAt.get(id) || s.blocked.has(id)) {
        stats.lastResync = { at: new Date().toISOString(), ok: false, error: s.stats.lastError };
        return;
      }
      await s.prune(id, keys);
      stats.resyncs += 1;
      stats.lastResync = { at: new Date().toISOString(), ok: true, files: keys.size };
      unsafe.delete(id);
      stats.unsafeDocuments = unsafe.size;
      if (!unsafe.size) {
        resyncDelay = RESYNC_FIRST_MS;
        worker?.postMessage({ type: "limit", recycleBytes: recycleBytes() });
      }
      publishStatus();
    });
  }
}

/** The worker that held the good copies is gone: what is on disk is all
 * there is, an older but whole copy (the store stopped at the first
 * failure), so the documents are trusted again from it. */
function forgetUnsafe(): void {
  for (const id of unsafe) store?.unblock(id);
  unsafe.clear();
  stats.unsafeDocuments = 0;
  resyncDelay = RESYNC_FIRST_MS;
}

/** One engine write, from the journal: queued for the disk in the order the
 * engine made it. Never waits. */
function onJournal(msg: Json): void {
  const entry: JournalEntry = {
    seq: Number(msg.seq),
    op: msg.op as JournalEntry["op"],
    key: String(msg.key),
    data: (msg.data as ArrayBuffer | null) ?? null,
  };
  checkSeq(entry.seq);
  const id = documentOf(entry.key);
  if (id && dropped.has(id)) return;
  if (!store) return;
  store.enqueue(entry);
  if (!id) return;
  // A document the index has not heard of (made by a request whose answer
  // has not arrived yet, or a project opened here): listed now, so
  // retention covers it whatever happens next.
  if (entry.op === "put" && entry.key === `documents/${id}/manifest.json` && !indexedIds.has(id)) {
    void indexDocument(id, {});
  }
  // The engine deleted a whole document itself: the rest of it goes too,
  // once the deletion is on disk.
  if (entry.op === "delete_prefix" && entry.key === `documents/${id}`) {
    void store.after(() => forgetEverywhere(id));
  }
}

/** The documents whose newest job, as kept on disk, the engine's resume
 * would pick up. */
async function unfinishedDocuments(): Promise<string[]> {
  if (!store) return [];
  const ids = await store.documentIds();
  const out: string[] = [];
  for (const id of ids) {
    if (dropped.has(id)) continue;
    try {
      if (isUnfinished(await store.latestJob(id))) out.push(id);
    } catch {
      /* unreadable: left for the page that opens it */
    }
  }
  return out;
}

type Stop = [string, string];

async function savedStops(): Promise<Stop[]> {
  const stops = (await getMeta<Stop[]>("stops").catch(() => undefined)) ?? [];
  return Array.isArray(stops) ? stops.filter((s) => Array.isArray(s) && s.length === 2) : [];
}

async function rememberStop(documentId: string, jobId: string): Promise<void> {
  const stops = (await savedStops()).filter(([, j]) => j !== jobId);
  await setMeta("stops", [...stops, [documentId, jobId]].slice(-50));
}

async function forgetStop(jobId: string): Promise<void> {
  const stops = await savedStops();
  if (stops.some(([, j]) => j === jobId)) await setMeta("stops", stops.filter(([, j]) => j !== jobId));
}

/** Put a document into the running worker's memory, from the disk.
 * `resume`: the engine then picks up its job if one was interrupted. */
async function hydrate(id: string, resume: boolean): Promise<"ok" | "missing" | "stale"> {
  if (!store || dropped.has(id)) return "missing";
  const w = worker;
  const s = store;
  // Read in the queue's order, so a deletion queued before it is seen.
  const files = await s.after(() => s.readDocument(id));
  if (dropped.has(id)) return "missing";
  if (!files.some(([k]) => k === `documents/${id}/manifest.json`)) return "missing";
  if (worker !== w || !w) return "stale";
  const stops = resume ? (await savedStops()).filter(([d]) => d === id) : [];
  if (worker !== w) return "stale";
  if (dropped.has(id)) return "missing";
  w.postMessage(
    { type: "hydrate", id, files, resume, stops, code: resume ? accessCode() : null },
    files.map(([, buf]) => buf),
  );
  hydrated.set(id, Date.now());
  stats.hydrations += 1;
  return "ok";
}

/** Whether a request for `id` can go to Python now; or why not (a refusal
 * when the engine cannot start at all). */
async function ensureDocument(
  id: string,
): Promise<"here" | "gone" | "unknown" | "expired" | "wait" | HostResponse> {
  if (hydrated.has(id) && !dropped.has(id)) {
    hydrated.set(id, Date.now());
    return "here";
  }
  if (expiredIds.includes(id)) return "expired";
  if (!store) return lostDocs.has(id) || (await wasHeldHere(id)) ? "gone" : "unknown";
  if (dropped.has(id) || !(await isIndexed(id))) return "unknown";
  if (!worker || status.state !== "ready") {
    return bootRefusal(false) ?? "wait";
  }
  let pending = hydrating.get(id);
  if (!pending) {
    pending = hydrate(id, true).finally(() => hydrating.delete(id));
    hydrating.set(id, pending);
  }
  const r = await pending;
  if (r === "ok") return "here";
  return r === "missing" ? "gone" : "wait";
}

/** Everything this browser keeps of a document, other than its files on
 * disk: its listing, its pictures, the copies the pages keep. */
async function forgetEverywhere(id: string): Promise<void> {
  hydrated.delete(id);
  jobs.delete(id);
  documents.delete(id);
  indexedIds.delete(id);
  if (unsafe.delete(id)) {
    stats.unsafeDocuments = unsafe.size;
    publishStatus();
  }
  await forgetDocument(id);
  await forgetImages(id);
  await forgetKeptCopies(id).catch(() => {});
  forgetRecent(id);
  announceDocs({ type: "deleted", id });
}

/** Delete a document from this computer: the worker's memory, the disk,
 * the index, the pictures, the pages' copies. Returns the files removed. */
async function purge(id: string, expired = false): Promise<number> {
  dropped.add(id);
  // A load from disk already under way finishes first (it sees `dropped`
  // and stops), so the document cannot come back into memory after this.
  await hydrating.get(id)?.catch(() => {});
  if (worker && hydrated.has(id)) worker.postMessage({ type: "evict", id });
  hydrated.delete(id);
  let files = 0;
  if (store) {
    const s = store;
    files = await s.after(async () => (await s.documentSize(id)).files);
    s.enqueue({ seq: 0, op: "delete_prefix", key: `documents/${id}`, data: null });
    await s.flushed();
  }
  if (expired) {
    expiredIds = [...expiredIds.filter((x) => x !== id), id].slice(-300);
    void setMeta("expired", expiredIds).catch(() => {});
  }
  await forgetEverywhere(id);
  // A picture that was on its way when this began is not kept either.
  await forgetImages(id);
  return files;
}

/* --- Retention ------------------------------------------------------------- */

let sweeping = false;

/** Delete every document uploaded more than the retention time ago, with
 * every copy of it, except the one being translated right now. Documents on
 * disk that the index lost are listed again (or deleted, if expired). */
async function sweep(): Promise<void> {
  if (sweeping) return;
  sweeping = true;
  const removed: string[] = [];
  try {
    const docs = await listDocuments();
    const listed = new Set(docs.map((d) => d.id));
    for (const id of store ? await store.documentIds() : []) {
      if (!store) break;
      if (listed.has(id) || dropped.has(id) || hydrated.has(id)) continue;
      const text = await store.readText(`documents/${id}/manifest.json`);
      let createdAt = NaN;
      let filename = "";
      try {
        const m = JSON.parse(text ?? "") as { created_at?: string; filename?: string };
        createdAt = Date.parse(m.created_at ?? "");
        filename = m.filename ?? "";
      } catch {
        /* no readable manifest: it never finished arriving */
      }
      if (!Number.isFinite(createdAt)) {
        // No readable manifest: an upload still arriving, or one a tab
        // never finished. Left alone until nothing in it has changed for
        // the retention time.
        const { changed } = await store.documentSize(id);
        if (!changed) continue;
        createdAt = changed;
      }
      docs.push({ id, filename, createdAt, engine: ENGINE_HASH, stored: true });
      await indexDocument(id, { filename, createdAt });
    }
    const now = Date.now();
    const marker = (id: string) => `/documents/${encodeURIComponent(id)}`;
    const busy = (id: string) =>
      runningJob?.documentId === id ||
      hydrating.has(id) ||
      unsafe.has(id) ||
      now - (hydrated.get(id) ?? 0) < SWEEP_SPARES_USED_MS ||
      [...inFlight.values()].some((r) => r.path.includes(marker(id)));
    for (const id of expiredDocuments(docs, now, LOCAL_RETENTION_MS, busy)) {
      await purge(id, true);
      removed.push(id);
    }
  } catch (e) {
    stats.lastSweep = { at: new Date().toISOString(), error: String((e as Error)?.message ?? e) };
    return;
  } finally {
    sweeping = false;
  }
  stats.lastSweep = { at: new Date().toISOString(), removed: removed.length };
}

/** Documents picked up again at start that the home page does not list:
 * one uploaded just before the browser closed (its list is saved a moment
 * after it changes, so a crash can lose it), or one a Drive batch was
 * working on. Listed, so she has a way back to it. */
async function listOnHomePage(ids: string[]): Promise<void> {
  try {
    const listed = new Set(parseRecent(recentSnapshot()).map((r) => r.id));
    for (const id of ids) {
      if (listed.has(id)) continue;
      const doc = await getDocument(id);
      if (!doc) continue;
      rememberRecent({
        id,
        filename: doc.filename || "Document",
        createdAt: new Date(doc.createdAt || Date.now()).toISOString(),
      });
    }
  } catch {
    /* the list is a convenience */
  }
}

/* --- Lasting storage ------------------------------------------------------------ */

/** After her first finished translation: ask the browser not to clear this
 * site's storage when space runs low. Chrome decides by itself (how much
 * the site is used, whether it is installed); the answer is recorded for
 * the diagnostic, never shown to her. */
async function askToKeepStorage(): Promise<void> {
  if (stats.persist) return;
  stats.persist = { asking: true };
  const storage = navigator.storage as StorageManager | undefined;
  let already: boolean | null = null;
  let granted: boolean | null = null;
  try {
    already = (await storage?.persisted?.()) ?? null;
    granted = already ? true : ((await storage?.persist?.()) ?? null);
  } catch {
    granted = null;
  }
  stats.persist = { askedAt: new Date().toISOString(), already, granted };
  await setMeta("persist", stats.persist).catch(() => {});
}

/* --- The worker ------------------------------------------------------------ */

function startWorker(): void {
  if (worker || recycling || bootAttempts >= MAX_BOOT_ATTEMPTS || leaving) return;
  bootAttempts += 1;
  stats.boots += 1;
  status.state = "booting";
  status.phase = "engine";
  status.loaded = 0;
  status.total = ENGINE_BOOT_BYTES;
  status.error = null;
  status.diagnostic = null;
  publishStatus();
  let w: Worker;
  try {
    w = new Worker(`${ENGINE_BASE}engine.worker.js`, { type: "module", name: "ct-engine" });
  } catch (e) {
    // Refused outright (a policy that blocks workers): a failed start.
    workerLost(`could not create the worker: ${String(e)}`, true);
    return;
  }
  worker = w;
  hydrated.clear();
  lastSeq = 0;
  w.onmessage = (ev) => {
    if (worker === w) onWorkerMessage(ev.data);
  };
  w.onerror = (ev) => {
    ev.preventDefault?.();
    if (worker !== w) return;
    // Before "ready" this is a start that failed: the script was blocked,
    // came back as a web page, or is gone after a deploy. It counts like
    // any other failed start, so the waiting upload hears why and the next
    // request does not wait on a worker that will never answer.
    workerLost(ev.message || "the worker stopped", status.state === "booting");
  };
  w.postMessage({ type: "boot", origin: location.origin, recycleBytes: unsafe.size ? 0 : recycleBytes() });
}

/** Start the engine if it is not running; a refusal when it cannot be. */
function bootRefusal(waitForBoot: boolean): HostResponse | null {
  // A fresh worker is about to start in place of the last one.
  if (recycling) return null;
  if (!worker && bootAttempts >= MAX_BOOT_ATTEMPTS) {
    // Three starts failed. Asked again by an upload (she is trying again),
    // or a minute later: one more round of starts. Otherwise the reason.
    if (!REMOTE_ENGINE_URL && (waitForBoot || Date.now() - lastBootFailure > BOOT_COOL_DOWN_MS)) {
      bootAttempts = 0;
    } else {
      return json(500, errorBody(status.error ?? BOOT_FAILED, { diagnostic: status.diagnostic }));
    }
  }
  startWorker();
  return null;
}

/** Forget what the worker held (its memory is gone or going). */
function resetWorkerState(): void {
  hydrated.clear();
  hydrating.clear();
  jobs.clear();
  documents.clear();
  runningJob = null;
  releaseJobGuards();
  status.busy = false;
  status.jobDocument = null;
}

/** The worker died (or would not start). With the lasting store nothing on
 * disk is lost: a fresh worker picks up where the disk says. */
function workerLost(reason: string, failedBoot = false): void {
  worker?.terminate();
  worker = null;
  if (!store) for (const id of hydrated.keys()) lostDocs.add(id);
  resetWorkerState();
  forgetUnsafe();
  status.restart = null;
  // A handover waiting for this worker to stop: it has.
  if (leaving) halted?.();
  // Started in Python and never answered: they may have been applied.
  refuseInFlight(() => json(502, errorBody(WORKER_LOST)));
  stopDocs.clear();
  status.diagnostic = reason || null;
  let refusal = WORKER_LOST;
  if (failedBoot) {
    lastBootFailure = Date.now();
    const giveUp = bootAttempts >= MAX_BOOT_ATTEMPTS;
    // The last try: this browser uses Render from now on, if there is one.
    const remote = giveUp && Boolean(REMOTE_ENGINE_URL);
    status.state = giveUp ? "failed" : "idle";
    status.error = giveUp ? (remote ? BOOT_FAILED_REMOTE : BOOT_FAILED) : null;
    refusal = remote ? BOOT_FAILED_REMOTE : BOOT_FAILED;
    if (remote) fallBackToRemoteEngine();
  } else {
    stats.crashes += 1;
    stats.lastCrash = (reason || "").slice(0, 300);
    status.state = "idle";
    status.error = null;
  }
  publishStatus();
  // Whoever waited for a start now hears why it did not happen. Not yet the
  // last try: the start is tried again for them, after a moment (a network
  // blip is often over by then).
  if (failedBoot && status.state === "idle") {
    if (waitingForBoot.length) setTimeout(startWorker, 2000);
  } else if (failedBoot) {
    for (const { done } of waitingForBoot.splice(0)) {
      done(json(500, errorBody(refusal, { diagnostic: status.diagnostic })));
    }
  }
  if (!store) {
    void caches.delete(IMG_CACHE).catch(() => {});
    if (!failedBoot) for (const { done } of waitingForBoot.splice(0)) done(json(500, errorBody(WORKER_LOST)));
    return;
  }
  // Kept on disk: a crashed worker is replaced at once when a job was left
  // unfinished (or an upload is waiting), so it carries on by itself.
  if (!failedBoot) void restartIfWorkWaits();
}

async function restartIfWorkWaits(): Promise<void> {
  if (!store || worker || leaving) return;
  await store.flushed();
  if (waitingForBoot.length || (await unfinishedDocuments()).length) startWorker();
}

/** A fresh worker in place of this one, between jobs: it had grown too big,
 * or ran out of memory. Everything it wrote is on disk once the queue is
 * flushed, so nothing is lost, and its unfinished jobs are resumed. */
async function replaceWorker(reason: Json): Promise<void> {
  const old = worker;
  if (!old) return;
  // Its journal is all queued already ("recycle" came after it), so it can
  // go now, before anything could start a second worker beside it.
  old.terminate();
  worker = null;
  recycling = true;
  stats.recycles += 1;
  stats.lastRecycle = { ...reason, at: new Date().toISOString() };
  status.state = "booting";
  status.restart = reason.reason === "memory-error" ? "memory" : "refresh";
  publishStatus();
  // The worker stopped between entries: nothing waiting reached Python.
  refuseInFlight(notYet);
  resetWorkerState();
  forgetUnsafe();
  // A handover waiting for this worker to stop: it has.
  if (leaving) halted?.();
  try {
    await store?.flushed();
  } finally {
    recycling = false;
  }
  bootAttempts = 0;
  if (!leaving) startWorker();
  else status.restart = null;
}

/** The worker is up: put back the documents whose jobs were interrupted,
 * have the engine resume them, then let requests in. */
async function afterReady(w: Worker, summary: Json): Promise<void> {
  stats.lastBootSeconds = Number(summary?.seconds) || null;
  stats.lastBootBytes = Number(summary?.bytes) || null;
  try {
    if (store) {
      const ids = await unfinishedDocuments();
      for (const id of ids) {
        if (worker !== w) return;
        const r = await hydrate(id, false);
        if (r === "stale") return;
      }
      if (worker !== w) return;
      if (ids.length) {
        w.postMessage({ type: "resume", ids: null, stops: await savedStops(), code: accessCode(), quiet: status.restart === "refresh" });
      }
    }
  } catch (e) {
    stats.lastJournalProblem = `resume: ${String((e as Error)?.message ?? e)}`.slice(0, 300);
  }
  if (worker !== w) return;
  status.state = "ready";
  status.phase = "engine";
  status.error = null;
  status.restart = null;
  bootAttempts = 0;
  publishStatus();
  if (unsafe.size) scheduleResync();
  for (const { req, done } of waitingForBoot.splice(0)) void toWorker(req).then(done);
}

function onWorkerMessage(msg: Json): void {
  switch (msg.type) {
    case "journal":
      onJournal(msg);
      break;
    case "journal-failed": {
      // A write the worker could not tell: the disk's copy of its document
      // would miss it, so it stops there (in order) until written out again.
      checkSeq(Number(msg.seq));
      stats.journalProblems += 1;
      stats.lastJournalProblem = `${String(msg.key)}: ${String(msg.message)}`.slice(0, 300);
      const id = documentOf(String(msg.key));
      if (id && store && !dropped.has(id)) {
        store.block(id);
        markUnsafe(id);
      }
      break;
    }
    case "resync-start":
    case "resync-file":
    case "resync-end":
      onResync(msg);
      break;
    case "progress": {
      status.phase = msg.phase as EngineStatus["phase"];
      status.total = Math.max(Number(msg.total) || 0, 1);
      status.loaded = Math.min(Number(msg.loaded) || 0, status.total);
      publishStatus();
      break;
    }
    case "ready":
      void afterReady(worker!, (msg.summary as Json) ?? {});
      break;
    case "boot-failed":
      workerLost(String(msg.message || ""), true);
      break;
    case "fatal":
      // Python can no longer run in it: a new worker takes over, from the
      // disk (or, with no lasting store, from the pages' backups).
      workerLost(String(msg.message || "Pyodide failed"));
      break;
    case "recycle":
      void replaceWorker({ reason: msg.reason, memory: msg.memory, limit: msg.limit });
      break;
    case "halted":
      halted?.();
      break;
    case "hydrated":
    case "resumed": {
      const resumed = Array.isArray(msg.resumed) ? (msg.resumed as string[]) : [];
      stats.resumedJobs += resumed.length;
      // Picked up only because the worker started afresh between jobs: not
      // a pause she should be told about.
      if (msg.quiet) {
        for (const id of resumed) quietResumes.set(id, (quietResumes.get(id) ?? 0) + 1);
        void setMeta("quietResumes", [...quietResumes].slice(-100)).catch(() => {});
      }
      break;
    }
    case "chrome-check": {
      // Before a job: can Chrome translate this language here, now?
      const w = worker;
      const code = String(msg.code);
      void chromeReady(code).then((ok) => w?.postMessage({ type: "chrome-status", code, ok }));
      break;
    }
    case "model-failed":
      // The job runs anyway; the chain's next engine translates its text.
      stats.lastModelProblem = `${String(msg.code)}: ${String(msg.message)}`.slice(0, 300);
      break;
    case "housekeeping-failed":
      stats.lastJournalProblem = `${String(msg.item)} ${String(msg.id ?? "")}: ${String(msg.message)}`.slice(0, 300);
      if (msg.item === "hydrate" && msg.id) hydrated.delete(String(msg.id));
      break;
    case "response": {
      const done = inFlight.get(String(msg.id))?.done;
      inFlight.delete(String(msg.id));
      done?.({
        status: Number(msg.status),
        headers: (msg.headers as [string, string][]) ?? [],
        body: (msg.body as ArrayBuffer) ?? null,
      });
      break;
    }
    case "dropped": {
      const done = inFlight.get(String(msg.id))?.done;
      inFlight.delete(String(msg.id));
      // Its page stopped listening; the Service Worker still needs an end.
      done?.(notYet());
      break;
    }
    case "snapshot":
      takeSnapshot(String(msg.raw));
      break;
    case "job":
      if (msg.state === "start") {
        runningJob = {
          documentId: String(msg.documentId),
          jobId: String(msg.jobId),
          translation: Boolean(msg.translation),
        };
        // Stopped before it existed: it ends at its first checkpoint.
        if (stopDocs.delete(runningJob.documentId)) stopRequested.add(runningJob.jobId);
        holdWhileJobRuns();
      } else {
        if (msg.error) jobEndedWithoutSaying(String(msg.documentId), String(msg.jobId));
        stopRequested.delete(String(msg.jobId));
        void forgetStop(String(msg.jobId));
        if (msg.stopChecks && typeof msg.stopChecks === "object") {
          stats.lastStopChecks = { ...(msg.stopChecks as Json), job: String(msg.jobId) };
        }
        if (msg.memory && typeof msg.memory === "object") {
          const m = msg.memory as { heap?: number; tess?: number; mt?: number; memfs?: number };
          const total = (m.heap ?? 0) + (m.tess ?? 0) + (m.mt ?? 0) + (m.memfs ?? 0);
          stats.lastMemory = { ...m, total, at: new Date().toISOString() };
          stats.peakMemory = Math.max(stats.peakMemory, total);
        }
        const finished = jobs.get(String(msg.documentId));
        if (msg.translation && finished?.stage === "COMPLETE") void askToKeepStorage();
        runningJob = null;
        releaseJobGuards();
        void letWaitingServiceWorkerIn();
      }
      status.busy = runningJob !== null;
      status.jobDocument = runningJob?.documentId ?? null;
      if ((status.phase === "reader" || status.phase === "model") && msg.state === "start") status.phase = "engine";
      publishStatus();
      break;
  }
}

/** A job that ended (Python raised past the engine's own guards) while its
 * last saved state still says it is working: said as the failure it is, so
 * the progress bar does not sit part-way for ever. */
function jobEndedWithoutSaying(documentId: string, jobId: string): void {
  const job = jobs.get(documentId);
  if (!job || job.job_id !== jobId || job.is_terminal) return;
  jobs.set(documentId, {
    ...job,
    stage: "FAILED",
    stage_label: "Could not be completed",
    is_terminal: true,
    error: JOB_LOST,
    updated_at: new Date().toISOString(),
  });
}

/* The engine's own words were written for a server ("on this server"); here
 * they are about this computer. Only its messages are changed, never a
 * document's text. */
function computerWords(text: string): string {
  return text
    .replace(/\bthe translation server\b/gi, (m) => (m[0] === "T" ? "The" : "the") + " translator on this computer")
    .replace(/\bthe service restarted\b/gi, (m) => (m[0] === "T" ? "The" : "the") + " translator on this computer had to start again")
    .replace(/\b(this|the) server\b/gi, (m) => (m[0] === m[0].toUpperCase() ? "This" : "this") + " computer");
}

function localJob(job: Json): Json {
  const out: Json = { ...job };
  if (typeof out.error === "string") out.error = computerWords(out.error);
  if (Array.isArray(out.page_errors)) {
    out.page_errors = (out.page_errors as Json[]).map((e) =>
      typeof e?.message === "string" ? { ...e, message: computerWords(e.message) } : e,
    );
  }
  return out;
}

/** An answer from Python, with an error's or a job's words made local. */
function localWordsIn(res: HostResponse, isJob: boolean): HostResponse {
  const type = res.headers.find(([k]) => k.toLowerCase() === "content-type")?.[1] ?? "";
  if (!res.body || !type.includes("json") || (res.status < 400 && !isJob)) return res;
  try {
    const body = JSON.parse(new TextDecoder().decode(res.body)) as Json;
    let out: Json = isJob && res.status < 400 ? localJob(body) : { ...body };
    for (const k of ["error", "detail"]) {
      if (typeof out[k] === "string") out = { ...out, [k]: computerWords(out[k] as string) };
    }
    const bytes = encoder.encode(JSON.stringify(out));
    return {
      status: res.status,
      headers: res.headers.filter(([k]) => k.toLowerCase() !== "content-length"),
      body: bytes.buffer as ArrayBuffer,
    };
  } catch {
    return res;
  }
}

// How many times each job has been picked up again (the engine's
// resume_count), from its snapshots.
const resumeCounts = new Map<string, number>();

/** The document's newest job as the page sees it: with "resumed" when it
 * was interrupted (a tab closed, the computer slept, memory ran short) and
 * picked up again; a fresh start between jobs does not count. */
function jobOut(id: string): Json | null {
  const job = jobs.get(id);
  if (!job) return null;
  const jobId = String(job.job_id);
  const resumed = (resumeCounts.get(jobId) ?? 0) > (quietResumes.get(jobId) ?? 0);
  return resumed ? { ...job, resumed: true } : job;
}

function takeSnapshot(raw: string): void {
  let message: Json;
  try {
    message = JSON.parse(raw) as Json;
  } catch {
    return;
  }
  const id = String(message.document_id ?? "");
  if (!id || dropped.has(id)) return;
  if (!hydrated.has(id)) hydrated.set(id, Date.now());
  if (message.type === "job") {
    const next = message.job as Json;
    const current = jobs.get(id);
    // The newest job is the document's job, as the engine's route reads it.
    if (
      !current ||
      current.job_id === next.job_id ||
      String(next.created_at) >= String(current.created_at)
    ) {
      resumeCounts.set(String(next.job_id), Number(message.resume_count) || 0);
      jobs.set(id, localJob(next));
      if (store && (!current || current.stage !== next.stage)) {
        void indexDocument(id, {
          job: { id: String(next.job_id), stage: String(next.stage), terminal: Boolean(next.is_terminal) },
        });
      }
    }
  } else if (message.type === "document") {
    documents.set(id, message.document as Json);
  }
}

function toWorker(req: HostRequest): Promise<HostResponse> {
  if (!worker || status.state !== "ready" || leaving) return Promise.resolve(notYet());
  if (abortedIds.delete(req.id)) return Promise.resolve(notYet());
  return new Promise((resolve) => {
    inFlight.set(req.id, { done: resolve, method: req.method.toUpperCase(), path: req.path });
    const transfer = req.body ? [req.body] : [];
    worker!.postMessage(
      {
        type: "request",
        id: req.id,
        method: req.method,
        path: req.path,
        headers: req.headers,
        body: req.body,
      },
      transfer,
    );
  });
}

/** To Python: at once when the engine is ready. Otherwise the engine is
 * started, and an upload waits for it (it has no timeout, and the page
 * shows the download meanwhile); anything else is told to ask again. */
async function forward(req: HostRequest, waitForBoot = false): Promise<HostResponse> {
  if (leaving) return notYet();
  if (status.state === "ready" && worker) return toWorker(req);
  const refused = bootRefusal(waitForBoot);
  if (refused) return refused;
  if (!waitForBoot) return notYet();
  return new Promise((done) => waitingForBoot.push({ req, done }));
}

/* --- Memory: documents nobody is looking at leave the worker ------------------ */

function evictIdle(): void {
  if (!store || !worker || status.state !== "ready" || leaving) return;
  const now = Date.now();
  for (const [id, used] of [...hydrated]) {
    if (now - used < EVICT_IDLE_MS) continue;
    if (runningJob?.documentId === id || hydrating.has(id) || unsafe.has(id)) continue;
    const job = jobs.get(id);
    if (job && !job.is_terminal && job.stage !== "REVIEW_READY") continue;
    const marker = `/documents/${encodeURIComponent(id)}`;
    if ([...inFlight.values()].some((r) => r.path.includes(marker))) continue;
    worker.postMessage({ type: "evict", id });
    hydrated.delete(id);
    jobs.delete(id);
    documents.delete(id);
    stats.evictions += 1;
  }
}

/* --- While a job runs ------------------------------------------------------ */

let wakeLock: { release(): Promise<void> } | null = null;
// Other tabs of the app open in this browser (they wait for the host's
// lock, and one of them takes over if this tab closes), counted every few
// seconds.
let otherTabs = 0;

/** Closing or reloading this tab now would lose writes not yet on disk
 * (they take a moment), or changes the disk refused (kept only in memory
 * until written out again). With no lasting store, it would also lose a
 * translation, or the documents another tab is showing. The browser asks
 * her first; otherwise it simply closes. A translation in progress is not a
 * reason by itself: it is on disk, and carries on when she opens the app
 * again (or at once, in another tab). */
function onBeforeUnload(e: BeforeUnloadEvent): void {
  if (leaving) return;
  const saving = store !== null && (store.pending > 0 || unsafe.size > 0);
  const memoryOnly = store === null && (runningJob !== null || (hydrated.size > 0 && otherTabs > 0));
  if (!saving && !memoryOnly) return;
  e.preventDefault();
  e.returnValue = "";
}

async function countOtherTabs(): Promise<void> {
  try {
    const locks = (navigator as Navigator & { locks?: LockManager }).locks;
    const q = await locks?.query();
    otherTabs = q?.pending?.filter((l) => l.name === ENGINE_LOCK).length ?? 0;
  } catch {
    otherTabs = 0;
  }
}

async function keepAwake(): Promise<void> {
  if (!runningJob || wakeLock || document.visibilityState !== "visible") return;
  try {
    const nav = navigator as Navigator & {
      wakeLock?: { request(type: "screen"): Promise<{ release(): Promise<void> }> };
    };
    wakeLock = (await nav.wakeLock?.request("screen")) ?? null;
  } catch {
    wakeLock = null; // refused (battery saver, policy): the job still runs
  }
}

function onVisibility(): void {
  if (document.visibilityState === "visible") void keepAwake();
  else wakeLock = null; // the browser releases it when the tab hides
}

function holdWhileJobRuns(): void {
  document.addEventListener("visibilitychange", onVisibility);
  void keepAwake();
}

function releaseJobGuards(): void {
  document.removeEventListener("visibilitychange", onVisibility);
  const held = wakeLock;
  wakeLock = null;
  void held?.release().catch(() => {});
}

/** A newer Service Worker waits while a job runs (switching mid-job could
 * change how requests are routed); it may take over once the job ends. */
async function letWaitingServiceWorkerIn(): Promise<void> {
  try {
    const reg = await navigator.serviceWorker.getRegistration("/");
    reg?.waiting?.postMessage({ type: "ct-idle" });
  } catch {
    /* no registration: nothing waits */
  }
}

/* --- A newer build in another tab ------------------------------------------------ */

/** Whether now is a quiet moment to hand the engine over: no job running,
 * nothing she is in the middle of on this page (English typed and not yet
 * sent, a change being saved, a Drive batch), nothing kept only in memory,
 * and the worker not starting afresh. */
function quietMoment(): boolean {
  return (
    runningJob === null &&
    !pageIsHeld() &&
    unsafe.size === 0 &&
    !recycling &&
    hydrating.size === 0 &&
    status.state !== "booting"
  );
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A tab opened on a newer build of the site: at the next quiet moment
 * (never part-way through a job, never while she is typing or a batch
 * runs), this tab makes sure everything is on disk, lets go of the engine,
 * and loads the page again, so the newer tab runs the engine. Jobs still
 * waiting are picked up there, from the disk. If she starts something in
 * the moment it takes, it carries on here and the handover waits. */
async function handOver(): Promise<void> {
  if (handOverWanted || !store) return;
  handOverWanted = true;
  for (;;) {
    while (!quietMoment()) {
      stats.handOverWaits += 1;
      await sleep(HANDOVER_LOOK_MS);
    }
    leaving = true;
    const w = worker;
    if (w && status.state === "ready") {
      // Finishes what it is doing (a job that started just now included)
      // and runs nothing more. A worker that stops or is replaced instead
      // ends the wait too (workerLost, replaceWorker).
      await new Promise<void>((resolve) => {
        halted = resolve;
        w.postMessage({ type: "halt" });
      });
      halted = null;
    }
    await store.flushed();
    if (!pageIsHeld() && unsafe.size === 0) break;
    // She started something meanwhile: carry on here, and wait again.
    leaving = false;
    if (worker === w && w) w.postMessage({ type: "unhalt" });
    else if (!worker) void restartIfWorkWaits();
  }
  refuseInFlight(notYet);
  worker?.terminate();
  worker = null;
  await store.flushed();
  for (const { done } of waitingForBoot.splice(0)) done(notYet());
  status.state = "idle";
  publishStatus();
  releaseLock?.();
  location.reload();
}

/* --- Pictures --------------------------------------------------------------- */

const IMG_CACHE = "ct-img";
// Page renders that never change for a document, and pictures whose address
// carries the page's edit count (?v=), which changes whenever they would.
const IMAGE_ROUTE =
  /^\/api\/documents\/[^/]+\/pages\/\d+\/(preview|thumb|translated|translated-thumb|plate|blocks\/[^/]+\/sprite)$/;

function imageCacheKey(req: HostRequest): string | null {
  if (req.method !== "GET") return null;
  const url = new URL(req.path, "http://engine.local");
  const m = url.pathname.match(IMAGE_ROUTE);
  if (!m) return null;
  const fixed = m[1] === "preview" || m[1] === "thumb";
  if (!fixed && !url.searchParams.has("v")) return null;
  // The code and the retry counter change nothing about the picture.
  url.searchParams.delete("code");
  url.searchParams.delete("retry");
  url.searchParams.sort();
  return `${location.origin}/engine${url.pathname}${url.search}`;
}

async function cachedImage(key: string, documentId: string): Promise<HostResponse | null> {
  if (dropped.has(documentId)) return null;
  try {
    const hit = await (await caches.open(IMG_CACHE)).match(key);
    if (!hit) return null;
    return {
      status: 200,
      headers: [...hit.headers.entries()] as [string, string][],
      body: await hit.arrayBuffer(),
    };
  } catch {
    return null;
  }
}

async function keepImage(key: string, res: HostResponse, documentId: string): Promise<void> {
  const type = res.headers.find(([k]) => k.toLowerCase() === "content-type")?.[1] ?? "";
  if (res.status !== 200 || !res.body || !type.startsWith("image/")) return;
  // Deleted (or expired) while it was being drawn: not kept.
  if (dropped.has(documentId)) return;
  try {
    await (await caches.open(IMG_CACHE)).put(
      key,
      new Response(res.body.slice(0), { status: 200, headers: res.headers }),
    );
  } catch {
    /* storage full or blocked: the picture is drawn again next time */
  }
}

/** On taking over: pictures an earlier host kept may show writes that never
 * reached the disk before it closed (they take a moment). A page's own
 * renders never change, and a block's picture is named by what it shows,
 * so those stay; a picture named by the page's edit count stays only if
 * the disk has reached that count (a lost edit would otherwise come back
 * under the same name). Without the disk, none stay. */
async function keepTrustworthyImages(): Promise<void> {
  try {
    if (!store) {
      await caches.delete(IMG_CACHE);
      return;
    }
    const cache = await caches.open(IMG_CACHE);
    const revisions = new Map<string, Map<number, number> | null>();
    const diskRevisions = async (id: string) => {
      if (!revisions.has(id)) {
        let pages: Map<number, number> | null = null;
        try {
          const m = JSON.parse((await store!.readText(`documents/${id}/manifest.json`)) ?? "null") as {
            pages?: { page_number?: number; number?: number; revision?: number }[];
          } | null;
          if (m?.pages) {
            pages = new Map(
              m.pages.map((p, i) => [Number(p.page_number ?? p.number ?? i + 1), Number(p.revision ?? 0)]),
            );
          }
        } catch {
          pages = null;
        }
        revisions.set(id, pages);
      }
      return revisions.get(id) ?? null;
    };
    const prefix = `${location.origin}/engine/api/documents/`;
    for (const req of await cache.keys()) {
      const url = new URL(req.url);
      const m = req.url.startsWith(prefix)
        ? url.pathname.match(/^\/engine\/api\/documents\/([^/]+)\/pages\/(\d+)\/(preview|thumb|translated|translated-thumb|plate|blocks\/[^/]+\/sprite)$/)
        : null;
      let keep = false;
      if (m) {
        const kind = m[3];
        if (kind === "preview" || kind === "thumb" || kind.startsWith("blocks/")) {
          keep = (await diskRevisions(decodeURIComponent(m[1]))) !== null;
        } else {
          const pages = await diskRevisions(decodeURIComponent(m[1]));
          const v = Number(url.searchParams.get("v"));
          const onDisk = pages?.get(Number(m[2]));
          keep = onDisk !== undefined && Number.isFinite(v) && v <= onDisk;
        }
      }
      if (!keep) await cache.delete(req);
    }
  } catch {
    // Unsure: start afresh, as before.
    await caches.delete(IMG_CACHE).catch(() => {});
  }
}

async function forgetImages(documentId: string): Promise<void> {
  try {
    const cache = await caches.open(IMG_CACHE);
    const prefix = `${location.origin}/engine/api/documents/${documentId}/`;
    for (const req of await cache.keys()) {
      if (req.url.startsWith(prefix)) await cache.delete(req);
    }
  } catch {
    /* nothing kept */
  }
}

/* --- Answering a request -------------------------------------------------- */

const DOCUMENT_PATH = /^\/api\/documents\/([^/?]+)(\/[^?]*)?(\?.*)?$/;

function healthAnswer(): Promise<HostResponse> {
  return accessRequired().then((required) =>
    json(200, {
      ...ENGINE_HEALTH,
      retention_hours: LOCAL_RETENTION_HOURS,
      access_required: required,
      // What this browser's engine is doing; api.ts reads it in local mode.
      engine: { ...status, stored: store !== null, hash: ENGINE_HASH, remote: Boolean(REMOTE_ENGINE_URL) },
    }),
  );
}

// The engine's 'chrome' provider, from the running job (chrome-mt.ts says
// why it comes here): {code, texts} -> {texts}, or 503 when Chrome cannot,
// which the provider reads as "translate with the next engine".
const CHROME_MT_PATH = "/api/__mt/chrome";

async function chromeAnswer(req: HostRequest): Promise<HostResponse> {
  try {
    const body = JSON.parse(new TextDecoder().decode(req.body ?? new ArrayBuffer(0))) as { code?: unknown; texts?: unknown };
    if (typeof body.code !== "string" || !Array.isArray(body.texts) || !body.texts.every((t) => typeof t === "string")) {
      return json(400, { error: "bad request" });
    }
    return json(200, { texts: await chromeTranslate(body.code, body.texts as string[]) });
  } catch (e) {
    stats.lastModelProblem = `chrome: ${String((e as Error)?.message ?? e)}`.slice(0, 300);
    return json(503, { error: "Chrome could not translate this" });
  }
}

async function answer(req: HostRequest): Promise<HostResponse> {
  const path = req.path.split("?")[0];
  const method = req.method.toUpperCase();

  if (method === "GET" && path === "/api/health") return healthAnswer();
  if (method === "GET" && path.startsWith(STOP_PATH)) {
    const jobId = decodeURIComponent(path.slice(STOP_PATH.length));
    return {
      status: 200,
      headers: [["content-type", "text/plain"], ["cache-control", "no-store"]],
      body: encoder.encode(stopRequested.has(jobId) ? "1" : "0").buffer as ArrayBuffer,
    };
  }
  if (method === "POST" && path === CHROME_MT_PATH) return chromeAnswer(req);
  if (method === "GET" && path === "/api/languages") {
    return (await refuseBadCode(req)) ?? json(200, ENGINE_LANGUAGES);
  }
  if (method === "GET" && path === "/api/output-modes") return json(200, ENGINE_OUTPUT_MODES);

  // Everything below may need the documents on disk. (While this tab hands
  // over to a newer one, the answers made here still come; what needs
  // Python is refused, to be asked again of the next host.)
  await storeOpened;

  if (method === "POST" && (path === "/api/documents" || path === "/api/projects")) {
    const res = await forward(req, true);
    if (res.status === 200 && res.body) {
      try {
        const body = JSON.parse(new TextDecoder().decode(res.body)) as Json;
        const id = String(body.document_id ?? "");
        if (id) {
          hydrated.set(id, Date.now());
          // Recorded before the page hears of it, so its very next request
          // for this document is routed here.
          await indexDocument(id, {
            filename: String(body.filename ?? ""),
            createdAt: Date.now(),
            engine: ENGINE_HASH,
          });
        }
      } catch {
        /* not the engine's answer: passed on as it is */
      }
    }
    return res;
  }

  if (method === "POST" && path === "/api/translate-texts") {
    return (await refuseBadCode(req)) ?? forward(req);
  }

  const doc = path.match(DOCUMENT_PATH);
  if (doc) {
    const id = decodeURIComponent(doc[1]);
    const rest = doc[2] ?? "";
    if (store && method === "DELETE" && rest === "") {
      // Being translated, or waiting to be: deleting it under the job would
      // leave the job working on nothing.
      const job = jobs.get(id);
      const queued = job && !job.is_terminal && job.stage !== "REVIEW_READY";
      if (runningJob?.documentId === id || queued) return json(409, errorBody(STILL_TRANSLATING));
      const removed = await purge(id);
      return json(200, { deleted: removed > 0, objects_removed: removed });
    }
    if (!store && method === "DELETE" && rest === "" && !hydrated.has(id) && (await isIndexed(id))) {
      // Kept nowhere but in this browser's lists and backups (the worker
      // that held it is gone): those go.
      dropped.add(id);
      await forgetEverywhere(id);
      return json(200, { deleted: true, objects_removed: 0 });
    }
    const where = await ensureDocument(id);
    if (typeof where === "object") return where;
    if (where === "wait") return notYet();
    if (where !== "here") {
      // Not in this engine: lost to a reload with nothing kept (said as the
      // restart it is, so the page puts it back from its backup), removed
      // because its time was up, or never here at all.
      return json(404, errorBody(where === "gone" ? DOCUMENT_GONE : where === "expired" ? EXPIRED : NOT_FOUND));
    }
    if (method === "GET" && rest === "/job") {
      // Always from the snapshot when there is one: Python may be about to
      // start a job (its "start" is still on its way here), and a poll sent
      // to it then would wait for the whole job. A job that ends without
      // saying so is marked failed when it ends (jobEndedWithoutSaying),
      // and a worker that dies takes its snapshots with it (workerLost), so
      // a snapshot never says "working" for ever.
      const job = jobOut(id);
      if (job) return json(200, job);
    }
    if (method === "GET" && rest === "" && runningJob) {
      const snapshot = documents.get(id);
      if (snapshot) return json(200, { ...snapshot, job: jobOut(id) });
    }
    if (method === "POST" && rest === "/cancel" && runningJob) {
      // Python is inside a job and cannot hear it until the job ends: the
      // stop is answered here, as the engine would answer it.
      const job = jobs.get(id);
      if (runningJob.documentId === id) {
        // The running job reads this at its next page (ct_browser/cancel.py),
        // and a tab that takes over before then still stops it.
        stopRequested.add(runningJob.jobId);
        void rememberStop(id, runningJob.jobId);
        return json(200, { cancelled: true });
      }
      if (job && !job.is_terminal) {
        // Waiting behind another document's job. Python drops it before
        // it can start (requests go before the next job); the mark stops
        // it at its first page should it start anyway.
        stopRequested.add(String(job.job_id));
        void rememberStop(id, String(job.job_id));
        void toWorker({ ...req, id: `${req.id}-stop` });
        return json(200, { cancelled: true });
      }
      // Her "Translate" may itself still be waiting for Python: the job it
      // makes is stopped the moment it starts.
      const translatePath = `/api/documents/${encodeURIComponent(id)}/translate`;
      const waiting = [...inFlight.values()].some(
        (r) => r.method === "POST" && r.path.split("?")[0] === translatePath,
      );
      if (waiting) stopDocs.add(id);
      return json(200, { cancelled: waiting });
    }
    if (method === "POST" && rest === "/translate") {
      const refused = await refuseBadCode(req);
      if (refused) return refused;
      // Asked again after a stop: she wants it after all.
      stopDocs.delete(id);
    }
    const key = imageCacheKey(req);
    if (key) {
      const hit = await cachedImage(key, id);
      if (hit) return hit;
      const res = await forward(req);
      await keepImage(key, res, id);
      return res;
    }
    const res = localWordsIn(await forward(req), method === "POST" && rest === "/translate");
    if (method === "DELETE" && rest === "" && res.status === 200) {
      // Only without a lasting store (with one, deleting is done above):
      // every copy this browser keeps goes too.
      dropped.add(id);
      await forgetEverywhere(id);
    }
    if (res.status === 404 && rest === "") void forgetImages(id);
    return res;
  }

  return localWordsIn(await forward(req), false);
}

/** Uploaded in this browser before (the index outlives the worker). */
function wasHeldHere(id: string): Promise<boolean> {
  return isIndexed(id);
}

/* --- Diagnostics ------------------------------------------------------------------ */

async function diagnostics(): Promise<Json> {
  const storage = navigator.storage as StorageManager | undefined;
  const estimate = await storage?.estimate?.().catch(() => null);
  const persisted = await storage?.persisted?.().catch(() => null);
  const docs = await listDocuments().catch(() => [] as IndexedDocument[]);
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    at: new Date().toISOString(),
    build: ENGINE_HASH,
    pageLoadedAt: new Date(SINCE).toISOString(),
    hostSince: stats.becameHostAt ? new Date(stats.becameHostAt).toISOString() : null,
    browser: navigator.userAgent,
    deviceMemoryGB: nav.deviceMemory ?? null,
    engine: {
      state: status.state,
      phase: status.phase,
      busy: status.busy,
      error: status.error,
      reason: status.diagnostic,
      boots: stats.boots,
      lastBootSeconds: stats.lastBootSeconds,
      lastBootBytes: stats.lastBootBytes,
      crashes: stats.crashes,
      lastCrash: stats.lastCrash,
      recycles: stats.recycles,
      lastRecycle: stats.lastRecycle,
      recycleLimitBytes: recycleBytes(),
      lastMemory: stats.lastMemory,
      peakMemoryBytes: stats.peakMemory,
      resumedJobs: stats.resumedJobs,
      stopChecksLastJob: stats.lastStopChecks,
      handOverWanted,
      handOverWaits: stats.handOverWaits,
      documentsInMemory: hydrated.size,
      hydrations: stats.hydrations,
      evictions: stats.evictions,
      otherTabs,
    },
    storage: {
      kept: store !== null,
      problem: storeProblem,
      usageBytes: estimate?.usage ?? null,
      quotaBytes: estimate?.quota ?? null,
      persisted,
      persistAsked: stats.persist,
      documents: docs.length,
      documentsWithUnfinishedJobs: docs.filter((d) => d.job && !d.job.terminal && d.job.stage !== "REVIEW_READY").length,
      journal: store ? { ...store.stats, waiting: store.pending } : null,
      journalProblems: stats.journalProblems,
      lastJournalProblem: stats.lastJournalProblem,
      journalGaps: stats.journalGaps,
      notSavedOnDisk: [...unsafe],
      blocked: store ? [...store.blocked] : [],
      rewrites: stats.resyncs,
      lastRewrite: stats.lastResync,
    },
    retention: { hours: LOCAL_RETENTION_HOURS, lastSweep: stats.lastSweep },
  };
}

/* --- Wiring ---------------------------------------------------------------------- */

function onServiceWorkerMessage(ev: MessageEvent): void {
  const data = ev.data as { type?: string } | null;
  const port = ev.ports[0];
  if (!data || !port) return;
  if (data.type === "ct-ping") {
    port.postMessage({ host: isHost && !leaving, busy: status.busy, hash: ENGINE_HASH });
    return;
  }
  if (data.type !== "ct-request") return;
  if (!isHost) {
    // Asked by a Service Worker that has not heard of the handover yet.
    port.postMessage(notYet());
    return;
  }
  const req = data as HostRequest;
  void answer(req)
    .catch(() => json(500, errorBody(WORKER_LOST)))
    .then((res) => {
      port.postMessage(res, res.body ? [res.body] : []);
    });
}

// The hosting tab (by when its page was loaded) already asked to hand
// over to this newer build; another older host is asked again.
let askedHost: number | null = null;

function onChannelMessage(ev: MessageEvent<ChannelMessage>): void {
  const msg = ev.data;
  if (!msg) return;
  if (!isHost) {
    // The host runs an older build than this page: ask it to hand over
    // once its job is done.
    if (
      msg.type === "status" &&
      msg.status.hash &&
      msg.status.hash !== ENGINE_HASH &&
      (msg.status.since ?? 0) < SINCE &&
      askedHost !== (msg.status.since ?? 0)
    ) {
      askedHost = msg.status.since ?? 0;
      channel?.postMessage({ type: "newer", hash: ENGINE_HASH, since: SINCE } satisfies ChannelMessage);
    }
    return;
  }
  if (msg.type === "status?") {
    publishStatus();
  } else if (msg.type === "abort") {
    // Its page gave up on it. Dropped if Python has not started it, so a
    // change that timed out is never applied later as a duplicate.
    if (inFlight.has(msg.id)) worker?.postMessage({ type: "abort", id: msg.id });
    else abortedIds.add(msg.id);
    if (abortedIds.size > 200) abortedIds.clear();
  } else if (msg.type === "newer") {
    if (msg.hash !== ENGINE_HASH && msg.since > SINCE) void handOver();
  } else if (msg.type === "diagnostics?") {
    void diagnostics().then((data) =>
      channel?.postMessage({ type: "diagnostics", id: msg.id, data } satisfies ChannelMessage),
    );
  }
}

/** The id the browser holds this tab's engine lock under (not the tab's
 * client id): the Service Worker watches it to know this tab still hosts. */
async function ownLockId(): Promise<string | null> {
  try {
    const locks = (navigator as Navigator & { locks?: LockManager }).locks;
    const q = await locks?.query();
    return q?.held?.find((l) => l.name === ENGINE_LOCK)?.clientId ?? null;
  } catch {
    return null;
  }
}

async function announce(): Promise<void> {
  try {
    const reg = await navigator.serviceWorker.ready;
    reg.active?.postMessage({ type: "ct-host", hash: ENGINE_HASH, lock: await ownLockId() });
  } catch {
    /* no Service Worker: requests cannot reach any host anyway */
  }
}

function becomeHost(shared = true): void {
  isHost = true;
  stats.becameHostAt = Date.now();
  window.addEventListener("beforeunload", onBeforeUnload);
  setInterval(() => void countOtherTabs(), 4000);
  setInterval(evictIdle, EVICT_CHECK_MS);
  setInterval(() => void sweep(), SWEEP_MS);
  // Without Web Locks every tab hosts its own engine: none of them may use
  // the disk every tab shares (they would write over each other, and run
  // each other's jobs twice). Documents then live in the worker only, and
  // the pages keep backups, as before documents were kept.
  storeOpened = (shared ? openStore() : Promise.resolve().then(() => {
    storeProblem = "this browser cannot tell its tabs apart (no Web Locks), so nothing is kept on disk";
    publishStatus();
  })).then(keepTrustworthyImages);
  void storeOpened.then(async () => {
    stats.persist = (await getMeta<Json>("persist").catch(() => undefined)) ?? null;
    const expired = await getMeta<string[]>("expired").catch(() => undefined);
    if (Array.isArray(expired)) expiredIds = expired.filter((x) => typeof x === "string");
    const quiet = await getMeta<[string, number][]>("quietResumes").catch(() => undefined);
    if (Array.isArray(quiet)) for (const [k, v] of quiet) quietResumes.set(String(k), Number(v) || 0);
    // A translation that was interrupted carries on now, without waiting
    // for a page to ask.
    const unfinished = store ? await unfinishedDocuments() : [];
    if (unfinished.length) {
      await listOnHomePage(unfinished);
      startWorker();
    }
    await sweep();
  });
  void announce();
  navigator.serviceWorker.addEventListener("controllerchange", () => void announce());
  publishStatus();
}

/* --- Every tab: the Service Worker ------------------------------------------- */

const RELOADED_KEY = "ct.engine.reloaded";

/** Register the Service Worker, and make sure it answers this page.
 *
 * On a first visit it takes the page over as soon as it is active
 * (clients.claim). A page loaded with Shift+Reload is never handed to it by
 * the browser: it is asked to claim the page, and if that does not work the
 * page is loaded once more. A browser where none of this works uses the
 * Render engine instead. */
async function connectServiceWorker(): Promise<void> {
  const url =
    `/engine-sw.js?v=${encodeURIComponent(ENGINE_HASH)}` +
    `&remote=${encodeURIComponent(REMOTE_ENGINE_URL)}`;
  let reg: ServiceWorkerRegistration;
  try {
    reg = await navigator.serviceWorker.register(url, { scope: "/", updateViaCache: "none" });
  } catch {
    // A network blip or a filter blocking the new build's address: the
    // Service Worker already answering this page keeps doing so (it is
    // fetched again next time). Only a page with none uses Render.
    if (!navigator.serviceWorker.controller) fallBackToRemoteEngine();
    return;
  }
  if (navigator.serviceWorker.controller) {
    try {
      sessionStorage.removeItem(RELOADED_KEY);
    } catch {
      /* storage blocked */
    }
    return;
  }
  const active = await navigator.serviceWorker.ready;
  if (navigator.serviceWorker.controller) return;
  const controlled = new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => resolve(false), 2500);
    navigator.serviceWorker.addEventListener(
      "controllerchange",
      () => {
        clearTimeout(timer);
        resolve(true);
      },
      { once: true },
    );
  });
  active.active?.postMessage({ type: "ct-claim" });
  void reg.update().catch(() => {});
  if (await controlled) return;
  let reloaded = false;
  try {
    reloaded = sessionStorage.getItem(RELOADED_KEY) === "1";
    if (!reloaded) sessionStorage.setItem(RELOADED_KEY, "1");
  } catch {
    reloaded = true; // cannot remember a reload: do not risk a loop
  }
  if (!reloaded) {
    location.reload();
    return;
  }
  fallBackToRemoteEngine();
}

let started = false;

/** Called once per tab, in local mode only (EngineHost.tsx). */
export function startEngineHost(): void {
  if (started || typeof window === "undefined") return;
  started = true;
  if (!("serviceWorker" in navigator) || typeof Worker === "undefined") {
    fallBackToRemoteEngine();
    return;
  }
  void connectServiceWorker();
  // For a developer's console (and the checks run over DevTools).
  (window as Window & { ctEngineDiagnostics?: typeof engineDiagnostics }).ctEngineDiagnostics =
    engineDiagnostics;
  channel = new BroadcastChannel(ENGINE_CHANNEL);
  channel.addEventListener("message", onChannelMessage);
  // The host's status, heard here too: an idle host says nothing until
  // asked, and this tab may run a different build than it (after a deploy,
  // or a rollback), which only its status shows.
  channel.postMessage({ type: "status?" } satisfies ChannelMessage);
  navigator.serviceWorker.addEventListener("message", onServiceWorkerMessage);
  navigator.serviceWorker.startMessages();

  const locks = (navigator as Navigator & { locks?: LockManager }).locks;
  if (locks) {
    // Held until this tab closes (or hands the engine to a newer build).
    void locks.request(ENGINE_LOCK, () => {
      becomeHost();
      return new Promise<void>((resolve) => {
        releaseLock = resolve;
      });
    });
  } else {
    becomeHost(false); // no Web Locks: every tab hosts its own, in memory only
  }
}

/** "Copy details for support", from any tab: what the hosting tab knows. */
export function engineDiagnostics(timeoutMs = 3000): Promise<Json | null> {
  // This tab's own build beside the host's: after a deploy they can differ.
  const thisTab = { build: ENGINE_HASH, pageLoadedAt: new Date(SINCE).toISOString(), hosting: isHost };
  return hostDiagnostics(timeoutMs).then((d) => (d ? { ...d, thisTab } : null));
}

function hostDiagnostics(timeoutMs: number): Promise<Json | null> {
  if (isHost) return diagnostics();
  if (!channel) return Promise.resolve(null);
  const id = Math.random().toString(36).slice(2);
  return new Promise((resolve) => {
    const listen = (ev: MessageEvent<ChannelMessage>) => {
      if (ev.data?.type === "diagnostics" && ev.data.id === id) {
        channel?.removeEventListener("message", listen);
        clearTimeout(timer);
        resolve(ev.data.data);
      }
    };
    const timer = setTimeout(() => {
      channel?.removeEventListener("message", listen);
      resolve(null);
    }, timeoutMs);
    channel!.addEventListener("message", listen);
    channel!.postMessage({ type: "diagnostics?", id } satisfies ChannelMessage);
  });
}

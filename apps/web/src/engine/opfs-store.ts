/**
 * The lasting copy of every document made in this browser, in OPFS (the
 * browser's private disk for this site).
 *
 * The engine (Python, in the worker) works on its own store in memory. Each
 * write it makes there is told to the host in order (ct_browser/journal.py:
 * a sequence number, "put" with the file's bytes, "delete", or
 * "delete_prefix"), and this store applies them STRICTLY first in, first out,
 * one at a time:
 *   put            written under a temporary name beside the key, then moved
 *                  over it, so a key is always either its old bytes or its
 *                  new ones, never half of either;
 *   delete         the key removed;
 *   delete_prefix  the key's folder removed (a document deleted or expired).
 * Whatever has been applied when the tab dies is therefore a prefix of the
 * engine's own writes, which is exactly what a server's disk holds after its
 * process is killed: the engine opens it and carries on as after a restart.
 *
 * A write the disk refuses (full, or blocked) would break that: a later
 * manifest could land pointing at the file that failed. So the first failure
 * for a document BLOCKS it: its later entries are skipped (counted), and
 * what is on disk stays a prefix of the engine's writes, only an older one.
 * The host is told, keeps the document in the worker's memory, and later
 * writes the whole document out again (unblock, then every file), or the
 * document's deletion clears the block.
 *
 * Keys are the engine's ("documents/<id>/pages/0001/source.png"); a key
 * always starts with a letter or digit, so the temporary names (starting
 * with ".") never collide with one, and are left out of every listing.
 *
 * Written against a small subset of the File System Access API, so the node
 * tests (opfs-store.test.mjs) can run it over an in-memory stand-in.
 */

export interface WritableLike {
  write(data: ArrayBuffer | Uint8Array): Promise<void>;
  close(): Promise<void>;
  abort?(): Promise<void>;
}

export interface FileLike {
  readonly kind: "file";
  getFile(): Promise<{
    size: number;
    lastModified?: number;
    arrayBuffer(): Promise<ArrayBuffer>;
    text(): Promise<string>;
  }>;
  createWritable(): Promise<WritableLike>;
  move(parent: DirLike, name: string): Promise<void>;
}

export interface DirLike {
  readonly kind: "directory";
  getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<DirLike>;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<FileLike>;
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
  entries(): AsyncIterable<[string, FileLike | DirLike]>;
}

export type JournalOp = "put" | "delete" | "delete_prefix";

export interface JournalEntry {
  seq: number;
  op: JournalOp;
  key: string;
  data?: ArrayBuffer | null;
}

type Task =
  | { kind: "entry"; entry: JournalEntry }
  | { kind: "run"; fn: () => Promise<void> };

export interface StoreStats {
  /** Journal entries written to disk. */
  applied: number;
  /** Bytes written to disk. */
  bytes: number;
  /** Entries that could not be applied (disk full, say). */
  failed: number;
  /** Entries skipped because their document was blocked by a failure. */
  skipped: number;
  lastError: string | null;
  /** The most entries ever waiting at once. */
  mostWaiting: number;
}

const DOCS = "documents";

function isNotFound(e: unknown): boolean {
  return (e as { name?: string } | null)?.name === "NotFoundError";
}

/** Split "a/b/c" into its folders and its name. */
function splitKey(key: string): { dirs: string[]; name: string } {
  const parts = key.split("/").filter(Boolean);
  const name = parts.pop() ?? "";
  return { dirs: parts, name };
}

export class OpfsStore {
  readonly root: DirLike;
  readonly stats: StoreStats = { applied: 0, bytes: 0, failed: 0, skipped: 0, lastError: null, mostWaiting: 0 };
  /** Documents whose writes stopped at their first failure (see above). */
  readonly blocked = new Set<string>();
  private queue: Task[] = [];
  private draining = false;
  private idle: (() => void)[] = [];
  private dirs = new Map<string, DirLike>();
  private tmpCounter = 0;
  /** Called after each entry is applied, fails or is skipped, in order. */
  onApplied: ((entry: JournalEntry, outcome: "ok" | "failed" | "skipped") => void) | null = null;

  constructor(root: DirLike) {
    this.root = root;
  }

  /** Entries not yet on disk. */
  get pending(): number {
    return this.queue.length + (this.draining ? 1 : 0);
  }

  /** Queue an engine write. Never waits: the order of calls is the order on
   * disk. */
  enqueue(entry: JournalEntry): void {
    this.queue.push({ kind: "entry", entry });
    this.stats.mostWaiting = Math.max(this.stats.mostWaiting, this.queue.length);
    void this.drain();
  }

  /** Run `fn` once everything queued before it is on disk, and before
   * anything queued after it (reads that must see, or not see, a write). */
  after<T>(fn: () => Promise<T> | T): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({
        kind: "run",
        fn: async () => {
          try {
            resolve(await fn());
          } catch (e) {
            reject(e);
          }
        },
      });
      void this.drain();
    });
  }

  /** From this point in the queue, skip the document's entries (a write
   * for it was lost before reaching the queue). */
  block(id: string): void {
    void this.after(() => {
      this.blocked.add(id);
    });
  }

  /** From this point in the queue, apply the document's entries again (its
   * whole document is about to be written out afresh). */
  unblock(id: string): void {
    void this.after(() => {
      this.blocked.delete(id);
    });
  }

  /** Resolves once everything queued so far is on disk. */
  flushed(): Promise<void> {
    if (!this.draining && this.queue.length === 0) return Promise.resolve();
    return new Promise((resolve) => this.idle.push(resolve));
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      for (;;) {
        const task = this.queue.shift();
        if (!task) break;
        if (task.kind === "run") {
          await task.fn();
          continue;
        }
        const entry = task.entry;
        const id = documentOf(entry.key);
        const wholeDocument = entry.op === "delete_prefix" && id !== null && entry.key.replace(/\/+$/, "") === `${DOCS}/${id}`;
        let outcome: "ok" | "failed" | "skipped" = "ok";
        if (id && this.blocked.has(id) && !wholeDocument) {
          outcome = "skipped";
          this.stats.skipped += 1;
        } else {
          try {
            await this.apply(entry);
            this.stats.applied += 1;
            this.stats.bytes += entry.data?.byteLength ?? 0;
            // The document is gone: nothing of it is left to be out of step.
            if (wholeDocument) this.blocked.delete(id);
          } catch (e) {
            outcome = "failed";
            this.stats.failed += 1;
            this.stats.lastError = `${entry.op} ${entry.key}: ${String((e as Error)?.message ?? e)}`.slice(0, 300);
            if (id) this.blocked.add(id);
          }
        }
        try {
          this.onApplied?.(entry, outcome);
        } catch {
          /* a listener never stops the queue */
        }
      }
    } finally {
      this.draining = false;
      const waiting = this.idle.splice(0);
      waiting.forEach((w) => w());
    }
  }

  /* --- Applying one entry ------------------------------------------------- */

  private async dir(dirs: string[], create: boolean): Promise<DirLike | null> {
    let path = "";
    let at = this.root;
    for (const name of dirs) {
      path = path ? `${path}/${name}` : name;
      const known = this.dirs.get(path);
      if (known) {
        at = known;
        continue;
      }
      try {
        at = await at.getDirectoryHandle(name, { create });
      } catch (e) {
        if (!create && (isNotFound(e) || (e as { name?: string })?.name === "TypeMismatchError")) return null;
        throw e;
      }
      this.dirs.set(path, at);
    }
    return at;
  }

  private forgetDirs(prefix: string): void {
    for (const k of [...this.dirs.keys()]) {
      if (k === prefix || k.startsWith(prefix + "/")) this.dirs.delete(k);
    }
  }

  private async apply(entry: JournalEntry): Promise<void> {
    const { dirs, name } = splitKey(entry.key);
    if (!name || name.startsWith(".")) throw new Error("not an engine key");
    if (entry.op === "put") {
      const parent = await this.dir(dirs, true);
      if (!parent) throw new Error("no folder");
      const tmpName = `.${name}.${++this.tmpCounter}.tmp`;
      const tmp = await parent.getFileHandle(tmpName, { create: true });
      const out = await tmp.createWritable();
      try {
        await out.write(entry.data ?? new ArrayBuffer(0));
        await out.close();
      } catch (e) {
        await out.abort?.().catch(() => {});
        await parent.removeEntry(tmpName).catch(() => {});
        throw e;
      }
      // Moves over an existing key in one step (Chrome replaces it).
      await tmp.move(parent, name);
      return;
    }
    const parent = await this.dir(dirs, false);
    if (!parent) return; // nothing there to remove
    try {
      await parent.removeEntry(name, { recursive: entry.op === "delete_prefix" });
    } catch (e) {
      if (!isNotFound(e)) throw e;
    }
    if (entry.op === "delete_prefix") this.forgetDirs(entry.key.replace(/\/+$/, ""));
  }

  /* --- Reading (after flushed(), so reads see every write) ----------------- */

  /** Every file of a document, as [key, bytes]; empty if it has none. Left-
   * over temporary files (a tab that died mid-write) are removed. */
  async readDocument(id: string): Promise<[string, ArrayBuffer][]> {
    const base = await this.dir([DOCS, id], false);
    if (!base) return [];
    const out: [string, ArrayBuffer][] = [];
    const walk = async (dir: DirLike, prefix: string): Promise<void> => {
      const stale: string[] = [];
      for await (const [name, handle] of dir.entries()) {
        if (name.startsWith(".")) {
          stale.push(name);
          continue;
        }
        if (handle.kind === "directory") await walk(handle, `${prefix}/${name}`);
        else out.push([`${prefix}/${name}`, await (await handle.getFile()).arrayBuffer()]);
      }
      for (const name of stale) await dir.removeEntry(name, { recursive: true }).catch(() => {});
    };
    await walk(base, `${DOCS}/${id}`);
    out.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return out;
  }

  /** Remove every file of a document that is not in `keep` (after the whole
   * document was written out again); returns how many went. */
  async prune(id: string, keep: Set<string>): Promise<number> {
    const base = await this.dir([DOCS, id], false);
    if (!base) return 0;
    let removed = 0;
    const walk = async (dir: DirLike, prefix: string): Promise<void> => {
      const gone: string[] = [];
      for await (const [name, handle] of dir.entries()) {
        const key = `${prefix}/${name}`;
        if (handle.kind === "directory") await walk(handle, key);
        else if (name.startsWith(".") || !keep.has(key)) gone.push(name);
      }
      for (const name of gone) {
        await dir.removeEntry(name).catch(() => {});
        removed += 1;
      }
    };
    await walk(base, `${DOCS}/${id}`);
    return removed;
  }

  /** The text of one key, or null. */
  async readText(key: string): Promise<string | null> {
    const { dirs, name } = splitKey(key);
    const parent = await this.dir(dirs, false);
    if (!parent) return null;
    try {
      return await (await (await parent.getFileHandle(name)).getFile()).text();
    } catch (e) {
      if (isNotFound(e) || (e as { name?: string })?.name === "TypeMismatchError") return null;
      throw e;
    }
  }

  /** The names in a folder, or [] if it does not exist. */
  async list(dirKey: string): Promise<{ name: string; kind: "file" | "directory" }[]> {
    const parent = await this.dir(dirKey.split("/").filter(Boolean), false);
    if (!parent) return [];
    const out: { name: string; kind: "file" | "directory" }[] = [];
    for await (const [name, handle] of parent.entries()) {
      if (!name.startsWith(".")) out.push({ name, kind: handle.kind });
    }
    return out;
  }

  /** The documents that have a folder here. */
  async documentIds(): Promise<string[]> {
    return (await this.list(DOCS)).filter((e) => e.kind === "directory").map((e) => e.name);
  }

  /** Bytes a document takes, how many files, and when one last changed. */
  async documentSize(id: string): Promise<{ bytes: number; files: number; changed: number }> {
    const base = await this.dir([DOCS, id], false);
    let bytes = 0;
    let files = 0;
    let changed = 0;
    if (!base) return { bytes, files, changed };
    const walk = async (dir: DirLike): Promise<void> => {
      for await (const [name, handle] of dir.entries()) {
        if (name.startsWith(".")) continue;
        if (handle.kind === "directory") await walk(handle);
        else {
          const file = await handle.getFile();
          bytes += file.size;
          files += 1;
          changed = Math.max(changed, file.lastModified ?? 0);
        }
      }
    };
    await walk(base);
    return { bytes, files, changed };
  }

  /** The document's newest job, read from its files: what the engine will
   * find when it opens the document. */
  async latestJob(id: string): Promise<{ job_id: string; stage: string; created_at: string } | null> {
    const names = (await this.list(`${DOCS}/${id}/jobs`)).filter(
      (e) => e.kind === "file" && e.name.endsWith(".json"),
    );
    let latest: { job_id: string; stage: string; created_at: string } | null = null;
    for (const { name } of names) {
      const text = await this.readText(`${DOCS}/${id}/jobs/${name}`);
      if (!text) continue;
      try {
        const job = JSON.parse(text) as { job_id: string; stage: string; created_at: string };
        if (!latest || String(job.created_at) >= String(latest.created_at)) latest = job;
      } catch {
        /* the engine skips a job it cannot read, and so does this */
      }
    }
    return latest;
  }
}

/** Stages after which nothing more happens to a job by itself. */
export const TERMINAL_STAGES = ["COMPLETE", "FAILED", "CANCELLED"];

/** A job the engine's resume would pick up: not finished, and not waiting
 * for her to choose a language (runner.resume_interrupted's rule). */
export function isUnfinished(job: { stage: string } | null): boolean {
  return Boolean(job && !TERMINAL_STAGES.includes(job.stage) && job.stage !== "REVIEW_READY");
}

/** The documents whose time is up: uploaded more than `retentionMs` ago, and
 * not the one a job is working on right now. */
export function expiredDocuments(
  docs: { id: string; createdAt: number }[],
  now: number,
  retentionMs: number,
  busy: (id: string) => boolean,
): string[] {
  return docs
    .filter((d) => Number.isFinite(d.createdAt) && now - d.createdAt >= retentionMs && !busy(d.id))
    .map((d) => d.id);
}

/** The document a key belongs to ("documents/<id>/..."), or null. */
export function documentOf(key: string): string | null {
  const m = /^documents\/([^/]+)(?:\/|$)/.exec(key);
  return m ? m[1] : null;
}

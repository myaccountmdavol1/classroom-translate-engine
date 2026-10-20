// Tests for the OPFS store's journal queue (opfs-store.ts), over an in-memory
// stand-in for the browser's private disk. Run from apps/web:
//
//   node --test src/engine/opfs-store.test.mjs
//
// Node runs the .ts module directly (type stripping).

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  OpfsStore,
  documentOf,
  expiredDocuments,
  isUnfinished,
} from "./opfs-store.ts";

/* --- An in-memory File System Access stand-in ----------------------------- */

const err = (name) => Object.assign(new Error(name), { name });

class FakeFile {
  kind = "file";
  constructor(parent, name, data = new Uint8Array()) {
    this.parent = parent;
    this.name = name;
    this.data = data;
  }
  async getFile() {
    const data = this.data;
    return {
      size: data.byteLength,
      arrayBuffer: async () => data.slice().buffer,
      text: async () => new TextDecoder().decode(data),
    };
  }
  async createWritable() {
    const parts = [];
    const fs = this.parent.fs;
    return {
      write: async (d) => {
        if (fs.failWrites) throw err("QuotaExceededError");
        parts.push(new Uint8Array(d instanceof ArrayBuffer ? d : d.buffer));
      },
      close: async () => {
        const n = parts.reduce((a, p) => a + p.length, 0);
        const out = new Uint8Array(n);
        let o = 0;
        for (const p of parts) { out.set(p, o); o += p.length; }
        this.data = out;
      },
      abort: async () => {},
    };
  }
  async move(parent, name) {
    this.parent.children.delete(this.name);
    parent.children.set(name, this); // replaces what was there
    this.parent = parent;
    this.name = name;
  }
}

class FakeDir {
  kind = "directory";
  constructor(fs) {
    this.fs = fs;
    this.children = new Map();
  }
  async getDirectoryHandle(name, opts = {}) {
    const got = this.children.get(name);
    if (got) {
      if (got.kind !== "directory") throw err("TypeMismatchError");
      return got;
    }
    if (!opts.create) throw err("NotFoundError");
    const d = new FakeDir(this.fs);
    this.children.set(name, d);
    return d;
  }
  async getFileHandle(name, opts = {}) {
    const got = this.children.get(name);
    if (got) {
      if (got.kind !== "file") throw err("TypeMismatchError");
      return got;
    }
    if (!opts.create) throw err("NotFoundError");
    const f = new FakeFile(this, name);
    this.children.set(name, f);
    return f;
  }
  async removeEntry(name, opts = {}) {
    const got = this.children.get(name);
    if (!got) throw err("NotFoundError");
    if (got.kind === "directory" && got.children.size && !opts.recursive) throw err("InvalidModificationError");
    this.children.delete(name);
  }
  async *entries() {
    for (const [k, v] of [...this.children]) yield [k, v];
  }
}

function fakeFs() {
  const fs = { failWrites: false };
  fs.root = new FakeDir(fs);
  return fs;
}

/** Every file under the root, as key -> text, temporary names included. */
function snapshot(dir, prefix = "", out = {}) {
  for (const [name, h] of dir.children) {
    const key = prefix ? `${prefix}/${name}` : name;
    if (h.kind === "directory") snapshot(h, key, out);
    else out[key] = new TextDecoder().decode(h.data);
  }
  return out;
}

const visible = (snap) => Object.fromEntries(Object.entries(snap).filter(([k]) => !k.split("/").some((p) => p.startsWith("."))));

const enc = (s) => new TextEncoder().encode(s).buffer;
let seq = 0;
const put = (key, text) => ({ seq: ++seq, op: "put", key, data: enc(text) });
const del = (key) => ({ seq: ++seq, op: "delete", key, data: null });
const delPrefix = (key) => ({ seq: ++seq, op: "delete_prefix", key, data: null });

/** What applying `entries` in order must leave, worked out plainly. */
function expected(entries) {
  const out = {};
  for (const e of entries) {
    if (e.op === "put") out[e.key] = new TextDecoder().decode(new Uint8Array(e.data));
    else if (e.op === "delete") delete out[e.key];
    else for (const k of Object.keys(out)) if (k === e.key || k.startsWith(e.key + "/")) delete out[k];
  }
  return out;
}

const JOURNAL = () => [
  put("documents/a/source.pdf", "pdf-a"),
  put("documents/a/manifest.json", "{m1}"),
  put("documents/a/jobs/job_1.json", '{"job_id":"job_1","stage":"QUEUED","created_at":"2026-10-20T10:00:00"}'),
  put("documents/a/pages/0001/source.png", "png1"),
  put("documents/a/manifest.json", "{m2}"),
  put("documents/b/source.pdf", "pdf-b"),
  put("documents/a/jobs/job_1.cancel", "1"),
  del("documents/a/jobs/job_1.cancel"),
  put("documents/a/jobs/job_1.json", '{"job_id":"job_1","stage":"TRANSLATING","created_at":"2026-10-20T10:00:00"}'),
  delPrefix("documents/b"),
  put("documents/b/source.pdf", "pdf-b2"),
  put("documents/a/manifest.json", "{m3}"),
];

describe("the journal queue", () => {
  test("applies entries strictly in order and leaves exactly their result", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    const entries = JOURNAL();
    for (const e of entries) store.enqueue(e);
    assert.ok(store.pending > 0);
    await store.flushed();
    assert.equal(store.pending, 0);
    assert.deepEqual(snapshot(fs.root), expected(entries));
    assert.equal(store.stats.applied, entries.length);
    assert.equal(store.stats.failed, 0);
  });

  test("whatever has been applied when the tab dies is a prefix of the journal", async () => {
    const entries = JOURNAL();
    for (let k = 0; k <= entries.length; k++) {
      const fs = fakeFs();
      const store = new OpfsStore(fs.root);
      let done = 0;
      let atDeath = null;
      store.onApplied = () => {
        done += 1;
        if (done === k) atDeath = snapshot(fs.root);
      };
      for (const e of entries) store.enqueue(e);
      await store.flushed();
      const seen = k === 0 ? {} : atDeath;
      // No half-written key is ever visible, and no temporary file is left
      // between entries.
      assert.deepEqual(seen, visible(seen), `temporary file left after ${k}`);
      assert.deepEqual(seen, expected(entries.slice(0, k)), `after ${k} entries`);
    }
  });

  test("a key mid-write still shows its old bytes until the move", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    store.enqueue(put("documents/a/manifest.json", "old"));
    await store.flushed();
    const original = FakeFile.prototype.move;
    let during = null;
    FakeFile.prototype.move = async function (parent, name) {
      during = visible(snapshot(fs.root));
      return original.call(this, parent, name);
    };
    try {
      store.enqueue(put("documents/a/manifest.json", "new"));
      await store.flushed();
    } finally {
      FakeFile.prototype.move = original;
    }
    assert.deepEqual(during, { "documents/a/manifest.json": "old" });
    assert.deepEqual(snapshot(fs.root), { "documents/a/manifest.json": "new" });
  });

  test("a write that fails blocks its document, so the disk stays a prefix", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    const outcomes = [];
    store.onApplied = (e, outcome) => outcomes.push([e.key, outcome]);
    store.enqueue(put("documents/a/one", "1"));
    await store.flushed();
    fs.failWrites = true;
    store.enqueue(put("documents/a/two", "2"));
    await store.flushed();
    fs.failWrites = false;
    // Later writes for "a" would point at the lost one: skipped. "b" is fine.
    store.enqueue(put("documents/a/manifest.json", "{refers to two}"));
    store.enqueue(put("documents/b/one", "b1"));
    await store.flushed();
    assert.equal(store.stats.failed, 1);
    assert.equal(store.stats.skipped, 1);
    assert.ok(store.blocked.has("a"));
    assert.match(store.stats.lastError, /documents\/a\/two/);
    assert.deepEqual(snapshot(fs.root), { "documents/a/one": "1", "documents/b/one": "b1" });
    assert.deepEqual(outcomes.map((o) => o[1]), ["ok", "failed", "skipped", "ok"]);
  });

  test("a blocked document is written out whole again after unblock, and pruned", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    store.enqueue(put("documents/a/old", "stale"));
    store.enqueue(put("documents/a/manifest.json", "{m1}"));
    store.block("a");
    store.enqueue(put("documents/a/manifest.json", "{skipped}"));
    store.unblock("a");
    for (const e of [put("documents/a/one", "1"), put("documents/a/manifest.json", "{m2}")]) store.enqueue(e);
    const removed = await store.after(() => store.prune("a", new Set(["documents/a/one", "documents/a/manifest.json"])));
    await store.flushed();
    assert.equal(removed, 1);
    assert.equal(store.blocked.size, 0);
    assert.deepEqual(snapshot(fs.root), { "documents/a/one": "1", "documents/a/manifest.json": "{m2}" });
  });

  test("deleting a blocked document applies, and clears the block", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    store.enqueue(put("documents/a/one", "1"));
    store.block("a");
    store.enqueue(delPrefix("documents/a"));
    store.enqueue(put("documents/a/again", "2"));
    await store.flushed();
    assert.equal(store.blocked.size, 0);
    assert.deepEqual(snapshot(fs.root), { "documents/a/again": "2" });
  });

  test("after() runs between the entries queued before and after it", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    store.enqueue(put("documents/a/one", "1"));
    let seen = null;
    const p = store.after(() => {
      seen = snapshot(fs.root);
    });
    store.enqueue(put("documents/a/two", "2"));
    await p;
    await store.flushed();
    assert.deepEqual(seen, { "documents/a/one": "1" });
  });
});

describe("reading documents back", () => {
  test("readDocument returns every file in key order and clears leftovers", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    for (const e of JOURNAL()) store.enqueue(e);
    await store.flushed();
    // A tab that died between writing a temporary file and moving it.
    const dir = await (await (await fs.root.getDirectoryHandle("documents")).getDirectoryHandle("a")).getDirectoryHandle("pages");
    (await dir.getFileHandle(".x.9.tmp", { create: true })).data = new Uint8Array([1]);
    const files = await store.readDocument("a");
    assert.deepEqual(files.map(([k]) => k), [
      "documents/a/jobs/job_1.json",
      "documents/a/manifest.json",
      "documents/a/pages/0001/source.png",
      "documents/a/source.pdf",
    ]);
    assert.equal(new TextDecoder().decode(files[1][1]), "{m3}");
    assert.equal(dir.children.has(".x.9.tmp"), false);
    assert.deepEqual(await store.readDocument("nope"), []);
    assert.deepEqual((await store.documentIds()).sort(), ["a", "b"]);
    assert.deepEqual(await store.documentSize("b"), { bytes: 6, files: 1, changed: 0 });
  });

  test("latestJob reads the newest job the engine will find", async () => {
    const fs = fakeFs();
    const store = new OpfsStore(fs.root);
    store.enqueue(put("documents/a/jobs/job_1.json", '{"job_id":"job_1","stage":"REVIEW_READY","created_at":"2026-10-20T10:00:00"}'));
    store.enqueue(put("documents/a/jobs/job_2.json", '{"job_id":"job_2","stage":"TRANSLATING","created_at":"2026-10-20T10:05:00"}'));
    store.enqueue(put("documents/a/jobs/job_2.cancel", "1"));
    await store.flushed();
    const job = await store.latestJob("a");
    assert.equal(job.job_id, "job_2");
    assert.equal(isUnfinished(job), true);
    assert.equal(await store.latestJob("b"), null);
  });
});

describe("rules", () => {
  test("isUnfinished follows the engine's resume rule", () => {
    for (const stage of ["QUEUED", "ANALYZING", "OCR", "TRANSLATING", "RECONSTRUCTING", "VERIFYING", "EXPORTING"]) {
      assert.equal(isUnfinished({ stage }), true, stage);
    }
    for (const stage of ["REVIEW_READY", "COMPLETE", "FAILED", "CANCELLED"]) {
      assert.equal(isUnfinished({ stage }), false, stage);
    }
    assert.equal(isUnfinished(null), false);
  });

  test("expiredDocuments keeps the busy one and the young ones", () => {
    const now = 100 * 3600e3;
    const day = 24 * 3600e3;
    const docs = [
      { id: "old", createdAt: now - day - 1 },
      { id: "edge", createdAt: now - day },
      { id: "young", createdAt: now - day + 60e3 },
      { id: "busy", createdAt: now - 2 * day },
      { id: "broken", createdAt: NaN },
    ];
    assert.deepEqual(expiredDocuments(docs, now, day, (id) => id === "busy"), ["old", "edge"]);
  });

  test("documentOf", () => {
    assert.equal(documentOf("documents/abc/pages/0001/x.png"), "abc");
    assert.equal(documentOf("documents/abc"), "abc");
    assert.equal(documentOf("other/abc"), null);
  });
});

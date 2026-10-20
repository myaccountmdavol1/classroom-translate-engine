/**
 * The documents this browser holds, in IndexedDB.
 *
 * Written by the engine host: when an upload is accepted, as each document's
 * job moves on, and when a document is deleted or expires. Read by the
 * Service Worker, which sends requests for a document listed here to this
 * browser's engine and everything else to Render, and by the host itself
 * (retention, and the documents to pick up again at start).
 *
 * The documents' files live in OPFS (opfs-store.ts); this is their index.
 * Every call is guarded: storage a browser refuses means an empty list.
 *
 * The "meta" store keeps a few host settings beside it (the stops she asked
 * for, whether lasting storage was asked for).
 */

import { INDEX_DB, INDEX_STORE, META_STORE } from "./protocol";

export interface IndexedDocument {
  id: string;
  filename: string;
  /** When it was uploaded here (ms); retention counts from this. */
  createdAt: number;
  /** The engine build that made it. */
  engine: string;
  /** Its files are kept in this browser's storage (OPFS), so it outlives
   * the tab. Absent for a document made before that (Phases 2-3). */
  stored?: boolean;
  /** Its newest job, as the engine last saved it. */
  job?: { id: string; stage: string; terminal: boolean } | null;
  /** When anything about it last changed here (ms). */
  updatedAt?: number;
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise<IDBDatabase | null>((resolve) => {
    try {
      const req = indexedDB.open(INDEX_DB, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(INDEX_STORE)) db.createObjectStore(INDEX_STORE, { keyPath: "id" });
        if (!db.objectStoreNames.contains(META_STORE)) db.createObjectStore(META_STORE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  }).then((db) => {
    if (!db) dbPromise = null; // asked again next time
    return db;
  });
  return dbPromise;
}

function run<T = undefined>(
  storeName: string,
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T> | void,
): Promise<{ ok: boolean; value: T | undefined }> {
  return openDb().then(
    (db) =>
      new Promise((resolve) => {
        if (!db) return resolve({ ok: false, value: undefined });
        try {
          const tx = db.transaction(storeName, mode);
          let value: T | undefined;
          const req = fn(tx.objectStore(storeName));
          if (req) req.onsuccess = () => (value = req.result);
          tx.oncomplete = () => resolve({ ok: true, value });
          tx.onerror = () => resolve({ ok: false, value: undefined });
          tx.onabort = () => resolve({ ok: false, value: undefined });
        } catch {
          resolve({ ok: false, value: undefined });
        }
      }),
  );
}

export async function rememberDocument(doc: IndexedDocument): Promise<boolean> {
  return (await run(INDEX_STORE, "readwrite", (s) => s.put(doc))).ok;
}

/** Change some fields of a document already listed; false if it is not. */
export async function updateDocument(
  id: string,
  patch: Partial<Omit<IndexedDocument, "id">>,
): Promise<boolean> {
  const db = await openDb();
  if (!db) return false;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(INDEX_STORE, "readwrite");
      const store = tx.objectStore(INDEX_STORE);
      let found = false;
      const get = store.get(id);
      get.onsuccess = () => {
        const current = get.result as IndexedDocument | undefined;
        if (!current) return;
        found = true;
        store.put({ ...current, ...patch, id, updatedAt: Date.now() });
      };
      tx.oncomplete = () => resolve(found);
      tx.onerror = () => resolve(false);
      tx.onabort = () => resolve(false);
    } catch {
      resolve(false);
    }
  });
}

export async function forgetDocument(id: string): Promise<boolean> {
  return (await run(INDEX_STORE, "readwrite", (s) => s.delete(id))).ok;
}

export async function getDocument(id: string): Promise<IndexedDocument | null> {
  return (await run<IndexedDocument>(INDEX_STORE, "readonly", (s) => s.get(id))).value ?? null;
}

export async function isIndexed(id: string): Promise<boolean> {
  const r = await run<IDBValidKey | undefined>(INDEX_STORE, "readonly", (s) => s.getKey(id));
  return r.value !== undefined;
}

export async function listDocuments(): Promise<IndexedDocument[]> {
  const r = await run<IndexedDocument[]>(
    INDEX_STORE,
    "readonly",
    (s) => s.getAll() as IDBRequest<IndexedDocument[]>,
  );
  return r.value ?? [];
}

export async function getMeta<T>(key: string): Promise<T | undefined> {
  return (await run<T>(META_STORE, "readonly", (s) => s.get(key) as IDBRequest<T>)).value;
}

export async function setMeta(key: string, value: unknown): Promise<boolean> {
  return (await run(META_STORE, "readwrite", (s) => s.put(value, key))).ok;
}

/**
 * Chrome's built-in translator (the Translator API, Chrome and Edge 138+
 * on a computer), for the browser engine's 'chrome' provider
 * (services/engine/ct_browser/local_mt.py).
 *
 * The API exists only on a page, never in a worker, so the engine worker
 * cannot call it: it sends each batch as a synchronous request to
 * /engine/api/__mt/chrome, which the Service Worker hands to the host tab,
 * and the host answers with chromeTranslate() below. Nothing leaves the
 * computer: Chrome translates with a model it keeps on disk.
 *
 * A language Chrome has not downloaded yet can only be fetched while the
 * teacher is clicking (create() needs a user activation then), so the
 * Translate button calls primeChromeTranslator() first. The host never
 * downloads one itself: a language that is not ready there is translated
 * with Mozilla's models instead.
 */

// Our target codes -> Chrome's, for the languages Chrome translates into
// from English. Mirrors CHROME_CODES in ct_browser/local_mt.py (the engine
// decides which code a job uses; this copy is for priming on the click).
// Left out: ht and ur (Chrome has no model), pt-PT (Chrome's "pt" is
// Brazilian Portuguese).
export const CHROME_CODES: Record<string, string> = {
  es: "es", fr: "fr", ar: "ar", bn: "bn", ru: "ru", ko: "ko", ja: "ja", vi: "vi",
  pl: "pl", he: "he", uk: "uk", "pt-BR": "pt", "zh-Hans": "zh", "zh-Hant": "zh-Hant",
};

type Availability = "unavailable" | "downloadable" | "downloading" | "available";

interface ChromeTranslator {
  translate(text: string): Promise<string>;
  destroy(): void;
}

interface TranslatorStatic {
  availability(o: { sourceLanguage: string; targetLanguage: string }): Promise<Availability>;
  create(o: {
    sourceLanguage: string;
    targetLanguage: string;
    monitor?: (m: EventTarget) => void;
  }): Promise<ChromeTranslator>;
}

function api(): TranslatorStatic | null {
  if (typeof self === "undefined") return null;
  const t = (self as unknown as { Translator?: TranslatorStatic }).Translator;
  return t && typeof t.availability === "function" ? t : null;
}

const pair = (code: string) => ({ sourceLanguage: "en", targetLanguage: code });

/** Called from the Translate button's click, before the job is sent: if
 * Chrome can translate into this language but has not downloaded it, the
 * download starts now, while the click still counts, and is waited for (a
 * few seconds; `onProgress` gets 0-1). Never throws: a browser without the
 * API, or a download that fails, just means Mozilla's models translate. */
export async function primeChromeTranslator(
  target: string,
  onProgress?: (fraction: number) => void,
): Promise<void> {
  const T = api();
  const code = CHROME_CODES[target];
  if (!T || !code) return;
  try {
    const state = await T.availability(pair(code));
    if (state !== "downloadable" && state !== "downloading") return;
    const made = await withTimeout(
      T.create({
        ...pair(code),
        monitor(m) {
          m.addEventListener("downloadprogress", (e) => onProgress?.((e as ProgressEvent).loaded));
        },
      }),
      PRIME_TIMEOUT_MS,
    );
    made?.destroy();
  } catch {
    /* Mozilla's models translate instead */
  }
}

// A slow network should not hold her click for long: past this the job is
// sent anyway, and Mozilla's models translate if Chrome is still fetching.
const PRIME_TIMEOUT_MS = 60_000;

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T | null> {
  return Promise.race([p, new Promise<null>((r) => setTimeout(() => r(null), ms))]);
}

// The host's translators, one per language, made once and kept.
const made = new Map<string, Promise<ChromeTranslator | null>>();

function translatorFor(code: string): Promise<ChromeTranslator | null> {
  let t = made.get(code);
  if (!t) {
    const T = api();
    t = !T
      ? Promise.resolve(null)
      : T.availability(pair(code))
          .then((state) => (state === "available" ? T.create(pair(code)) : null))
          .catch(() => null);
    made.set(code, t);
    // A language that was not ready is asked about again next time.
    void t.then((x) => { if (!x) made.delete(code); });
  }
  return t;
}

/** The host: whether Chrome can translate into `code` right now, without a
 * download (the worker asks before a job). */
export async function chromeReady(code: string): Promise<boolean> {
  return (await translatorFor(code)) !== null;
}

/** The host: each text translated, in order. Throws if Chrome cannot. */
export async function chromeTranslate(code: string, texts: string[]): Promise<string[]> {
  const t = await translatorFor(code);
  if (!t) throw new Error(`Chrome cannot translate into ${code} here`);
  const out: string[] = [];
  // One at a time: Chrome holds later calls until the earlier finish anyway.
  for (const text of texts) out.push(text.trim() ? await t.translate(text) : text);
  return out;
}

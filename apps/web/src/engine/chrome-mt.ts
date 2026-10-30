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

/* --- Checking in English, from the editor page ---
 * A teacher who cannot read the translation reads it back in English. That
 * changes nothing, so it needs no engine: the page asks Chrome directly,
 * the other way round (Spanish to English). Never the DeepL backup: that
 * shared allowance is kept for translating worksheets. */

export type EnglishCheckProblem = "no-chrome" | "no-language" | "failed";

export class EnglishCheckError extends Error {
  constructor(readonly problem: EnglishCheckProblem) {
    super(problem);
  }
}

const toEnglishMade = new Map<string, Promise<ChromeTranslator>>();

/** Chrome's translator from `target` (our code, such as "es") into English.
 * Call it from a click first: a language Chrome has not downloaded can
 * only be fetched while the click still counts. Rejects with an
 * EnglishCheckError. */
export function englishTranslator(target: string): Promise<ChromeTranslator> {
  const T = api();
  if (!T) return Promise.reject(new EnglishCheckError("no-chrome"));
  const code = CHROME_CODES[target];
  if (!code) return Promise.reject(new EnglishCheckError("no-language"));
  let t = toEnglishMade.get(code);
  if (!t) {
    const back = { sourceLanguage: code, targetLanguage: "en" };
    t = T.availability(back)
      .catch(() => "unavailable" as const)
      .then((state) => {
        if (state === "unavailable") throw new EnglishCheckError("no-language");
        return T.create(back);
      })
      .catch((e) => {
        throw e instanceof EnglishCheckError ? e : new EnglishCheckError("failed");
      });
    toEnglishMade.set(code, t);
    // Asked again next time: the download may have needed a click.
    t.catch(() => toEnglishMade.delete(code));
  }
  return t;
}

/** Whether the English check can run with no click: Chrome already has
 * this language here, so nothing needs downloading. Never rejects. */
export async function englishReadyHere(target: string): Promise<boolean> {
  const T = api();
  const code = CHROME_CODES[target];
  if (!T || !code) return false;
  try {
    return (await T.availability({ sourceLanguage: code, targetLanguage: "en" })) === "available";
  } catch {
    return false;
  }
}

const englishSaid = new Map<string, string>();

/** What `text`, in `target`, says in English. */
export async function toEnglish(target: string, text: string): Promise<string> {
  const key = `${target}\u0000${text}`;
  const known = englishSaid.get(key);
  if (known !== undefined) return known;
  const translator = await englishTranslator(target);
  let said: string;
  try {
    said = await translator.translate(text);
  } catch {
    throw new EnglishCheckError("failed");
  }
  englishSaid.set(key, said);
  return said;
}

/** Her words for each problem, naming the language. */
export function englishCheckMessage(problem: EnglishCheckProblem, language: string): string {
  switch (problem) {
    case "no-chrome":
      return "Checking in English needs Chrome or Edge on a computer. Open this page there to check it.";
    case "no-language":
      return `Chrome cannot read ${language} back into English on this computer.`;
    default:
      return "Chrome's translator did not answer. Try again in a moment.";
  }
}

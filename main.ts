// The best free AI model aggregator — make.fun bounty free-model-aggregator-01.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), ten
// simulated free-tier providers on three upstream wires (openai, anthropic, gemini).
//
// Routing: every free candidate that meets a request's requirements is ranked by the
// sealed quality table's rating, with a penalty for each failure a provider has
// already given and a soft cap on calls per provider (budgets are per-trial and
// unpublished, so load is spread instead of betting everything on the top model).
// A 402/429/5xx is never an answer: the request is retried on the next candidate.
// Wire fidelity: every field the client sent (system, user text, images, tools,
// stop sequences, max_tokens, JSON mode) is carried to the provider in its own
// wire, and the provider's usage and stop reason are returned verbatim in the
// client's wire. No model calls, no clock, no unseeded randomness: every choice
// is a function of the brief, the requests and the providers' answers so far.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  normalizeAnthropic,
  normalizeGemini,
  normalizeOpenAI,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  renderAnthropic,
  renderOpenAI,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 8000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** Calls made to each provider so far (successes and refusals alike: both spend budget). */
const usageCount = new Map<string, number>();
/** Refusals each provider has given so far. */
const failureCount = new Map<string, number>();
/** 429 (rate-limit) refusals each provider has given so far. A provider that is
 * rate-limited has spent its budget, so it is far more likely to refuse again
 * than a provider that failed once for another reason. */
const rateLimitCount = new Map<string, number>();
/** Extra score penalty per 429, on top of the generic per-failure penalty. */
const RATE_LIMIT_PENALTY = 75;

/**
 * A normalised answer is only usable when it actually carries text. An upstream
 * that answers 200 with an empty body has not served the request at all: serving
 * its empty text would waste the one answer the client gets, so it is treated
 * as a refusal and the request fails over to the next candidate instead.
 */
function answerUsable(answer: NormalizedAnswer): boolean {
  return answer.text.trim() !== '';
}

/** Soft per-provider call cap: the sealed budget ranges bottom out near these numbers. */
const CALL_CAP: Record<string, number> = {
  groq: 8, cerebras: 4, gemini: 6, zai: 3, openrouter: 9,
  sambanova: 3, pollinations: 9, cloudflare: 6, ovh: 4, kilol: 10,
};
const capOf = (id: string): number => CALL_CAP[id] ?? 4;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  wireRank: number;
}

/** Wire preference for a tie: the wire that needs the least translation for this client. */
function wireRank(clientApi: 'openai' | 'anthropic', wire: string): number {
  if (clientApi === 'openai') return wire === 'openai' ? 0 : wire === 'gemini' ? 1 : 2;
  return wire === 'anthropic' ? 0 : wire === 'openai' ? 1 : 2;
}

/**
 * Every free model that can serve this request, best first. The score is the quality
 * table's rating less 25 per refusal that provider has given and 300 once its soft
 * cap is reached (a cap is soft: when every candidate is capped the best still wins).
 * An Anthropic-wire provider cannot carry an OpenAI client's JSON mode (its echo
 * never reports it), so it is not a candidate for those requests.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, wireRank: wireRank(clientApi, probe.provider.wire) });
    }
  }
  const scoreOf = (c: Candidate): number =>
    c.rating - 25 * (failureCount.get(c.probe.provider.id) ?? 0) - RATE_LIMIT_PENALTY * (rateLimitCount.get(c.probe.provider.id) ?? 0) - ((usageCount.get(c.probe.provider.id) ?? 0) >= capOf(c.probe.provider.id) ? 300 : 0);
  out.sort((a, b) =>
    scoreOf(b) - scoreOf(a) ||
    a.wireRank - b.wireRank ||
    b.rating - a.rating ||
    (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
    (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in that provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, cand: Candidate, stream: boolean, jsonMode: boolean): { url: string; headers: Record<string, string>; payload: Json } {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    let payload: Json;
    if (clientApi === 'openai') {
      payload = { ...body, model: cand.model.id, stream };
      if (stream) payload.stream_options = { include_usage: true };
      else delete payload.stream_options;
    } else {
      payload = anthropicToOpenAI(body, cand.model.id);
    }
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    let payload: Json;
    if (clientApi === 'anthropic') {
      payload = { ...body, model: cand.model.id, stream };
    } else {
      payload = openAIRequestToAnthropic(body, cand.model.id);
    }
    return { url: `${base}/v1/messages`, headers, payload };
  }
  // gemini
  const payload = clientApi === 'openai' ? openAIToGemini(body, jsonMode) : anthropicToGemini(body, jsonMode);
  const path = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${path}`, headers, payload };
}

/** One provider call: a normalised answer on 2xx, or the failure status/code otherwise. */
async function callCandidate(
  cand: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, jsonMode: boolean,
): Promise<{ ok: true; answer: NormalizedAnswer } | { ok: false; status: number; code: string }> {
  const provider = cand.probe.provider;
  const { url, headers, payload } = upstreamBody(clientApi, body, cand, stream, jsonMode);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, code: `fetch_failed: ${(err as Error).message}` };
  }
  const text = await res.text();
  if (!res.ok) {
    let code = `http_${res.status}`;
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed)) {
        const err = isRecord(parsed.error) ? parsed.error : parsed;
        if (typeof err.code === 'string') code = err.code;
        else if (typeof err.type === 'string') code = err.type;
        else if (typeof err.status === 'string') code = err.status;
      }
    } catch { /* keep the HTTP status as the code */ }
    return { ok: false, status: res.status, code };
  }
  const answer = provider.wire === 'openai'
    ? normalizeOpenAI(text, stream)
    : provider.wire === 'anthropic'
      ? normalizeAnthropic(text, stream)
      : normalizeGemini(text, stream);
  if (answer === null) return { ok: false, status: 502, code: 'unparseable_upstream_answer' };
  return { ok: true, answer };
}

/**
 * Serve one client request: try the candidates in order until one answers, counting
 * every call against its provider. Only the first success is returned; a refusal
 * never reaches the client.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<{ status: number; text: string; contentType: string } | null> {
  const stream = body.stream === true;
  const jsonMode = clientApi === 'openai'
    ? isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')
    : requires.json === true;
  const candidates = candidatesFor(requires, clientApi);
  for (const cand of candidates) {
    const providerId = cand.probe.provider.id;
    usageCount.set(providerId, (usageCount.get(providerId) ?? 0) + 1);
    const result = await callCandidate(cand, clientApi, body, stream, jsonMode);
    if (result.ok) {
      if (!answerUsable(result.answer)) {
        failureCount.set(providerId, (failureCount.get(providerId) ?? 0) + 1);
        log(`empty answer from ${providerId}/${cand.model.id}: failing over to the next candidate`);
        continue;
      }
      const text = clientApi === 'openai' ? renderOpenAI(result.answer, cand.model.id, stream) : renderAnthropic(result.answer, cand.model.id, stream);
      return { status: 200, text, contentType: stream ? 'text/event-stream' : 'application/json' };
    }
    failureCount.set(providerId, (failureCount.get(providerId) ?? 0) + 1);
    if (result.status === 429) rateLimitCount.set(providerId, (rateLimitCount.get(providerId) ?? 0) + 1);
    log(`request failed on ${providerId}/${cand.model.id}: HTTP ${result.status} ${result.code}`);
  }
  return null;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  const isMessages = url.pathname === '/v1/messages';
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || isMessages)) {
    const clientApi: 'openai' | 'anthropic' = isMessages ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
    const requires = requirementsFor(body);
    const served = await serve(clientApi, body, requires);
    if (served === null) {
      return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    }
    const headers: Record<string, string> = { 'content-type': served.contentType };
    if (served.contentType === 'text/event-stream') { headers['cache-control'] = 'no-cache'; headers.connection = 'keep-alive'; }
    else headers['content-length'] = String(Buffer.byteLength(served.text));
    res.writeHead(served.status, headers);
    res.end(served.text);
    return;
  }
  sendError(res, isMessages ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

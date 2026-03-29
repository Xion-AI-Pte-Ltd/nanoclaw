/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF, like before)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json - polled and consumed
 *          Sentinel: /workspace/ipc/input/_close - signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 *   Final marker after loop ends signals completion.
 */

import fs from 'fs';
import path from 'path';
import { spawn, spawnSync } from 'child_process';
import { GoogleGenAI } from '@google/genai';
import {
  query,
  HookCallback,
  PreCompactHookInput,
  PreToolUseHookInput,
} from '@anthropic-ai/claude-agent-sdk';
import { fileURLToPath } from 'url';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  secrets?: Record<string, string>;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

interface QueryResult {
  newSessionId?: string;
  lastAssistantUuid?: string;
  closedDuringQuery: boolean;
}

interface ExecutedGeminiResult {
  handled: boolean;
  resultText: string;
}

class GeminiTemplateScriptError extends Error {
  marker: string;

  constructor(marker: string) {
    super(`Gemini returned templated Python scaffold containing ${marker}`);
    this.name = 'GeminiTemplateScriptError';
    this.marker = marker;
  }
}

type Provider = 'claude' | 'gemini';
const MISSING_MODULE_REGEX = /No module named ['"]([^'"]+)['"]/i;

interface GeminiHistoryMessage {
  role: 'user' | 'model';
  text: string;
}

interface GeminiSessionState {
  history: GeminiHistoryMessage[];
  lastAssistantUuid?: string;
}

interface GeminiPart {
  text?: string;
}

interface GeminiContent {
  parts?: GeminiPart[];
}

interface GeminiCandidate {
  content?: GeminiContent;
}

interface GeminiGenerateResponse {
  candidates?: GeminiCandidate[];
}

type GeminiPromptContent = {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
};

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;
const GEMINI_SESSION_ROOT = '/workspace/group/.nanoclaw/gemini-sessions';

/**
 * Push-based async iterable for streaming user messages to the SDK.
 * Keeps the iterable alive until end() is called, preventing isSingleUserTurn.
 */
class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>((r) => {
        this.waiting = r;
      });
      this.waiting = null;
    }
  }
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

function getSessionSummary(
  sessionId: string,
  transcriptPath: string,
): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    log(`Sessions index not found at ${indexPath}`);
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(
      fs.readFileSync(indexPath, 'utf-8'),
    );
    const entry = index.entries.find((e) => e.sessionId === sessionId);
    if (entry?.summary) {
      return entry.summary;
    }
  } catch (err) {
    log(
      `Failed to read sessions index: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  return null;
}

/**
 * Archive the full transcript to conversations/ before compaction.
 */
function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      log('No transcript found for archiving');
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);

      if (messages.length === 0) {
        log('No messages to archive');
        return {};
      }

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(
        messages,
        summary,
        assistantName,
      );
      fs.writeFileSync(filePath, markdown);

      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(
        `Failed to archive transcript: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    return {};
  };
}

// Secrets to strip from Bash tool subprocess environments.
// These are needed by the model providers for API auth but should never
// be visible to commands kit runs.
const SECRET_ENV_VARS = [
  'ANTHROPIC_API_KEY',
  'CLAUDE_CODE_OAUTH_TOKEN',
  'GEMINI_API_KEY',
  'GOOGLE_API_KEY',
];

function createSanitizeBashHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preInput = input as PreToolUseHookInput;
    const command = (preInput.tool_input as { command?: string })?.command;
    if (!command) return {};

    const unsetPrefix = `unset ${SECRET_ENV_VARS.join(' ')} 2>/dev/null; `;
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        updatedInput: {
          ...(preInput.tool_input as Record<string, unknown>),
          command: unsetPrefix + command,
        },
      },
    };
  };
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text =
          typeof entry.message.content === 'string'
            ? entry.message.content
            : entry.message.content
                .map((c: { text?: string }) => c.text || '')
                .join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {
      // Ignore malformed lines.
    }
  }

  return messages;
}

function formatTranscriptMarkdown(
  messages: ParsedMessage[],
  title?: string | null,
  assistantName?: string,
): string {
  const now = new Date();
  const formatDateTime = (d: Date) =>
    d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : assistantName || 'Assistant';
    const content =
      msg.content.length > 2000
        ? `${msg.content.slice(0, 2000)}...`
        : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Check for _close sentinel.
 */
function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try {
      fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
    } catch {
      // Ignore unlink failures.
    }
    return true;
  }
  return false;
}

/**
 * Drain all pending IPC input messages.
 * Returns messages found, or empty array.
 */
function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs
      .readdirSync(IPC_INPUT_DIR)
      .filter((f) => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(
          `Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`,
        );
        try {
          fs.unlinkSync(filePath);
        } catch {
          // Ignore cleanup errors.
        }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages as a single string, or null if _close.
 */
function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

function resolveProvider(sdkEnv: Record<string, string | undefined>): Provider {
  const rawProvider = (sdkEnv.AGENT_PROVIDER || 'claude').toLowerCase();
  if (rawProvider === 'claude' || rawProvider === 'gemini') {
    return rawProvider;
  }
  throw new Error(
    `Unsupported AGENT_PROVIDER="${sdkEnv.AGENT_PROVIDER}". Supported values: claude, gemini`,
  );
}

function getGeminiApiKey(sdkEnv: Record<string, string | undefined>): string {
  return sdkEnv.GEMINI_API_KEY || sdkEnv.GOOGLE_API_KEY || '';
}

function getGeminiVertexProjectId(
  sdkEnv: Record<string, string | undefined>,
): string {
  const configured =
    sdkEnv.VERTEX_PROJECT_ID ||
    sdkEnv.GOOGLE_CLOUD_PROJECT ||
    sdkEnv.RUNTIME_PROJECT_ID ||
    sdkEnv.GCP_PROJECT ||
    sdkEnv.GCLOUD_PROJECT ||
    'vault-dev-270023';
  return configured.trim();
}

function getGeminiVertexLocation(
  sdkEnv: Record<string, string | undefined>,
): string {
  const configured =
    sdkEnv.VERTEX_LOCATION ||
    sdkEnv.GOOGLE_CLOUD_LOCATION ||
    sdkEnv.GEMINI_LOCATION ||
    'global';
  return configured.trim();
}

function envEnabled(value: string | undefined): boolean {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  return (
    normalized === '1' ||
    normalized === 'true' ||
    normalized === 'yes' ||
    normalized === 'on'
  );
}

async function fetchMetadata(pathSuffix: string): Promise<string> {
  const res = await fetch(
    `http://metadata.google.internal/computeMetadata/v1/${pathSuffix}`,
    {
      method: 'GET',
      headers: { 'Metadata-Flavor': 'Google' },
    },
  );
  if (!res.ok) {
    throw new Error(
      `Metadata request failed (${res.status}): ${pathSuffix}`,
    );
  }
  return (await res.text()).trim();
}

async function getVertexProjectId(
  sdkEnv: Record<string, string | undefined>,
): Promise<string> {
  const configured =
    sdkEnv.VERTEX_PROJECT_ID || sdkEnv.GOOGLE_CLOUD_PROJECT || '';
  if (configured) return configured;
  return fetchMetadata('project/project-id');
}

async function getGcpAccessToken(
  sdkEnv: Record<string, string | undefined>,
): Promise<string> {
  if (sdkEnv.GOOGLE_ACCESS_TOKEN) return sdkEnv.GOOGLE_ACCESS_TOKEN;
  const res = await fetch(
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
    {
      method: 'GET',
      headers: { 'Metadata-Flavor': 'Google' },
    },
  );
  if (!res.ok) {
    throw new Error(`Failed to get access token from metadata (${res.status})`);
  }
  const body = (await res.json()) as { access_token?: string };
  if (!body.access_token) {
    throw new Error('Metadata token response missing access_token');
  }
  return body.access_token;
}

function maskedPrefix(value: string): string {
  const prefix = value.slice(0, 4);
  return `${prefix}... (len=${value.length})`;
}

function logProviderSecretPreview(
  provider: Provider,
  sdkEnv: Record<string, string | undefined>,
): void {
  if (provider === 'gemini') {
    const key = getGeminiApiKey(sdkEnv);
    if (key) {
      const keyName = sdkEnv.GEMINI_API_KEY ? 'GEMINI_API_KEY' : 'GOOGLE_API_KEY';
      log(`Auth debug: ${keyName} prefix=${maskedPrefix(key)}`);
      return;
    }
    log(
      `Auth debug: no Gemini API key present; using Vertex AI ADC fallback project=${getGeminiVertexProjectId(
        sdkEnv,
      )} location=${getGeminiVertexLocation(sdkEnv)}`,
    );
    return;
  }

  if (sdkEnv.ANTHROPIC_API_KEY) {
    log(
      `Auth debug: ANTHROPIC_API_KEY prefix=${maskedPrefix(sdkEnv.ANTHROPIC_API_KEY)}`,
    );
    return;
  }
  if (sdkEnv.CLAUDE_CODE_OAUTH_TOKEN) {
    log(
      `Auth debug: CLAUDE_CODE_OAUTH_TOKEN prefix=${maskedPrefix(
        sdkEnv.CLAUDE_CODE_OAUTH_TOKEN,
      )}`,
    );
    return;
  }
  log('Auth debug: no Claude auth secret present');
}

function buildMemoryContext(containerInput: ContainerInput): string {
  const sections: string[] = [];

  const groupClaudePath = '/workspace/group/CLAUDE.md';
  if (fs.existsSync(groupClaudePath)) {
    sections.push(
      `## Group Memory (CLAUDE.md)\n${fs.readFileSync(groupClaudePath, 'utf-8')}`,
    );
  }

  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    sections.push(
      `## Global Memory (CLAUDE.md)\n${fs.readFileSync(globalClaudeMdPath, 'utf-8')}`,
    );
  }

  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (!fs.statSync(fullPath).isDirectory()) continue;
      const claudeMdPath = path.join(fullPath, 'CLAUDE.md');
      if (!fs.existsSync(claudeMdPath)) continue;
      const relativeName = fullPath.replace('/workspace/', '');
      sections.push(
        `## Additional Memory (${relativeName}/CLAUDE.md)\n${fs.readFileSync(claudeMdPath, 'utf-8')}`,
      );
    }
  }

  return sections.join('\n\n');
}

function ensureGeminiSessionDir(): void {
  fs.mkdirSync(GEMINI_SESSION_ROOT, { recursive: true });
}

function generateGeminiSessionId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `g_${Date.now().toString(36)}_${rand}`;
}

function getGeminiSessionPath(sessionId: string): string {
  return path.join(GEMINI_SESSION_ROOT, `${sessionId}.json`);
}

function loadGeminiSession(sessionId: string | undefined): {
  sessionId: string;
  state: GeminiSessionState;
} {
  ensureGeminiSessionDir();

  const effectiveSessionId = sessionId || generateGeminiSessionId();
  const sessionPath = getGeminiSessionPath(effectiveSessionId);
  if (!fs.existsSync(sessionPath)) {
    return { sessionId: effectiveSessionId, state: { history: [] } };
  }

  try {
    const parsed = JSON.parse(
      fs.readFileSync(sessionPath, 'utf-8'),
    ) as GeminiSessionState;
    if (!Array.isArray(parsed.history)) {
      return { sessionId: effectiveSessionId, state: { history: [] } };
    }
    return { sessionId: effectiveSessionId, state: parsed };
  } catch (err) {
    log(
      `Failed to load Gemini session ${effectiveSessionId}: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
    return { sessionId: effectiveSessionId, state: { history: [] } };
  }
}

function saveGeminiSession(sessionId: string, state: GeminiSessionState): void {
  ensureGeminiSessionDir();
  fs.writeFileSync(
    getGeminiSessionPath(sessionId),
    JSON.stringify(state, null, 2) + '\n',
  );
}

function normalizeGeminiText(response: GeminiGenerateResponse): string {
  const candidate = response.candidates?.[0];
  const parts = candidate?.content?.parts || [];
  const text = parts
    .map((part) => part.text)
    .filter((value): value is string => typeof value === 'string')
    .join('');
  return text.trim();
}

function buildGeminiSystemInstruction(containerInput: ContainerInput): string {
  const memory = buildMemoryContext(containerInput);
  const base =
    'You are NanoClaw, a coding and task assistant running inside an isolated workspace container. ' +
    'Be concise, accurate, and explicit about limitations. ' +
    'If a Python execution fails due to a missing module, correct the issue by installing the dependency and retrying before giving up. ' +
    'If the task involves PDFs, do not write local PDF parsing code; use Gemini-based extraction of the PDF content or structured fields instead of local read/parsing libraries. ' +
    'Prefer the Google GenAI SDK for Gemini work. If a Gemini API key is present, use direct API mode. ' +
    'If no API key is present, use Vertex AI with Google Application Default Credentials and do not fail solely because an API key is missing. ' +
    'Do not invent a local PDF parser or require a Gemini API key when ADC is available. ' +
    'When creating spreadsheets with xlsxwriter, only use chart types that xlsxwriter supports. ' +
    "Do not call add_chart('waterfall') or other unsupported chart types. " +
    'If a waterfall visualization is requested, approximate it using supported chart types such as stacked bar or column charts and note the approximation in the workbook. ' +
    'Treat trial_balance as optional and untrusted: it may be missing, null, a string, or another non-dict value. Only call .get() on it after checking isinstance(trial_balance, dict).';
  if (!memory) return base;

  return `${base}\n\nUse this persistent memory context if relevant:\n\n${memory}`;
}

function extractMissingPythonModule(errorText: string): string | null {
  const match = errorText.match(MISSING_MODULE_REGEX);
  if (!match) return null;
  const moduleName = match[1].trim();
  return moduleName || null;
}

function extractFencedBlock(text: string, language: string): string | null {
  const pattern = new RegExp(String.raw`\`\`\`${language}\n([\s\S]*?)\n\`\`\``, 'i');
  const match = text.match(pattern);
  return match ? match[1].trim() : null;
}

function normalizePythonScriptText(text: string): string {
  const trimmed = text.trim();
  const fenced = extractFencedBlock(trimmed, 'python') ?? extractFencedBlock(trimmed, 'py');
  const body = (fenced || trimmed)
    .replace(/^\s*```(?:python|py)?\s*\n/i, '')
    .replace(/\n\s*```\s*$/i, '')
    .replace(/^\s*(?:python|py)\s*$/im, '')
    .trim();
  return body;
}

function sanitizeGeminiPythonScript(scriptSource: string): {
  script: string;
  notes: string[];
} {
  let sanitized = scriptSource;
  const notes: string[] = [];

  if (/\btrial_balance\.get\(/.test(sanitized)) {
    sanitized = sanitized.replace(
      /\btrial_balance\.get\(/g,
      '_nanoclaw_trial_balance_get(trial_balance, ',
    );
    notes.push(
      'Rewrote direct trial_balance.get(...) calls to a guarded helper because trial_balance may be missing or non-dict.',
    );
  }

  const waterfallReplacements: Array<[RegExp, string]> = [
    [
      /add_chart\(\s*\{\s*['"]type['"]\s*:\s*['"]waterfall['"]\s*\}\s*\)/g,
      "add_chart({'type': 'column'})",
    ],
    [
      /add_chart\(\s*['"]waterfall['"]\s*\)/g,
      "add_chart('column')",
    ],
    [
      /\{\s*['"]type['"]\s*:\s*['"]waterfall['"]\s*\}/g,
      "{'type': 'column'}",
    ],
  ];
  for (const [pattern, replacement] of waterfallReplacements) {
    if (pattern.test(sanitized)) {
      sanitized = sanitized.replace(pattern, replacement);
      notes.push(
        'Replaced an unsupported waterfall chart request with a supported column chart before execution.',
      );
    }
  }

  if (notes.length > 0) {
    const prelude = [
      'def _nanoclaw_trial_balance_get(value, key, default=None):',
      '    if isinstance(value, dict):',
      '        return value.get(key, default)',
      '    return default',
      '',
    ].join('\n');
    sanitized = `${prelude}${sanitized}`;
  }

  return { script: sanitized, notes };
}

function detectTemplateScriptMarker(scriptSource: string): string | null {
  const lowered = scriptSource.toLowerCase();
  const markers = [
    'dummy_json_content',
    'cash_flow_actuals',
    'cash_flow_budget',
    'get_cash_flow',
    'get_cash_balance',
    'actual_cash_flow_items',
    'budget_cash_flow_items',
    'sample dataset',
    'sample data',
    'example dataset',
    'example data',
    'synthetic data',
    'placeholder data',
    'mock data',
    'demo data',
    'lorem ipsum',
  ];

  for (const marker of markers) {
    if (lowered.includes(marker)) {
      return marker;
    }
  }

  if (lowered.includes('create_empty_workbook_with_note(')) {
    return 'create_empty_workbook_with_note';
  }

  return null;
}

function extractLastJsonObject(text: string): Record<string, unknown> | null {
  const jsonBlock = extractFencedBlock(text, 'json');
  if (jsonBlock) {
    try {
      return JSON.parse(jsonBlock) as Record<string, unknown>;
    } catch {
      // Fall through to brace scanning.
    }
  }

  const start = text.lastIndexOf('{');
  const end = text.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try {
      return JSON.parse(text.slice(start, end + 1)) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  return null;
}

async function executePythonScript(
  scriptSource: string,
  allowDependencyRepair = true,
): Promise<{ stdout: string; stderr: string; exitCode: number | null }> {
  const tmpDir = '/workspace/group/.nanoclaw/runtime';
  const finalOutputDir = '/workspace/group/output/final';
  fs.mkdirSync(tmpDir, { recursive: true });
  fs.mkdirSync(finalOutputDir, { recursive: true });
  const scriptPath = path.join(tmpDir, `gemini_exec_${Date.now().toString(36)}.py`);
  const bootstrap = [
    'import os',
    "os.chdir('/workspace/group')",
    "os.makedirs('output/final', exist_ok=True)",
    "os.makedirs('/workspace/group/output/final', exist_ok=True)",
    '',
  ].join('\n');
  fs.writeFileSync(scriptPath, bootstrap + scriptSource);

  return new Promise((resolve, reject) => {
    const child = spawn('python3', [scriptPath], {
      cwd: '/workspace/group',
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });
    child.on('error', reject);
    child.on('close', (exitCode) => {
      resolve({ stdout, stderr, exitCode });
    });
  });
}

async function requestGeminiPythonCorrection(
  client: GoogleGenAI,
  model: string,
  contents: GeminiPromptContent[],
  systemInstruction: string,
  failureContext: string,
): Promise<string> {
  const contextHints: string[] = [];
  const lowerContext = failureContext.toLowerCase();
  if (lowerContext.includes("keyerror: 'data'") || lowerContext.includes('response[\'data\']') || lowerContext.includes('response["data"]')) {
    contextHints.push(
      "The bundle schema for this report uses datasets_by_key['general_ledger']['response']['result']['gl'] for ledger rows and datasets_by_key['chart_of_accounts']['response']['result'] for chart-of-accounts classifications. Do not access response['data'].",
    );
  }
  if (lowerContext.includes("keyerror: 'account_id'") || lowerContext.includes('account_id')) {
    contextHints.push(
      "There is no account_id column in the ledger rows. Use the Account field from general_ledger rows and join it to the chart_of_accounts dict keyed by account name.",
    );
  }
  if (lowerContext.includes("str' object has no attribute 'get'") || lowerContext.includes('trial_balance')) {
    contextHints.push(
      'trial_balance is optional and may not be a dict. Guard with isinstance(trial_balance, dict) before using .get(), otherwise treat it as unavailable and continue.',
    );
  }
  if (lowerContext.includes('cash_flow_actuals') || lowerContext.includes('cash_flow_budget')) {
    contextHints.push(
      "Do not emit scaffold names such as cash_flow_actuals or cash_flow_budget. Read only the staged bundle datasets and their existing keys.",
    );
  }
  if (lowerContext.includes('```python') || lowerContext.includes('leading "python" line')) {
    contextHints.push(
      'Return plain Python only. Do not wrap it in markdown fences and do not prefix it with the word python.',
    );
  }
  if (
    lowerContext.includes("unknown chart type 'waterfall'") ||
    lowerContext.includes("chart.add_series(") ||
    lowerContext.includes("add_chart('waterfall')")
  ) {
    contextHints.push(
      'xlsxwriter does not support a waterfall chart type. Replace it with a supported chart, such as stacked bar or column, and do not call add_chart("waterfall").',
    );
  }
  const codeSkeleton = [
    'Use this bundle access pattern as the starting point:',
    'bundle = json.load(open("input/financial_cash_flow_variance_analysis_bundle.json", "r", encoding="utf-8"))',
    'datasets = bundle["datasets_by_key"]',
    'gl_rows = datasets["general_ledger"]["response"]["result"]["gl"]',
    'coa_map = datasets["chart_of_accounts"]["response"]["result"]',
    'trial_balance = datasets.get("trial_balance", {}).get("response", {}).get("result")',
    'Then build the workbook from those objects and write it to output/final/financial_cash_flow_variance_analysis.xlsx.',
  ].join(' ');
  const contextHintBlock = contextHints.length > 0 ? ` Additional correction hints: ${contextHints.join(' ')}` : '';
  const retryResponse = await client.models.generateContent({
    model,
    contents: [
      ...contents,
      {
        role: 'user',
        parts: [
          {
            text:
              'The previous Python output was invalid because it used templated or synthetic content. ' +
              `Rejected context: ${failureContext}. ` +
              'Regenerate a real script that reads the staged report bundle from input/, uses the staged data as the only source of truth, ' +
              'does not embed dummy_json_content or sample datasets, does not invent dataset names like cash_flow_actuals, cash_flow_budget, get_cash_flow, or get_cash_balance, ' +
              'and writes final deliverables only under output/final.' +
              ` ${codeSkeleton}` +
              contextHintBlock +
              ' Return only Python code with no markdown fences, no prose, and no leading "python" line.',
          },
        ],
      },
    ],
    config: {
      systemInstruction,
    },
  });

  const retrySdkText =
    retryResponse && typeof retryResponse === 'object' && 'text' in retryResponse
      ? String((retryResponse as { text?: unknown }).text || '').trim()
      : '';
  return normalizePythonScriptText(retrySdkText || normalizeGeminiText(retryResponse as GeminiGenerateResponse));
}

async function installPythonPackage(packageName: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn('python3', [
      '-m',
      'pip',
      'install',
      '--break-system-packages',
      '--no-cache-dir',
      packageName,
    ], {
      cwd: '/workspace/group',
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stderr = '';
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });
    child.on('error', reject);
    child.on('close', (exitCode) => {
      if (exitCode === 0) {
        resolve();
        return;
      }
      reject(new Error(stderr || `pip install ${packageName} exited with code ${exitCode}`));
    });
  });
}

function pythonModuleExists(moduleName: string): boolean {
  const probe = spawnSync(
    'python3',
    [
      '-c',
      'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)',
      moduleName,
    ],
    {
      cwd: '/workspace/group',
      env: process.env,
      stdio: 'ignore',
    },
  );
  return probe.status === 0;
}

async function ensurePythonDependencies(scriptSource: string): Promise<void> {
  const imports = new Set<string>();
  for (const rawLine of scriptSource.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const importMatch = line.match(/^import\s+([a-zA-Z0-9_.,\s]+)$/);
    if (importMatch) {
      const raw = (importMatch[1] || '').split(',');
      for (const item of raw) {
        const moduleName = item.trim().split(/\s+as\s+/i)[0]?.split('.')[0]?.trim();
        if (moduleName) imports.add(moduleName);
      }
      continue;
    }

    const fromMatch = line.match(/^from\s+([a-zA-Z0-9_\.]+)\s+import\s+/);
    if (fromMatch) {
      const moduleName = (fromMatch[1] || '').split('.')[0]?.trim();
      if (moduleName) imports.add(moduleName);
    }
  }

  const builtins = new Set([
    'os',
    'sys',
    'json',
    're',
    'math',
    'time',
    'datetime',
    'pathlib',
    'typing',
    'collections',
    'subprocess',
    'shutil',
    'tempfile',
    'traceback',
    'functools',
    'itertools',
    'statistics',
    'csv',
    'gzip',
    'zipfile',
    'hashlib',
    'base64',
    'urllib',
    'http',
  ]);

  const missing: string[] = [];
  for (const moduleName of imports) {
    if (builtins.has(moduleName)) continue;
    if (!pythonModuleExists(moduleName)) {
      missing.push(moduleName);
    }
  }

  for (const moduleName of missing) {
    log(`Installing missing Python dependency inferred from script: ${moduleName}`);
    await installPythonPackage(moduleName);
  }
}

async function maybeExecuteGeminiCodeResult(
  textResult: string,
  client: GoogleGenAI,
  model: string,
  contents: GeminiPromptContent[],
  systemInstruction: string,
): Promise<ExecutedGeminiResult> {
  const normalizedTextResult = normalizePythonScriptText(textResult);
  const pythonBlock = extractFencedBlock(textResult, 'python') || normalizedTextResult;
  if (!pythonBlock) {
    return { handled: false, resultText: normalizedTextResult || textResult };
  }

  const templateMarker = detectTemplateScriptMarker(pythonBlock);
  if (templateMarker) {
    throw new GeminiTemplateScriptError(templateMarker);
  }

  const sanitized = sanitizeGeminiPythonScript(pythonBlock);
  if (sanitized.notes.length > 0) {
    log(`Gemini script sanitized before execution: ${sanitized.notes.join(' | ')}`);
  }

  log('Gemini returned fenced Python; executing script in workspace.');
  await ensurePythonDependencies(sanitized.script);
  const execution = await executePythonScript(sanitized.script);
  const stdout = execution.stdout.trim();
  const stderr = execution.stderr.trim();

  if (execution.exitCode !== 0) {
    const errorText = stderr || stdout || `Python exited with code ${execution.exitCode}`;
    const missingModule = extractMissingPythonModule(errorText);
    if (missingModule) {
      log(`Missing Python module detected at runtime; installing ${missingModule} and retrying once.`);
      await installPythonPackage(missingModule);
      const retryExecution = await executePythonScript(sanitized.script, false);
      const retryStdout = retryExecution.stdout.trim();
      const retryStderr = retryExecution.stderr.trim();
      if (retryExecution.exitCode !== 0) {
        const retryErrorText =
          retryStderr || retryStdout || `Python exited with code ${retryExecution.exitCode}`;
        throw new Error(`Gemini-generated Python failed: ${retryErrorText}`);
      }
      const retryParsedJson = extractLastJsonObject(retryStdout);
      if (retryParsedJson) {
        return {
          handled: true,
          resultText: JSON.stringify(retryParsedJson),
        };
      }
      const retryOutputFiles = Array.from(
        retryStdout.matchAll(/output\/final\/[^\s"'`]+/g),
        (match) => match[0],
      );
      if (retryOutputFiles.length > 0) {
        return {
          handled: true,
          resultText: JSON.stringify({
            confidence: 1.0,
            output_files: retryOutputFiles,
            stdout: retryStdout.slice(0, 2000),
          }),
        };
      }
      return {
        handled: true,
        resultText: retryStdout || textResult,
      };
    }

    log(`Gemini Python failed at runtime; requesting corrected script once. Error: ${errorText}`);
    const correctedScript = await requestGeminiPythonCorrection(client, model, contents, systemInstruction, errorText);
    const normalizedCorrectedScript = normalizePythonScriptText(correctedScript);
    const correctedTemplateMarker = detectTemplateScriptMarker(normalizedCorrectedScript);
    if (correctedTemplateMarker) {
      throw new GeminiTemplateScriptError(correctedTemplateMarker);
    }
    const sanitizedCorrected = sanitizeGeminiPythonScript(normalizedCorrectedScript);
    if (sanitizedCorrected.notes.length > 0) {
      log(`Corrected Gemini script sanitized before execution: ${sanitizedCorrected.notes.join(' | ')}`);
    }
    await ensurePythonDependencies(sanitizedCorrected.script);
    const correctedExecution = await executePythonScript(sanitizedCorrected.script, false);
    const correctedStdout = correctedExecution.stdout.trim();
    const correctedStderr = correctedExecution.stderr.trim();
    if (correctedExecution.exitCode !== 0) {
      const correctedErrorText =
        correctedStderr || correctedStdout || `Python exited with code ${correctedExecution.exitCode}`;
      throw new Error(`Gemini-generated Python failed: ${correctedErrorText}`);
    }
    const correctedParsedJson = extractLastJsonObject(correctedStdout);
    if (correctedParsedJson) {
      return {
        handled: true,
        resultText: JSON.stringify(correctedParsedJson),
      };
    }
    const correctedOutputFiles = Array.from(
      correctedStdout.matchAll(/output\/final\/[^\s"'`]+/g),
      (match) => match[0],
    );
    if (correctedOutputFiles.length > 0) {
      return {
        handled: true,
        resultText: JSON.stringify({
          confidence: 1.0,
          output_files: correctedOutputFiles,
          stdout: correctedStdout.slice(0, 2000),
        }),
      };
    }
      return {
        handled: true,
        resultText: correctedStdout || textResult,
      };
    }

  const parsedJson = extractLastJsonObject(stdout);
  if (parsedJson) {
    return {
      handled: true,
      resultText: JSON.stringify(parsedJson),
    };
  }

  const outputFiles = Array.from(
    stdout.matchAll(/output\/final\/[^\s"'`]+/g),
    (match) => match[0],
  );
  if (outputFiles.length > 0) {
    return {
      handled: true,
      resultText: JSON.stringify({
        confidence: 1.0,
        output_files: outputFiles,
        stdout: stdout.slice(0, 2000),
      }),
    };
  }

  return {
    handled: true,
    resultText: stdout || textResult,
  };
}

/**
 * Run a single Claude query and stream results via writeOutput.
 */
async function runClaudeQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<QueryResult> {
  const stream = new MessageStream();
  stream.push(prompt);

  // Poll IPC for follow-up messages and _close sentinel during the query.
  let ipcPolling = true;
  let closedDuringQuery = false;
  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query, ending stream');
      closedDuringQuery = true;
      stream.end();
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    for (const text of messages) {
      log(`Piping IPC message into active query (${text.length} chars)`);
      stream.push(text);
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };
  setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;

  // Load global CLAUDE.md as additional system context (shared across all groups).
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }

  // Discover additional directories mounted at /workspace/extra/*.
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  for await (const message of query({
    prompt: stream,
    options: {
      cwd: '/workspace/group',
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      resume: sessionId,
      resumeSessionAt: resumeAt,
      systemPrompt: globalClaudeMd
        ? {
            type: 'preset' as const,
            preset: 'claude_code' as const,
            append: globalClaudeMd,
          }
        : undefined,
      allowedTools: [
        'Bash',
        'Read',
        'Write',
        'Edit',
        'Glob',
        'Grep',
        'WebSearch',
        'WebFetch',
        'Task',
        'TaskOutput',
        'TaskStop',
        'TeamCreate',
        'TeamDelete',
        'SendMessage',
        'TodoWrite',
        'ToolSearch',
        'Skill',
        'NotebookEdit',
        'mcp__nanoclaw__*',
      ],
      env: sdkEnv,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: {
        nanoclaw: {
          command: 'node',
          args: [mcpServerPath],
          env: {
            NANOCLAW_CHAT_JID: containerInput.chatJid,
            NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
            NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
          },
        },
      },
      hooks: {
        PreCompact: [
          { hooks: [createPreCompactHook(containerInput.assistantName)] },
        ],
        PreToolUse: [{ matcher: 'Bash', hooks: [createSanitizeBashHook()] }],
      },
    },
  })) {
    messageCount++;
    const msgType =
      message.type === 'system'
        ? `system/${(message as { subtype?: string }).subtype}`
        : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
    }

    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }

    if (
      message.type === 'system' &&
      (message as { subtype?: string }).subtype === 'task_notification'
    ) {
      const tn = message as {
        task_id: string;
        status: string;
        summary: string;
      };
      log(
        `Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`,
      );
    }

    if (message.type === 'result') {
      resultCount++;
      const textResult =
        'result' in message ? (message as { result?: string }).result : null;
      log(
        `Result #${resultCount}: subtype=${message.subtype}${
          textResult ? ` text=${textResult.slice(0, 200)}` : ''
        }`,
      );
      writeOutput({
        status: 'success',
        result: textResult || null,
        newSessionId,
      });
    }
  }

  ipcPolling = false;
  log(
    `Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${
      lastAssistantUuid || 'none'
    }, closedDuringQuery: ${closedDuringQuery}`,
  );
  return { newSessionId, lastAssistantUuid, closedDuringQuery };
}

/**
 * Run a single Gemini query and emit one result.
 * This initial implementation supports chat continuity, but not Claude-specific
 * local tool orchestration (Bash/Read/Edit/TeamCreate) yet.
 */
async function runGeminiQuery(
  prompt: string,
  sessionId: string | undefined,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<QueryResult> {
  if (shouldClose()) {
    log('Close sentinel detected before Gemini query');
    return { newSessionId: sessionId, closedDuringQuery: true };
  }

  const apiKey = getGeminiApiKey(sdkEnv);

  const model = sdkEnv.GEMINI_MODEL || 'gemini-2.0-flash';
  const { sessionId: effectiveSessionId, state } = loadGeminiSession(sessionId);
  const systemInstruction = buildGeminiSystemInstruction(containerInput);

  const pending = drainIpcInput();
  const combinedPrompt =
    pending.length > 0 ? `${prompt}\n${pending.join('\n')}` : prompt;

  const contents: GeminiPromptContent[] = [
    ...state.history.map((message) => ({
      role: message.role,
      parts: [{ text: message.text }],
    })),
    { role: 'user', parts: [{ text: combinedPrompt }] },
  ];

  let client: GoogleGenAI;
  if (apiKey) {
    client = new GoogleGenAI({ apiKey });
    log(`Gemini mode: Google GenAI SDK model=${model} auth=api_key`);
  } else {
    const project = getGeminiVertexProjectId(sdkEnv);
    const location = getGeminiVertexLocation(sdkEnv);
    client = new GoogleGenAI({
      vertexai: true,
      project,
      location,
    });
    log(
      `Gemini mode: Google GenAI SDK model=${model} auth=vertex_adc project=${project} location=${location}`,
    );
  }

  let response: unknown;
  try {
    response = await client.models.generateContent({
      model,
      contents,
      config: {
        systemInstruction,
      },
    });
  } catch (error) {
    const msg =
      error instanceof Error ? error.message : `Unknown Gemini SDK error: ${String(error)}`;
    throw new Error(msg);
  }

  const sdkText =
    response && typeof response === 'object' && 'text' in response
      ? String((response as { text?: unknown }).text || '').trim()
      : '';
  let textResult = sdkText || normalizeGeminiText(response as GeminiGenerateResponse);
  if (!textResult) {
    throw new Error('Gemini returned no text output');
  }

  let executed: ExecutedGeminiResult | null = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      executed = await maybeExecuteGeminiCodeResult(textResult, client, model, contents, systemInstruction);
      if (executed.handled) {
        textResult = executed.resultText;
      }
      break;
    } catch (error) {
      if (error instanceof GeminiTemplateScriptError && attempt === 0) {
        log(`Gemini Python looked templated (${error.marker}); requesting a corrected script once.`);
        textResult = await requestGeminiPythonCorrection(client, model, contents, systemInstruction, error.marker);
        if (!textResult) {
          throw new Error('Gemini correction retry returned no text output');
        }
        continue;
      }
      throw error;
    }
  }
  if (executed?.handled) {
    textResult = executed.resultText;
  }

  const lastAssistantUuid = `gemini-${Date.now().toString(36)}`;
  state.history.push({ role: 'user', text: combinedPrompt });
  state.history.push({ role: 'model', text: textResult });
  state.lastAssistantUuid = lastAssistantUuid;
  saveGeminiSession(effectiveSessionId, state);

  writeOutput({
    status: 'success',
    result: textResult,
    newSessionId: effectiveSessionId,
  });

  return {
    newSessionId: effectiveSessionId,
    lastAssistantUuid,
    closedDuringQuery: false,
  };
}

async function runQuery(
  provider: Provider,
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<QueryResult> {
  if (provider === 'gemini') {
    return runGeminiQuery(prompt, sessionId, containerInput, sdkEnv);
  }

  return runClaudeQuery(
    prompt,
    sessionId,
    mcpServerPath,
    containerInput,
    sdkEnv,
    resumeAt,
  );
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    // Delete the temp file the entrypoint wrote - it contains secrets.
    try {
      fs.unlinkSync('/tmp/input.json');
    } catch {
      // May not exist.
    }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`,
    });
    process.exit(1);
  }

  // Build provider env: merge secrets into process.env for the provider only.
  // Secrets never touch process.env itself, so Bash subprocesses cannot see them.
  const sdkEnv: Record<string, string | undefined> = { ...process.env };
  for (const [key, value] of Object.entries(containerInput.secrets || {})) {
    sdkEnv[key] = value;
  }

  const provider = resolveProvider(sdkEnv);
  log(`Using provider: ${provider}`);
  logProviderSecretPreview(provider, sdkEnv);

  if (provider === 'gemini') {
    log(
      'Gemini provider is active. Claude-specific local tools and agent swarms are not implemented yet in this provider.',
    );
  }

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs.
  try {
    fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
  } catch {
    // Ignore missing sentinel.
  }

  // Build initial prompt (drain any pending IPC messages too).
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += `\n${pending.join('\n')}`;
  }

  // Query loop: run query -> wait for IPC message -> run new query -> repeat.
  let resumeAt: string | undefined;
  try {
    while (true) {
      log(
        `Starting query (session: ${sessionId || 'new'}, resumeAt: ${resumeAt || 'latest'})...`,
      );

      const queryResult = await runQuery(
        provider,
        prompt,
        sessionId,
        mcpServerPath,
        containerInput,
        sdkEnv,
        resumeAt,
      );
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }
      if (queryResult.lastAssistantUuid) {
        resumeAt = queryResult.lastAssistantUuid;
      }

      // If _close was consumed during the query, exit immediately.
      // Do not emit a session-update marker (it would reset the host's
      // idle timer and cause a 30-min delay before the next _close).
      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      // Marathon invokes the Gemini runner as a single-turn batch executor.
      // Once a result has been emitted, do not fall back into the interactive
      // IPC loop or the container will idle indefinitely waiting for a follow-up
      // message that never comes.
      if (provider === 'gemini' && containerInput.groupFolder === 'marathon') {
        log('Gemini marathon query completed, exiting single-turn runner');
        break;
      }

      // Emit session update so host can track it.
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');

      // Wait for the next message or _close sentinel.
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage,
    });
    process.exit(1);
  }
}

main();

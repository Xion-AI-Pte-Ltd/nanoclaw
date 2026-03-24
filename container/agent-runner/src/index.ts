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
import { spawn } from 'child_process';
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

type Provider = 'claude' | 'gemini';

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

function useVertexGemini(
  sdkEnv: Record<string, string | undefined>,
  apiKey: string,
): boolean {
  if (envEnabled(sdkEnv.GEMINI_USE_VERTEX)) return true;
  if (envEnabled(sdkEnv.GOOGLE_GENAI_USE_VERTEXAI)) return true;
  return apiKey.length === 0;
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
    const keyName = sdkEnv.GEMINI_API_KEY ? 'GEMINI_API_KEY' : 'GOOGLE_API_KEY';
    const key = getGeminiApiKey(sdkEnv);
    if (key) {
      log(`Auth debug: ${keyName} prefix=${maskedPrefix(key)}`);
      return;
    }
    if (envEnabled(sdkEnv.GEMINI_USE_VERTEX) || envEnabled(sdkEnv.GOOGLE_GENAI_USE_VERTEXAI)) {
      log('Auth debug: no Gemini API key; Vertex mode requested via env');
    } else {
      log('Auth debug: no Gemini API key; will attempt Vertex ADC fallback');
    }
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
    'Be concise, accurate, and explicit about limitations.';
  if (!memory) return base;

  return `${base}\n\nUse this persistent memory context if relevant:\n\n${memory}`;
}

function extractFencedBlock(text: string, language: string): string | null {
  const pattern = new RegExp(String.raw`\`\`\`${language}\n([\s\S]*?)\n\`\`\``, 'i');
  const match = text.match(pattern);
  return match ? match[1].trim() : null;
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
): Promise<{ stdout: string; stderr: string; exitCode: number | null }> {
  const tmpDir = '/workspace/group/.nanoclaw/runtime';
  fs.mkdirSync(tmpDir, { recursive: true });
  const scriptPath = path.join(tmpDir, `gemini_exec_${Date.now().toString(36)}.py`);
  fs.writeFileSync(scriptPath, scriptSource);

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

async function maybeExecuteGeminiCodeResult(
  textResult: string,
): Promise<ExecutedGeminiResult> {
  const pythonBlock = extractFencedBlock(textResult, 'python');
  if (!pythonBlock) {
    return { handled: false, resultText: textResult };
  }

  log('Gemini returned fenced Python; executing script in workspace.');
  const execution = await executePythonScript(pythonBlock);
  const stdout = execution.stdout.trim();
  const stderr = execution.stderr.trim();

  if (execution.exitCode !== 0) {
    const errorText = stderr || stdout || `Python exited with code ${execution.exitCode}`;
    throw new Error(`Gemini-generated Python failed: ${errorText}`);
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

  const contents = [
    ...state.history.map((message) => ({
      role: message.role,
      parts: [{ text: message.text }],
    })),
    { role: 'user', parts: [{ text: combinedPrompt }] },
  ];

  let client: GoogleGenAI;
  if (useVertexGemini(sdkEnv, apiKey)) {
    const projectId = await getVertexProjectId(sdkEnv);
    const location =
      sdkEnv.VERTEX_LOCATION || sdkEnv.GOOGLE_CLOUD_LOCATION || 'us-central1';
    log(
      `Gemini mode: Vertex AI (${location}) project=${projectId} model=${model}`,
    );
    client = new GoogleGenAI({
      vertexai: true,
      project: projectId,
      location,
    });
  } else {
    client = new GoogleGenAI({ apiKey });
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

  const executed = await maybeExecuteGeminiCodeResult(textResult);
  if (executed.handled) {
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

import { FunctionCallingConfigMode } from '@google/genai';

/** Leave room for both reasoning and the generated runnable program. */
export function generationConfig(systemInstruction: string) {
  return {
    systemInstruction: systemInstruction + '\nNo function-calling tools are registered for this provider. '
      + 'Respond with text. For code execution return one fenced runnable Python program; '
      + 'the runner executes it after your reply. Never emit a tool or function call.',
    maxOutputTokens: 16384,
    toolConfig: { functionCallingConfig: { mode: FunctionCallingConfigMode.NONE } },
  };
}

export function emptyResponseReason(response: unknown): string {
  const value = response as { candidates?: { finishReason?: string }[]; promptFeedback?: { blockReason?: string } };
  return JSON.stringify({
    finishReasons: value?.candidates?.map((candidate) => candidate.finishReason) || [],
    blockReason: value?.promptFeedback?.blockReason || null,
  });
}

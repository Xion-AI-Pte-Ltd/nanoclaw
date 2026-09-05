import { FunctionCallingConfigMode, Type } from '@google/genai';

/** Leave room for both reasoning and the generated runnable program. */
export function generationConfig(systemInstruction: string, structuredAnswer = false) {
  return {
    systemInstruction: systemInstruction + '\nNo function-calling tools are registered for this provider. '
      + 'Respond with text. For code execution return one fenced runnable Python program; '
      + 'the runner executes it after your reply. Never emit a tool or function call.'
      + (structuredAnswer ? '\nReturn JSON with an answer string containing your complete response, including the runnable Python code fence when code is needed.' : ''),
    maxOutputTokens: 16384,
    toolConfig: { functionCallingConfig: { mode: FunctionCallingConfigMode.NONE } },
    ...(structuredAnswer ? {
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: { answer: { type: Type.STRING } },
        required: ['answer'],
      },
    } : {}),
  };
}

/** Fail closed: never execute a truncated envelope or interpret a function call. */
export function decodeGenerationText(text: string, structuredAnswer: boolean): string {
  if (!structuredAnswer) return text;
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error('Gemini returned an invalid structured answer');
  }
  if (!value || typeof value !== 'object' || !('answer' in value)
      || typeof value.answer !== 'string' || !value.answer.trim()) {
    throw new Error('Gemini structured answer is missing non-empty answer text');
  }
  return value.answer.trim();
}

export function emptyResponseReason(response: unknown): string {
  const value = response as { candidates?: { finishReason?: string }[]; promptFeedback?: { blockReason?: string } };
  return JSON.stringify({
    finishReasons: value?.candidates?.map((candidate) => candidate.finishReason) || [],
    blockReason: value?.promptFeedback?.blockReason || null,
  });
}

/** Leave room for both reasoning and the generated runnable program. */
export function generationConfig(systemInstruction: string) {
  return { systemInstruction, maxOutputTokens: 16384 };
}

export function emptyResponseReason(response: unknown): string {
  const value = response as { candidates?: { finishReason?: string }[]; promptFeedback?: { blockReason?: string } };
  return JSON.stringify({
    finishReasons: value?.candidates?.map((candidate) => candidate.finishReason) || [],
    blockReason: value?.promptFeedback?.blockReason || null,
  });
}

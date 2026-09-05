import assert from 'node:assert/strict';
import test from 'node:test';
import { generationConfig, emptyResponseReason, decodeGenerationText } from './gemini-generation.js';

test('generation reserves output space explicitly', () => {
  const config = generationConfig('instructions');
  assert.equal(config.maxOutputTokens, 16384);
  assert.equal(config.toolConfig.functionCallingConfig.mode, 'NONE');
  assert.match(config.systemInstruction, /Never emit a tool or function call/);
});

test('batch response is schema-bound text, decoded before execution', () => {
  const config = generationConfig('instructions', true);
  assert.equal(config.responseMimeType, 'application/json');
  assert.deepEqual(config.responseSchema?.required, ['answer']);
  const code = '```python\nprint("done")\n```';
  assert.equal(decodeGenerationText(JSON.stringify({ answer: code }), true), code);
  assert.equal(decodeGenerationText(code, false), code);
  assert.equal(generationConfig('chat').responseSchema, undefined);
});

test('invalid batch envelopes fail closed without exposing content', () => {
  for (const value of ['private truncated', '{}', 'null', '{"answer":{}}', '{"answer":" "}']) {
    assert.throws(() => decodeGenerationText(value, true), /Gemini.*answer/);
  }
});

test('empty output exposes finish reasons without leaking prompt data', () => {
  assert.equal(emptyResponseReason({ candidates: [{ finishReason: 'MAX_TOKENS' }], prompt: 'private' }),
    '{"finishReasons":["MAX_TOKENS"],"blockReason":null}');
  assert.equal(emptyResponseReason(null), '{"finishReasons":[],"blockReason":null}');
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { generationConfig, emptyResponseReason } from './gemini-generation.js';

test('generation reserves output space explicitly', () => {
  const config = generationConfig('instructions');
  assert.equal(config.maxOutputTokens, 16384);
  assert.equal(config.toolConfig.functionCallingConfig.mode, 'NONE');
  assert.match(config.systemInstruction, /Never emit a tool or function call/);
});

test('empty output exposes finish reasons without leaking prompt data', () => {
  assert.equal(emptyResponseReason({ candidates: [{ finishReason: 'MAX_TOKENS' }], prompt: 'private' }),
    '{"finishReasons":["MAX_TOKENS"],"blockReason":null}');
  assert.equal(emptyResponseReason(null), '{"finishReasons":[],"blockReason":null}');
});

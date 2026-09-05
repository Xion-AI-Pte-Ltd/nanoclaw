import assert from 'node:assert/strict';
import test from 'node:test';
import { generationConfig, emptyResponseReason } from './gemini-generation.js';

test('generation reserves output space explicitly', () => {
  assert.deepEqual(generationConfig('instructions'), { systemInstruction: 'instructions', maxOutputTokens: 16384 });
});

test('empty output exposes finish reasons without leaking prompt data', () => {
  assert.equal(emptyResponseReason({ candidates: [{ finishReason: 'MAX_TOKENS' }], prompt: 'private' }),
    '{"finishReasons":["MAX_TOKENS"],"blockReason":null}');
  assert.equal(emptyResponseReason(null), '{"finishReasons":[],"blockReason":null}');
});

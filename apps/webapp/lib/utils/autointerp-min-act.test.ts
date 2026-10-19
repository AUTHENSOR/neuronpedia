import { describe, expect, it } from 'vitest';
import { flipToMostNegative } from './autointerp';

const row = (id: string, values: number[]) => ({ id, values });

describe('flipToMostNegative', () => {
  it('orders by the most negative value and negates the values', () => {
    const out = flipToMostNegative([row('a', [0, -2, 1]), row('b', [-9, 0]), row('c', [3, 4])], 10);
    expect(out.map((r) => r.id)).toEqual(['b', 'a']);
    expect(out[0].values).toEqual([9, -0]);
    expect(out[1].values.indexOf(Math.max(...out[1].values))).toBe(1);
  });

  it('keeps only examples below zero, up to count', () => {
    expect(flipToMostNegative([row('a', [0, 1]), row('b', [])], 10)).toEqual([]);
    expect(flipToMostNegative([row('a', [-1]), row('b', [-3]), row('c', [-2])], 2).map((r) => r.id)).toEqual([
      'b',
      'c',
    ]);
  });

  it('does not change the input rows', () => {
    const input = [row('a', [-1, 2])];
    flipToMostNegative(input, 10);
    expect(input[0].values).toEqual([-1, 2]);
  });
});

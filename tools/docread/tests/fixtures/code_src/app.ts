import { readFileSync } from "fs";

export interface Plot {
  name: string;
  area: number;
}

export const area = (p: Plot): number => p.area * 2;

const label = async (p: Plot) => {
  return p.name.toUpperCase();
};

export default class Garden {
  constructor(private plots: Plot[]) {}

  total(): number {
    return this.plots.reduce((s, p) => s + p.area, 0);
  }
}

export function load(path: string): Plot[] {
  return JSON.parse(readFileSync(path, "utf8"));
}

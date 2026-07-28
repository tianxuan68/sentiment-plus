import { ref } from 'vue';
import type { AnalyzeResult } from '../review.api';

const STORAGE_KEY = 'sentiment_review_history_v1';
const MAX_ITEMS = 12;

export interface ReviewHistoryItem extends AnalyzeResult {
  id: string;
  text: string;
  analyzed_at: string;
}

function readStorage(): ReviewHistoryItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeStorage(items: ReviewHistoryItem[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
}

export function useReviewHistory() {
  const history = ref<ReviewHistoryItem[]>(readStorage());

  function persist(items: ReviewHistoryItem[]) {
    history.value = items;
    writeStorage(items);
  }

  function addEntry(text: string, result: AnalyzeResult) {
    const entry: ReviewHistoryItem = {
      ...result,
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      text,
      analyzed_at: new Date().toISOString(),
    };
    persist([entry, ...history.value.filter((h) => h.text !== text)].slice(0, MAX_ITEMS));
  }

  function removeEntry(id: string) {
    persist(history.value.filter((h) => h.id !== id));
  }

  function clearHistory() {
    persist([]);
  }

  return { history, addEntry, removeEntry, clearHistory };
}

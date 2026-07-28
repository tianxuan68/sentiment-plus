import { defHttp } from '/@/utils/http/axios';

enum Api {
  analyze = '/sys/review/analyze',
  analyzeBatch = '/sys/review/analyze/batch',
  polarity = '/sys/review/stats/polarity',
  trend = '/sys/review/stats/trend',
  summary = '/sys/review/stats/summary',
  prosCons = '/sys/review/pros-cons',
  aiStatus = '/sys/review/ai/status',
  modelsCompare = '/sys/review/models/compare',
  keywordsCompare = '/sys/review/keywords/compare',
  rating = '/sys/review/stats/rating',
}

export interface AnalyzeParams {
  text: string;
  top_n?: number;
}

export interface BatchAnalyzeParams {
  texts: string[];
  top_n?: number;
}

export interface SentimentResult {
  label: number;
  label_text: string;
  prob: number;
}

export interface KeywordItem {
  word: string;
  score: number;
}

export interface AspectItem {
  aspect: string;
  polarity: number;
  polarity_text: string;
  prob: number;
}

export interface AspectEntity {
  aspect: string;
  text: string;
  start: number;
  end: number;
  polarity: number;
  polarity_text: string;
}

export interface AnalyzeResult {
  sentiment: SentimentResult;
  keywords: KeywordItem[];
  aspects: AspectItem[];
  entities?: AspectEntity[];
  active_model?: string;
  source: string;
  latency_ms?: number;
  text_preview?: string;
}

export interface BatchAnalyzeResult {
  items: AnalyzeResult[];
  total: number;
  avg_latency_ms?: number;
}

export interface PolarityStat {
  positive: number;
  negative: number;
  total: number;
  positive_ratio: number;
  negative_ratio: number;
}

export interface TrendPoint {
  name: string;
  value: number;
  positive?: number;
  negative?: number;
}

export interface ProsConsItem {
  word: string;
  score: number;
}

export interface ProsConsResult {
  pros: ProsConsItem[];
  cons: ProsConsItem[];
  source: string;
}

export interface DashboardSummary {
  satisfaction_score: number;
  avg_rating: number;
  review_count: number;
  positive_ratio: number;
  trend_direction: 'up' | 'down' | 'stable';
  source: string;
}

export interface AiStatusResult {
  mock_enabled: boolean;
  ai_base_url: string;
  ai_reachable: boolean;
  ai_latency_ms?: number;
  mode: 'mock' | 'live' | 'degraded';
}

export interface ModelMetric {
  model: string;
  acc: number;
  f1: number;
  owner: string;
}

export interface ModelCompareResult {
  metrics: ModelMetric[];
  best_model: string;
  bilstm_vs_baseline_f1_gain: number;
  source: string;
}

export interface KeywordCompareItem {
  word: string;
  keybert_score: number;
  tfidf_score: number;
}

export interface KeywordCompareResult {
  keybert: KeywordItem[];
  tfidf: KeywordItem[];
  overlap: KeywordCompareItem[];
  keybert_unique_count: number;
  source: string;
}

export interface RatingBucket {
  rating: number;
  count: number;
  ratio: number;
}

export interface RatingDistribution {
  buckets: RatingBucket[];
  avg_rating: number;
  total: number;
  source: string;
}

export const analyzeReview = (params: AnalyzeParams) =>
  defHttp.post<AnalyzeResult>({ url: Api.analyze, params });

export const analyzeBatch = (params: BatchAnalyzeParams) =>
  defHttp.post<BatchAnalyzeResult>({ url: Api.analyzeBatch, params });

export const getPolarityStats = () => defHttp.get<PolarityStat>({ url: Api.polarity });

export const getTrendStats = () => defHttp.get<TrendPoint[]>({ url: Api.trend });

export const getDashboardSummary = () => defHttp.get<DashboardSummary>({ url: Api.summary });

export const getProsCons = () => defHttp.get<ProsConsResult>({ url: Api.prosCons });

export const getAiStatus = () => defHttp.get<AiStatusResult>({ url: Api.aiStatus });

export const getModelCompare = () => defHttp.get<ModelCompareResult>({ url: Api.modelsCompare });

export const compareKeywords = (params: AnalyzeParams) =>
  defHttp.post<KeywordCompareResult>({ url: Api.keywordsCompare, params });

export const getRatingDistribution = () => defHttp.get<RatingDistribution>({ url: Api.rating });

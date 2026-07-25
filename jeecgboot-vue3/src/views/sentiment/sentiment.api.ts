import { defHttp } from '/@/utils/http/axios';

enum Api {
  overview = '/sys/sentimentAnalysis/overview',
  predict = '/sys/sentimentAnalysis/predict',
  predictBatch = '/sys/sentimentAnalysis/predictBatch',
  keywords = '/sys/sentimentAnalysis/keywords',
  aspects = '/sys/sentimentAnalysis/aspects',
}

export const getSentimentOverview = () => defHttp.get({ url: Api.overview });
export const predictSentiment = (data: { text: string; model?: string }) =>
  defHttp.post({ url: Api.predict, data });
export const predictSentimentBatch = (data: { texts: string[]; model?: string }) =>
  defHttp.post({ url: Api.predictBatch, data });
export const extractKeywords = (data: { text: string; topN?: number }) =>
  defHttp.post({ url: Api.keywords, data });
export const analyzeAspects = (data: { text: string }) => defHttp.post({ url: Api.aspects, data });

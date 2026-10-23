const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export interface PredictResponse {
  prediction: 'distress' | 'not-distress'
  confidence: number
  code_mix_bucket: 'low' | 'medium' | 'high'
  effective_score: number
}

export interface ExplainResponse {
  shap_top_words: string[]
  shap_scores: number[]
  lime_top_words: string[]
  lime_scores: number[]
  lime_faithfulness_score: number
  shap_faithfulness_score: number
  saved: boolean
  history_id: string | null
}

export interface HistoryItem {
  id: string
  created_at: string
  text: string
  prediction: 'distress' | 'not-distress'
  confidence: number
  code_mix_bucket: 'low' | 'medium' | 'high'
  effective_score: number
  lime_faithfulness_score: number
  shap_faithfulness_score: number
  model_version: string
}

export interface HistoryListResponse {
  items: HistoryItem[]
  total: number
}

export interface HistoryDetail extends HistoryItem {
  shap_top_words: string[]
  shap_scores: number[]
  lime_top_words: string[]
  lime_scores: number[]
}

export interface BucketStats {
  bucket: 'low' | 'medium' | 'high'
  n: number
  lime_avg: number
  shap_avg: number
}

export interface DashboardStatsResponse {
  model_version: string
  buckets: BucketStats[]
}

async function postText<T>(path: string, text: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export function predict(text: string): Promise<PredictResponse> {
  return postText<PredictResponse>('/predict', text)
}

export function explain(text: string): Promise<ExplainResponse> {
  return postText<ExplainResponse>('/explain', text)
}

export async function fetchDashboardStats(): Promise<DashboardStatsResponse> {
  const res = await fetch(`${API_BASE_URL}/dashboard-stats`)
  if (!res.ok) {
    throw new Error(`/dashboard-stats failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<DashboardStatsResponse>
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`)
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export function fetchHistory(limit: number, skip: number): Promise<HistoryListResponse> {
  return getJson<HistoryListResponse>(`/history?limit=${limit}&skip=${skip}`)
}

export function fetchHistoryItem(id: string): Promise<HistoryDetail> {
  return getJson<HistoryDetail>(`/history/${id}`)
}

export async function deleteHistoryItem(id: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/history/${id}`, { method: 'DELETE' })
  if (!res.ok) {
    throw new Error(`DELETE /history/${id} failed: ${res.status} ${res.statusText}`)
  }
}

export async function clearHistory(): Promise<{ deleted: number }> {
  const res = await fetch(`${API_BASE_URL}/history`, { method: 'DELETE' })
  if (!res.ok) {
    throw new Error(`DELETE /history failed: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<{ deleted: number }>
}

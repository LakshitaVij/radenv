import { useState } from 'react'
import { theme } from './theme.js'

const API_BASE = 'http://localhost:7071/api'

// "live: true" = actually wired up in the backend (real API key present).
// The rest are real entries in run_model_eval.py's READERS (usable from the
// CLI) but not yet exposed here - shown disabled, not hidden, so it's clear
// they're coming, not forgotten.
const MODEL_OPTIONS = [
  { value: 'medgemma', label: 'MedGemma', live: true },
  { value: 'gemini-3.6-flash', label: 'Gemini 3.6 Flash', live: true },
  { value: 'gemini-3.5-flash', label: 'Gemini 3.5 Flash', live: true },
  { value: 'nemotron', label: 'Nemotron', live: false },
  { value: 'qwen', label: 'Qwen', live: false },
  { value: 'kimi', label: 'Kimi', live: false },
  { value: 'deepseek', label: 'DeepSeek', live: false },
  { value: '__add_own__', label: '+ Add your own model (coming soon)', live: false },
]

export default function ModelTesting({ patientId, visitFolder, hasGold, loadingGold }) {
  const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS.find((m) => m.live).value)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const selected = MODEL_OPTIONS.find((m) => m.value === selectedModel)

  const runModel = async () => {
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/model-eval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          visit_folder: visitFolder,
          model: selectedModel,
        }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
      } else {
        setResult(data)
      }
    } catch (err) {
      setError('Model run failed: ' + err.message)
    } finally {
      setRunning(false)
    }
  }

  const cardStyle = {
    marginTop: '1.5rem',
    padding: '1.25rem',
    borderRadius: '12px',
    backgroundColor: theme.card,
    boxShadow: theme.shadow,
    border: `1px solid ${theme.border}`,
  }

  return (
    <div style={cardStyle}>
      <h3 style={{ marginTop: 0, color: theme.text }}>Model Testing</h3>
      <p style={{ fontSize: '0.85rem', color: theme.textMuted, marginTop: '-0.5rem' }}>
        Runs a real model on this case's image, then scores its actual output against
        gold. This is testing a MODEL's performance - not your own notes (see the
        Note Entry section above for that).
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        <label style={{ fontSize: '0.9rem', color: theme.text }}>
          Model:{' '}
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{
              padding: '0.4rem 0.6rem',
              borderRadius: '6px',
              border: `1px solid ${theme.border}`,
              backgroundColor: theme.card,
            }}
          >
            {MODEL_OPTIONS.map((m) => (
              <option key={m.value} value={m.value} disabled={!m.live}>
                {m.label}{!m.live ? ' (coming soon)' : ''}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={runModel}
          disabled={running || !hasGold || loadingGold || !selected.live}
          title={!hasGold ? 'No gold answer available for this case' : ''}
          style={{
            padding: '0.6rem 1.2rem',
            borderRadius: '8px',
            border: 'none',
            backgroundColor: theme.mint,
            color: '#fff',
            fontWeight: 600,
            cursor: 'pointer',
            opacity: (running || !hasGold || loadingGold || !selected.live) ? 0.5 : 1,
          }}
        >
          {running ? 'Running model...' : 'Run Model & Score'}
        </button>

        {!hasGold && !loadingGold && (
          <span style={{ fontSize: '0.8rem', color: theme.textMuted }}>
            No gold available for this case.
          </span>
        )}
      </div>

      {error && <p style={{ color: theme.danger, marginTop: '1rem' }}>{error}</p>}

      {result && (
        <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{
              flex: '1 1 300px',
              padding: '1rem',
              borderRadius: '8px',
              backgroundColor: theme.cardAlt,
            }}>
              <div style={{ fontSize: '0.8rem', color: theme.textMuted, marginBottom: '0.5rem' }}>
                {result.model}'s generated report
              </div>
              <div><strong>Findings:</strong> {result.model_generated_findings}</div>
              <div style={{ marginTop: '0.5rem' }}>
                <strong>Impression:</strong> {result.model_generated_impressions}
              </div>
            </div>

            <div style={{
              flex: '1 1 300px',
              padding: '1rem',
              borderRadius: '8px',
              backgroundColor: theme.cardAlt,
              border: `1px solid ${theme.mint}`,
            }}>
              <div style={{ fontSize: '0.8rem', color: theme.mintDark, marginBottom: '0.5rem', fontWeight: 600 }}>
                Gold (clinician-authored)
              </div>
              <div><strong>Findings:</strong> {result.gold_findings}</div>
              <div style={{ marginTop: '0.5rem' }}>
                <strong>Impression:</strong> {result.gold_impressions}
              </div>
            </div>
          </div>

          <div style={{
            padding: '1rem',
            borderRadius: '8px',
            backgroundColor: theme.mintLight,
          }}>
            <div style={{ fontWeight: 600, marginBottom: '0.5rem', color: theme.text }}>
              Score vs. gold
            </div>
            <div>Findings score: {result.findings_score?.toFixed(2)}</div>
            <div>Impressions score: {result.impressions_score?.toFixed(2)}</div>
            <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: theme.textMuted }}>
              {result.findings_reasoning}
            </div>
            <div style={{ fontSize: '0.85rem', color: theme.textMuted }}>
              {result.impressions_reasoning}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

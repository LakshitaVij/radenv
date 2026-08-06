import { useState, useRef } from 'react'
import { theme } from './theme.js'

const API_BASE = 'http://localhost:7071/api'

function parseTranscript(text) {
  // Split transcript into findings / impressions / follow_up based on spoken markers.
  // Doctor says things like: "The findings are: ... The impressions are: ... The follow up is: ..."
  const lower = text.toLowerCase()

  const markers = [
    { key: 'findings', patterns: ['findings are', 'finding is', 'findings:'] },
    { key: 'impressions', patterns: ['impressions are', 'impression is', 'impressions:'] },
    { key: 'follow_up', patterns: ['follow up is', 'follow-up is', 'follow up:', 'follow-up:'] },
  ]

  // Find index positions of each marker in the lowercased text
  const positions = []
  for (const { key, patterns } of markers) {
    for (const pattern of patterns) {
      const idx = lower.indexOf(pattern)
      if (idx !== -1) {
        positions.push({ key, idx, end: idx + pattern.length })
        break // only take first match per marker
      }
    }
  }

  positions.sort((a, b) => a.idx - b.idx)

  const result = { findings: '', impressions: '', follow_up: '' }

  for (let i = 0; i < positions.length; i++) {
    const start = positions[i].end
    const end = i + 1 < positions.length ? positions[i + 1].idx : text.length
    result[positions[i].key] = text.slice(start, end).trim().replace(/^:\s*/, '')
  }

  // If no markers were found at all, dump everything into findings as a fallback
  if (positions.length === 0) {
    result.findings = text.trim()
  }

  return result
}

// Note: this component is ONLY for the clinician's own note entry. Model
// testing (running a real model and scoring it against gold) lives in its
// own tab (ModelTesting.jsx) - a clinician's own entry is a candidate GOLD
// answer, not something to be scored against gold, so that machinery
// intentionally doesn't live here anymore.
export default function RecordFindings({ patientId, visitFolder, seriesName, onSaved }) {
  const [mode, setMode] = useState('audio') // 'audio' | 'type'
  const [typedText, setTypedText] = useState('')
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [findings, setFindings] = useState('')
  const [impressions, setImpressions] = useState('')
  const [followUp, setFollowUp] = useState('')
  const [reviewer, setReviewer] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null)
  const [error, setError] = useState(null)
  const [saveAsGold, setSaveAsGold] = useState(false)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  const startRecording = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach((track) => track.stop())
        await transcribeAudio(audioBlob)
      }

      mediaRecorder.start()
      setRecording(true)
    } catch (err) {
      console.error('Mic access error:', err)
      setError('Could not access microphone: ' + err.message)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
    }
  }

  const transcribeAudio = async (audioBlob) => {
    setTranscribing(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/webm' },
        body: audioBlob,
      })
      const data = await res.json()

      if (data.error) {
        setError(data.error)
      } else {
        const parsed = parseTranscript(data.text)
        setFindings(parsed.findings)
        setImpressions(parsed.impressions)
        setFollowUp(parsed.follow_up)
      }
    } catch (err) {
      console.error('Transcription error:', err)
      setError('Transcription failed: ' + err.message)
    } finally {
      setTranscribing(false)
    }
  }

  const handleParseTyped = () => {
    setError(null)
    const parsed = parseTranscript(typedText)
    setFindings(parsed.findings)
    setImpressions(parsed.impressions)
    setFollowUp(parsed.follow_up)
  }

  const handleSave = async () => {
    setSaving(true)
    setSaveStatus(null)
    try {
      const res = await fetch(`${API_BASE}/findings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          visit_folder: visitFolder,
          series_name: seriesName || '',
          findings,
          impressions,
          follow_up: followUp,
          reviewer: reviewer || 'unknown',
        }),
      })
      const data = await res.json()
      if (data.error) {
        setSaveStatus('Error: ' + data.error)
      } else {
        // "Save as new gold" is a placeholder for now - the entry is saved to
        // the findings table either way, this just changes the confirmation
        // message. Actually writing to the gold data source is a separate,
        // not-yet-built task.
        setSaveStatus(saveAsGold ? 'Saved! (as new gold - placeholder, not yet written to gold data)' : 'Saved!')
        if (onSaved) onSaved(data.entry)
      }
    } catch (err) {
      setSaveStatus('Error: ' + err.message)
    } finally {
      setSaving(false)
    }
  }

  const tabButtonStyle = (active) => ({
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    border: active ? `2px solid ${theme.mint}` : `1px solid ${theme.border}`,
    backgroundColor: active ? theme.mintLight : theme.card,
    color: theme.text,
    fontWeight: active ? 600 : 400,
    cursor: 'pointer',
  })

  const labelStyle = { fontSize: '0.85rem', color: theme.textMuted, fontWeight: 500 }
  const inputStyle = {
    width: '100%',
    marginTop: '0.35rem',
    padding: '0.5rem',
    borderRadius: '6px',
    border: `1px solid ${theme.border}`,
    backgroundColor: theme.card,
    color: theme.text,
    fontFamily: 'inherit',
  }
  const primaryButton = {
    padding: '0.6rem 1.2rem',
    borderRadius: '8px',
    border: 'none',
    backgroundColor: theme.mint,
    color: '#fff',
    fontWeight: 600,
    cursor: 'pointer',
  }

  return (
    <div style={{
      marginTop: '1.5rem',
      padding: '1.25rem',
      borderRadius: '12px',
      backgroundColor: theme.card,
      boxShadow: theme.shadow,
      border: `1px solid ${theme.border}`,
    }}>
      <h3 style={{ marginTop: 0, color: theme.text }}>Note Entry</h3>

      <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem' }}>
        <button onClick={() => setMode('audio')} style={tabButtonStyle(mode === 'audio')}>
          🎙 Audio
        </button>
        <button onClick={() => setMode('type')} style={tabButtonStyle(mode === 'type')}>
          ⌨ Type
        </button>
      </div>

      {mode === 'audio' && (
        <div style={{ marginBottom: '1rem' }}>
          {!recording ? (
            <button onClick={startRecording} disabled={transcribing} style={primaryButton}>
              🎙 Start Recording
            </button>
          ) : (
            <button
              onClick={stopRecording}
              style={{ ...primaryButton, backgroundColor: theme.danger }}
            >
              ⏹ Stop Recording
            </button>
          )}
          {transcribing && <span style={{ marginLeft: '1rem', color: theme.textMuted }}>Transcribing...</span>}
        </div>
      )}

      {mode === 'type' && (
        <div style={{ marginBottom: '1rem' }}>
          <textarea
            value={typedText}
            onChange={(e) => setTypedText(e.target.value)}
            placeholder="Type your dictation, e.g. 'Findings are: ... Impressions are: ... Follow up is: ...'"
            rows={4}
            style={inputStyle}
          />
          <button onClick={handleParseTyped} style={{ ...primaryButton, marginTop: '0.5rem' }}>
            Parse into fields
          </button>
        </div>
      )}

      {error && <p style={{ color: theme.danger }}>{error}</p>}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
        <label style={labelStyle}>
          Findings
          <textarea
            value={findings}
            onChange={(e) => setFindings(e.target.value)}
            rows={3}
            style={inputStyle}
          />
        </label>

        <label style={labelStyle}>
          Impressions
          <textarea
            value={impressions}
            onChange={(e) => setImpressions(e.target.value)}
            rows={3}
            style={inputStyle}
          />
        </label>

        <label style={labelStyle}>
          Follow-up
          <textarea
            value={followUp}
            onChange={(e) => setFollowUp(e.target.value)}
            rows={2}
            style={inputStyle}
          />
        </label>

        <label style={labelStyle}>
          Reviewer name
          <input
            type="text"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
            style={inputStyle}
          />
        </label>

        <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input
            type="checkbox"
            checked={saveAsGold}
            onChange={(e) => setSaveAsGold(e.target.checked)}
          />
          Save as new gold (placeholder - not yet written to gold data)
        </label>

        <button onClick={handleSave} disabled={saving} style={{ ...primaryButton, marginTop: '0.25rem' }}>
          {saving ? 'Saving...' : 'Save Entry'}
        </button>

        {saveStatus && <p style={{ color: theme.mintDark, fontWeight: 500 }}>{saveStatus}</p>}
      </div>
    </div>
  )
}

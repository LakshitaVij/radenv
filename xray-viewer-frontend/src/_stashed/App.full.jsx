import { useState, useEffect } from 'react'
import './App.css'
import RecordFindings from './recording.jsx'
import ModelTesting from './ModelTesting.jsx'
import XrayFullScreenViewer from './XrayFullScreenViewer.jsx'
import { theme } from './theme.js'

const API_BASE = 'http://localhost:7071/api'

function parseVisitFolder(visitFolder) {
  const parts = visitFolder.split('_')
  const dateStr = parts[1]
  if (dateStr && dateStr.length === 8) {
    const year = dateStr.slice(0, 4)
    const month = dateStr.slice(4, 6)
    const day = dateStr.slice(6, 8)
    return `${year}-${month}-${day}`
  }
  return visitFolder
}

function getRawDateFromVisitFolder(visitFolder) {
  // visit_YYYYMMDD_UID -> "YYYYMMDD" (the format Orthanc/QIDO expects for StudyDate)
  const parts = visitFolder.split('_')
  return parts[1] || ''
}

const ORTHANC_QIDO_BASE = 'http://localhost:8043/dicom-web'
const OHIF_BASE = 'http://localhost:3000'

async function findStudyInstanceUID(patientId, rawDate) {
  const url = `${ORTHANC_QIDO_BASE}/studies?PatientID=${encodeURIComponent(patientId)}&StudyDate=${encodeURIComponent(rawDate)}`
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`QIDO query failed: ${res.status}`)
  }
  const results = await res.json()
  if (!results || results.length === 0) {
    return null
  }
  // StudyInstanceUID is DICOM tag 0020,000D
  return results[0]['0020000D']?.Value?.[0] || null
}

// Small style helpers, shared across sections below.
const sectionCard = {
  marginTop: '2rem',
  padding: '1.25rem',
  borderRadius: '12px',
  backgroundColor: theme.card,
  boxShadow: theme.shadow,
  border: `1px solid ${theme.border}`,
}
const sectionHeading = { marginTop: 0, color: theme.text }
const mutedNote = { fontSize: '0.8rem', color: theme.textMuted }

function App() {
  const [patients, setPatients] = useState([])
  const [selectedPatient, setSelectedPatient] = useState('')
  const [visits, setVisits] = useState([])
  const [loadingVisits, setLoadingVisits] = useState(false)
  const [selectedVisit, setSelectedVisit] = useState(null)
  const [studyUid, setStudyUid] = useState(null)
  const [studyUidError, setStudyUidError] = useState(null)
  const [loadingStudyUid, setLoadingStudyUid] = useState(false)

  // Only whether gold EXISTS for this case, never the gold text itself - the
  // text stays server-side, used only inside /api/judge and /api/model-eval's
  // scoring. Shared by both RecordFindings (disables nothing, just informs)
  // and ModelTesting (disables the Run button when there's nothing to score).
  const [hasGold, setHasGold] = useState(false)
  const [loadingGold, setLoadingGold] = useState(false)

  const [ehrMode, setEhrMode] = useState(false)
  const [patientDemographics, setPatientDemographics] = useState({ age: '', sex: '' })
  const [actionPlaceholders, setActionPlaceholders] = useState({ structured_follow_up: [], order_panel: [] })
  const [orderedActions, setOrderedActions] = useState([])
  const [auditTrail, setAuditTrail] = useState([])
  const [loadingAudit, setLoadingAudit] = useState(false)
  const [clinicalVars, setClinicalVars] = useState({
    oxygenSaturation: '', respiratoryRate: '', bloodPressure: '',
    heartRate: '', temperature: '', mentalStatus: '',
  })

  useEffect(() => {
    fetch(`${API_BASE}/patients`)
      .then(res => res.json())
      .then(data => setPatients(data.patients))
      .catch(err => console.error('Failed to fetch patients:', err))

    fetch(`${API_BASE}/ehr/action-placeholders`)
      .then(res => res.json())
      .then(data => setActionPlaceholders(data))
      .catch(err => console.error('Failed to fetch action placeholders:', err))
  }, [])

  useEffect(() => {
    if (!selectedPatient) {
      setVisits([])
      setSelectedVisit(null)
      setPatientDemographics({ age: '', sex: '' })
      setAuditTrail([])
      return
    }
    setLoadingVisits(true)
    setSelectedVisit(null)
    fetch(`${API_BASE}/patient/${selectedPatient}/visits`)
      .then(res => res.json())
      .then(data => {
        setVisits(data.visits || [])
        setPatientDemographics({ age: data.age || '', sex: data.sex || '' })
        setLoadingVisits(false)
      })
      .catch(err => {
        console.error('Failed to fetch visits:', err)
        setLoadingVisits(false)
      })

    setLoadingAudit(true)
    fetch(`${API_BASE}/findings/${selectedPatient}`)
      .then(res => res.json())
      .then(data => {
        setAuditTrail(data.findings || [])
        setLoadingAudit(false)
      })
      .catch(err => {
        console.error('Failed to fetch audit trail:', err)
        setLoadingAudit(false)
      })
  }, [selectedPatient])

  useEffect(() => {
    if (!selectedVisit || !selectedPatient) {
      setStudyUid(null)
      setStudyUidError(null)
      return
    }

    let cancelled = false
    setLoadingStudyUid(true)
    setStudyUidError(null)
    setStudyUid(null)

    const rawDate = getRawDateFromVisitFolder(selectedVisit.visit_folder)

    findStudyInstanceUID(selectedPatient, rawDate)
      .then((uid) => {
        if (cancelled) return
        if (uid) {
          setStudyUid(uid)
        } else {
          setStudyUidError('No matching study found in local Orthanc for this visit date.')
        }
        setLoadingStudyUid(false)
      })
      .catch((err) => {
        if (cancelled) return
        console.error('Failed to find Study UID:', err)
        setStudyUidError('Could not reach local Orthanc/OHIF. Is it running?')
        setLoadingStudyUid(false)
      })

    return () => {
      cancelled = true
    }
  }, [selectedVisit, selectedPatient])

  useEffect(() => {
    if (!selectedVisit || !selectedPatient) {
      setHasGold(false)
      return
    }

    let cancelled = false
    setLoadingGold(true)
    setHasGold(false)

    // Only checks whether gold EXISTS - the endpoint never returns gold text.
    fetch(`${API_BASE}/oracle/${selectedPatient}/${selectedVisit.visit_folder}`)
      .then((res) => res.json())
      .then((data) => {
        if (cancelled) return
        setHasGold(!!data.has_gold)
        setLoadingGold(false)
      })
      .catch((err) => {
        if (cancelled) return
        console.error('Failed to check gold availability:', err)
        setLoadingGold(false)
      })

    return () => {
      cancelled = true
    }
  }, [selectedVisit, selectedPatient])

  return (
    <div style={{
      padding: '2rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      backgroundColor: theme.bg,
      minHeight: '100vh',
      color: theme.text,
    }}>
      <h1 style={{ color: theme.text }}>X-Ray Patient Review</h1>

      <button
        onClick={() => setEhrMode(!ehrMode)}
        style={{
          marginBottom: '1rem',
          padding: '0.5rem 1rem',
          borderRadius: '8px',
          border: ehrMode ? `2px solid ${theme.mint}` : `1px solid ${theme.border}`,
          backgroundColor: ehrMode ? theme.mintLight : theme.card,
          fontWeight: ehrMode ? 600 : 400,
          color: theme.text,
          cursor: 'pointer',
        }}
      >
        {ehrMode ? '✓ Realistic EHR View' : 'Switch to Realistic EHR View'}
      </button>

      <div>
        <label style={{ color: theme.text }}>
          Select Patient:{' '}
          <select
            value={selectedPatient}
            onChange={(e) => setSelectedPatient(e.target.value)}
            style={{
              padding: '0.4rem 0.6rem',
              borderRadius: '6px',
              border: `1px solid ${theme.border}`,
              backgroundColor: theme.card,
            }}
          >
            <option value="">-- Choose a patient --</option>
            {patients.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ marginTop: '1rem' }}>
        <a
          href={`${API_BASE}/export/csv${selectedPatient ? `?patient_id=${selectedPatient}` : ''}`}
          download
        >
          <button style={{
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            border: `1px solid ${theme.border}`,
            backgroundColor: theme.card,
            color: theme.text,
            cursor: 'pointer',
          }}>
            ⬇ Download {selectedPatient ? `${selectedPatient}'s` : 'All'} Findings CSV
          </button>
        </a>
      </div>

      {ehrMode && selectedPatient && (
        <div style={{
          ...sectionCard,
          border: `2px solid ${theme.mint}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <div>
            <strong style={{ fontSize: '1.2rem', color: theme.text }}>{selectedPatient}</strong>
            <div style={{ ...mutedNote, marginTop: '0.25rem' }}>
              Age: {patientDemographics.age || '—'} &nbsp;|&nbsp; Sex: {patientDemographics.sex || '—'}
            </div>
          </div>
          <div style={{
            padding: '0.5rem 1rem',
            border: `1px solid ${theme.border}`,
            borderRadius: '6px',
            fontSize: '0.85rem',
            color: theme.textMuted,
            backgroundColor: theme.cardAlt,
          }}>
            Tier: — (placeholder, no urgency tier data source yet)
          </div>
        </div>
      )}

      {loadingVisits && <p style={{ marginTop: '1rem', color: theme.textMuted }}>Loading visits...</p>}

      {!loadingVisits && visits.length > 0 && (
        <div style={sectionCard}>
          <h2 style={sectionHeading}>Timeline</h2>
          <div style={{
            display: 'flex',
            overflowX: 'auto',
            gap: '1.5rem',
            paddingBottom: '1rem'
          }}>
            {visits.map((visit) => {
              const isSelected = selectedVisit?.visit_folder === visit.visit_folder
              return (
                <div
                  key={visit.visit_folder}
                  onClick={() => setSelectedVisit(visit)}
                  style={{
                    minWidth: '180px',
                    border: isSelected ? `2px solid ${theme.mint}` : `1px solid ${theme.border}`,
                    borderRadius: '8px',
                    padding: '1rem',
                    backgroundColor: isSelected ? theme.mintLight : theme.cardAlt,
                    cursor: 'pointer',
                  }}
                >
                  <strong style={{ color: theme.text }}>{parseVisitFolder(visit.visit_folder)}</strong>
                  <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: theme.textMuted }}>
                    {visit.series.map((s, idx) => (
                      <div key={idx}>
                        {s.series_name.split('_')[0]} ({s.images.length} img)
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {selectedVisit && (
        <XrayFullScreenViewer
          visit={selectedVisit}
          patientId={selectedPatient}
          patientDemographics={patientDemographics}
        />
      )}

      {selectedVisit && (
        <div style={sectionCard}>
          <h2 style={sectionHeading}>Viewer</h2>

          {loadingStudyUid && <p style={{ color: theme.textMuted }}>Looking up study in local PACS...</p>}
          {studyUidError && <p style={{ color: theme.danger }}>{studyUidError}</p>}

          {studyUid && (
            <iframe
              title="OHIF Viewer"
              src={`${OHIF_BASE}/viewer?StudyInstanceUIDs=${studyUid}`}
              style={{
                width: '100%',
                height: '80vh',
                border: `1px solid ${theme.border}`,
                borderRadius: '8px',
                backgroundColor: '#000',
              }}
            />
          )}
        </div>
      )}

      {/* Two clearly separate sections, same page: your own notes vs. testing
          a model's performance. Not the same thing, kept visually distinct. */}
      {selectedVisit && (
        <RecordFindings
          key={`${selectedPatient}-${selectedVisit.visit_folder}`}
          patientId={selectedPatient}
          visitFolder={selectedVisit.visit_folder}
          seriesName={selectedVisit.series[0]?.series_name || ''}
        />
      )}

      {selectedVisit && (
        <ModelTesting
          key={`model-${selectedPatient}-${selectedVisit.visit_folder}`}
          patientId={selectedPatient}
          visitFolder={selectedVisit.visit_folder}
          hasGold={hasGold}
          loadingGold={loadingGold}
        />
      )}

      {ehrMode && selectedVisit && (
        <div style={sectionCard}>
          <h2 style={sectionHeading}>Structured Follow-up Action</h2>
          <p style={mutedNote}>
            5 placeholder picks from the real Action Library - selection doesn't do
            anything yet, pending review.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {actionPlaceholders.structured_follow_up.map((a) => (
              <button
                key={a.action_id}
                title={a.action_id}
                style={{
                  padding: '0.5rem 0.75rem',
                  fontSize: '0.85rem',
                  borderRadius: '6px',
                  border: `1px solid ${theme.border}`,
                  backgroundColor: theme.cardAlt,
                  color: theme.text,
                  cursor: 'pointer',
                }}
              >
                {a.action_name}
              </button>
            ))}
          </div>
        </div>
      )}

      {ehrMode && selectedVisit && (
        <div style={sectionCard}>
          <h2 style={sectionHeading}>Order / Tool-Call Panel</h2>
          <p style={mutedNote}>
            5 placeholder order buttons - clicking simulates "order placed" only, no
            real EHR/ordering system integration exists yet.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {actionPlaceholders.order_panel.map((a) => (
              <button
                key={a.action_id}
                title={a.action_id}
                onClick={() => setOrderedActions([...orderedActions, a.action_id])}
                style={{
                  padding: '0.5rem 0.75rem',
                  fontSize: '0.85rem',
                  borderRadius: '6px',
                  border: `1px solid ${theme.border}`,
                  backgroundColor: theme.cardAlt,
                  color: theme.text,
                  cursor: 'pointer',
                }}
              >
                {a.action_name}
              </button>
            ))}
          </div>
          {orderedActions.length > 0 && (
            <div style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: theme.mintDark }}>
              Simulated orders placed: {orderedActions.join(', ')}
            </div>
          )}
        </div>
      )}

      {ehrMode && selectedVisit && (
        <div style={sectionCard}>
          <h2 style={sectionHeading}>Clinical Variables</h2>
          <p style={mutedNote}>
            Placeholder inputs - no real vitals/labs data exists in this dataset
            (X-ray reports only). Editable for demo purposes only.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', maxWidth: '600px' }}>
            {Object.entries({
              oxygenSaturation: 'O2 Saturation',
              respiratoryRate: 'Respiratory Rate',
              bloodPressure: 'Blood Pressure',
              heartRate: 'Heart Rate',
              temperature: 'Temperature',
              mentalStatus: 'Mental Status',
            }).map(([key, label]) => (
              <label key={key} style={{ fontSize: '0.8rem', color: theme.textMuted }}>
                {label}
                <input
                  type="text"
                  value={clinicalVars[key]}
                  onChange={(e) => setClinicalVars({ ...clinicalVars, [key]: e.target.value })}
                  style={{
                    width: '100%',
                    marginTop: '0.25rem',
                    padding: '0.4rem',
                    borderRadius: '6px',
                    border: `1px solid ${theme.border}`,
                    backgroundColor: theme.card,
                  }}
                />
              </label>
            ))}
          </div>
        </div>
      )}

      {ehrMode && selectedPatient && (
        <div style={sectionCard}>
          <h2 style={sectionHeading}>Audit / History Trail</h2>
          {loadingAudit && <p style={{ color: theme.textMuted }}>Loading history...</p>}
          {!loadingAudit && auditTrail.length === 0 && (
            <p style={{ color: theme.textMuted }}>No prior entries for this patient.</p>
          )}
          {auditTrail.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {auditTrail.map((entry) => (
                <div
                  key={entry.RowKey}
                  style={{
                    border: `1px solid ${theme.border}`,
                    borderRadius: '6px',
                    padding: '0.75rem',
                    backgroundColor: theme.cardAlt,
                    fontSize: '0.85rem',
                  }}
                >
                  <div style={{ color: theme.mintDark, fontWeight: 500 }}>
                    {entry.reviewer} — {entry.visit_folder} — {entry.timestamp}
                  </div>
                  <div style={{ marginTop: '0.25rem' }}><strong>Findings:</strong> {entry.findings}</div>
                  <div><strong>Impression:</strong> {entry.impressions}</div>
                  {entry.follow_up && <div><strong>Follow-up:</strong> {entry.follow_up}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App

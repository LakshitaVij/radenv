import { useState, useEffect } from 'react'
import XrayFullScreenViewer from './XrayFullScreenViewer.jsx'
import { OPENEMR_ENCOUNTER_MAP } from './openemrEncounterMap.js'

const API_BASE = 'http://localhost:7071/api'
const OPENEMR_BASE = 'https://localhost:9300'

// Static PatientID -> OpenEMR internal pid, since all 10 gold-set patients
// were seeded once and their pids are stable (1-10, in seed order) -
// confirmed directly against patient_data. Not derived dynamically since
// there's no live lookup endpoint for this in xray-viewer-backend yet.
const OPENEMR_PID_MAP = {
  GRDN003M0HXJ2YHT: 1,
  GRDN004RP5BFHE0T: 2,
  GRDN006MCYEVYFE8: 3,
  GRDN006VFINFR877: 4,
  GRDN006WOMKG4AVY: 5,
  GRDN007ELLXXY3OJ: 6,
  GRDN00A5ZYWHZY7D: 7,
  GRDN00BE97VCHW4E: 8,
  GRDN00BJ2R8QRO0L: 9,
  GRDN00BKCEOKC7S1: 10,
}

// Bigger than a normal small button - this is the one thing the
// computer-use agent has to click reliably to get from the X-ray viewer
// into OpenEMR, and Gemini kept missing a smaller version by ~90px in
// testing.
const openEmrButtonStyle = {
  padding: '0.6rem 1.1rem',
  backgroundColor: '#1a3a2e',
  color: '#7fd9a8',
  border: '2px solid #2d5a44',
  borderRadius: '5px',
  fontSize: '1.05rem',
  fontWeight: 600,
  cursor: 'pointer',
  textDecoration: 'none',
  display: 'inline-block',
  lineHeight: 1.4,
}

function parseVisitDate(visitFolder) {
  const parts = visitFolder.split('_')
  const dateStr = parts[1]
  if (dateStr && dateStr.length === 8) {
    return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`
  }
  return visitFolder
}

// Stripped down to just: pick a patient, pick a visit, see the X-ray full
// screen. Everything else (timeline cards, thumbnail grid, note entry,
// model testing, the old OHIF iframe) is stashed in ./_stashed/ - not
// deleted, just not part of this page.
export default function App() {
  const [patients, setPatients] = useState([])
  const [selectedPatient, setSelectedPatient] = useState('')
  const [visits, setVisits] = useState([])
  const [selectedVisitFolder, setSelectedVisitFolder] = useState('')
  const [patientDemographics, setPatientDemographics] = useState({ age: '', sex: '' })

  useEffect(() => {
    fetch(`${API_BASE}/patients`)
      .then((res) => res.json())
      .then((data) => setPatients(data.patients || []))
      .catch((err) => console.error('Failed to fetch patients:', err))
  }, [])

  useEffect(() => {
    if (!selectedPatient) {
      setVisits([])
      setSelectedVisitFolder('')
      setPatientDemographics({ age: '', sex: '' })
      return
    }
    setSelectedVisitFolder('')
    fetch(`${API_BASE}/patient/${selectedPatient}/visits`)
      .then((res) => res.json())
      .then((data) => {
        setVisits(data.visits || [])
        setPatientDemographics({ age: data.age || '', sex: data.sex || '' })
        if (data.visits?.length) setSelectedVisitFolder(data.visits[0].visit_folder)
      })
      .catch((err) => console.error('Failed to fetch visits:', err))
  }, [selectedPatient])

  const selectedVisit = visits.find((v) => v.visit_folder === selectedVisitFolder) || null
  // OPENEMR_ENCOUNTER_MAP (real per-visit encounter IDs, generated from
  // OpenEMR's own DB) isn't used for the link right now - deep-linking
  // straight into an encounter tab 400s (OpenEMR needs a session token that
  // only exists when navigating from within the app). Kept for later if a
  // same-tab/backend-proxied navigation approach replaces the raw link.

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#000', color: '#e0e0e0' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
        padding: '0.75rem 1rem',
        borderBottom: '1px solid #222',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: '0.85rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <select
            value={selectedPatient}
            onChange={(e) => setSelectedPatient(e.target.value)}
            style={{ padding: '0.35rem 0.5rem', backgroundColor: '#111', color: '#e0e0e0', border: '1px solid #333' }}
          >
            <option value="">Select patient...</option>
            {patients.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>

          {visits.length > 0 && (
            <select
              value={selectedVisitFolder}
              onChange={(e) => setSelectedVisitFolder(e.target.value)}
              style={{ padding: '0.35rem 0.5rem', backgroundColor: '#111', color: '#e0e0e0', border: '1px solid #333' }}
            >
              {visits.map((v) => (
                <option key={v.visit_folder} value={v.visit_folder}>
                  {parseVisitDate(v.visit_folder)}
                </option>
              ))}
            </select>
          )}

          {selectedPatient && OPENEMR_PID_MAP[selectedPatient] && (
            <a
              href={OPENEMR_BASE}
              target="_blank"
              rel="noopener noreferrer"
              style={openEmrButtonStyle}
            >
              View in OpenEMR →
            </a>
          )}
        </div>

      </div>

      {selectedVisit && (
        <XrayFullScreenViewer
          visit={selectedVisit}
          patientId={selectedPatient}
          patientDemographics={patientDemographics}
          allVisits={visits}
        />
      )}
    </div>
  )
}

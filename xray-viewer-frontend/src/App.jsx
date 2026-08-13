import XrayFullScreenViewer from './XrayFullScreenViewer.jsx'

// Embedded deployment (inside OpenEMR's interface/forms/xray_viewer/
// public/index.php): identity comes from window.__XRAY_BOOTSTRAP__,
// injected server-side from the PHP session (pid/encounter) plus two
// already-native OpenEMR fields that happen to already be the exact DICOM
// identifiers needed - patient_data.lname (== DICOM PatientID) and
// form_encounter.date (== DICOM StudyDateTime) - confirmed against the
// real seeded data, not guessed. No patient/visit picker UI needed here:
// one embedded instance = one encounter = one visit, already answered by
// the embedding context itself. The old standalone dropdown-driven shell
// (API_BASE fetches, OPENEMR_PID_MAP, the "View in OpenEMR" link) is gone -
// its equivalent (a "Back to Chart" link) now lives in new.php's own
// chrome, outside this iframe entirely.
const bootstrap = window.__XRAY_BOOTSTRAP__ || {}

export default function App() {
  if (!bootstrap.patientId || !bootstrap.studyDateYYYYMMDD) {
    return (
      <div style={{ padding: '2rem', color: '#e0704f', fontFamily: 'monospace' }}>
        X-ray viewer bootstrap data missing or incomplete.
      </div>
    )
  }

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#000', color: '#e0e0e0' }}>
      <XrayFullScreenViewer
        patientId={bootstrap.patientId}
        studyDateYYYYMMDD={bootstrap.studyDateYYYYMMDD}
        patientDemographics={bootstrap.patientDemographics || {}}
      />
    </div>
  )
}

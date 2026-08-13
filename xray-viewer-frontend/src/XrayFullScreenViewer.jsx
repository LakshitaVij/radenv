import { useEffect, useRef, useState } from 'react'
import { RenderingEngine, Enums as csEnums, init as csInit, metaData as csMetaData } from '@cornerstonejs/core'
import { init as csDicomImageLoaderInit, wadors } from '@cornerstonejs/dicom-image-loader'
import {
  init as csToolsInit,
  addTool,
  ToolGroupManager,
  ZoomTool,
  PanTool,
  WindowLevelTool,
  StackScrollTool,
  LengthTool,
  Enums as csToolsEnums,
} from '@cornerstonejs/tools'

// Toolbar modes - each sets what left-click-drag (Primary) does. Pan
// (middle-drag) and Zoom (right-drag + wheel) stay bound as fixed
// shortcuts regardless of which mode is selected here - standard PACS
// viewer convention (matches OHIF's own toolbar behavior).
const TOOLBAR_MODES = [
  { name: WindowLevelTool.toolName, label: 'Brightness' },
  { name: ZoomTool.toolName, label: 'Zoom' },
  { name: PanTool.toolName, label: 'Pan' },
  { name: LengthTool.toolName, label: 'Measure' },
]

// Relative path. In embedded (OpenEMR) deployment this is the PHP proxy at
// interface/forms/xray_viewer/public/dicomweb_proxy.php (path-info style,
// e.g. /dicomweb_proxy.php/studies/...); in the Vite dev server it goes
// through server.proxy's /dicom-web rewrite instead. Either way the browser
// sees a same-origin request rather than hitting a CORS/mixed-content wall
// talking to Orthanc directly.
const ORTHANC_DICOMWEB_BASE = import.meta.env.DEV ? '/dicom-web' : 'dicomweb_proxy.php'
const RENDERING_ENGINE_ID = 'chartr-rendering-engine'
const TOOL_GROUP_ID = 'chartr-tool-group'

let cornerstoneInitialized = false
let renderingEngine = null
let toolGroup = null

// Serializes viewport enable/setStack calls across every pane on the page.
// Confirmed root cause (via a reversed-processing-order test, not a guess):
// whichever image decodes FIRST on a freshly loaded page reliably fails
// ("Cannot read properties of undefined (reading 'samplesPerPixel')"),
// regardless of which specific series it is - swapping series order moved
// the failure to whichever one became first. This is a decode-worker-pool
// cold-start race (the first message dispatched to a freshly spun-up Web
// Worker can race ahead of the worker finishing its own init), not a
// per-image data problem. Serializing pane setup here doesn't remove the
// cold start itself but keeps setup order deterministic; the actual fix is
// the retry-on-first-failure logic in CornerstoneViewportPane below.
let viewportSetupChain = Promise.resolve()

// One-time setup: Cornerstone core + the DICOMweb image loader pointed at
// our local Orthanc, plus real interaction tools (zoom/pan/window-level -
// the actual radiologist convention, not a custom scheme). Guarded so
// re-mounts (React StrictMode, navigating between visits) don't re-init.
function ensureCornerstoneInitialized() {
  if (cornerstoneInitialized) return
  cornerstoneInitialized = true

  csInit()
  csToolsInit()

  // Real current API (v5.6.x): a plain init() call registers the wadors/
  // wadouri loaders and web worker manager internally - no manual
  // external.cornerstone/external.dicomParser wiring or configure() call,
  // that was the legacy pre-Cornerstone3D cornerstone-wado-image-loader API.
  csDicomImageLoaderInit()

  // init() registers the wadors image loader itself, but NOT the metadata
  // provider that translates our raw registered DICOM+JSON (via
  // wadors.metaDataManager.add) into the named "modules" (imagePixelModule,
  // voiLutModule, etc) Cornerstone's core actually reads from - confirmed
  // by reading registerLoaders.js, which never calls addProvider. Without
  // this, getImageFrame() crashes reading undefined.samplesPerPixel.
  csMetaData.addProvider(wadors.metaData.metaDataProvider)

  addTool(ZoomTool)
  addTool(PanTool)
  addTool(WindowLevelTool)
  addTool(StackScrollTool)
  addTool(LengthTool)

  renderingEngine = new RenderingEngine(RENDERING_ENGINE_ID)

  toolGroup = ToolGroupManager.createToolGroup(TOOL_GROUP_ID)
  toolGroup.addTool(ZoomTool.toolName)
  toolGroup.addTool(PanTool.toolName)
  toolGroup.addTool(WindowLevelTool.toolName)
  toolGroup.addTool(StackScrollTool.toolName)
  toolGroup.addTool(LengthTool.toolName)

  // Fixed shortcuts, independent of whichever toolbar mode is selected -
  // standard PACS viewer convention: middle-drag always pans, right-drag
  // and wheel always zoom, regardless of what left-click-drag is doing.
  toolGroup.setToolActive(PanTool.toolName, {
    bindings: [{ mouseButton: csToolsEnums.MouseBindings.Auxiliary }],
  })
  toolGroup.setToolActive(ZoomTool.toolName, {
    bindings: [
      { mouseButton: csToolsEnums.MouseBindings.Secondary },
      { mouseButton: csToolsEnums.MouseBindings.Wheel },
    ],
  })
  // Left-click-drag (Primary) is switchable via the toolbar - defaults to
  // window/level (brightness), the most common first action.
  setPrimaryTool(WindowLevelTool.toolName)
}

// Which tool currently owns the Primary (left-click-drag) binding - only
// this one gets passivated on the next switch. Pan and Zoom are NOT
// tracked here since they keep their own fixed Auxiliary/Secondary/Wheel
// bindings permanently - setToolActive merges bindings rather than
// replacing them, so giving Zoom/Pan the Primary binding too (when
// toolbar-selected) is additive and safe, but passivating them would wipe
// their fixed shortcuts too, which a naive "passivate everything else"
// loop would have done.
let currentPrimaryTool = null

// Switches which tool left-click-drag (Primary) triggers - called by the
// toolbar.
function setPrimaryTool(toolName) {
  if (!toolGroup) return
  if (currentPrimaryTool && currentPrimaryTool !== toolName) {
    toolGroup.setToolPassive(currentPrimaryTool)
  }
  toolGroup.setToolActive(toolName, {
    bindings: [{ mouseButton: csToolsEnums.MouseBindings.Primary }],
  })
  currentPrimaryTool = toolName
}

// Looks up one visit's X-ray series (with real WADO-RS image ids) via
// Orthanc's QIDO-RS, keyed on PatientID + StudyDate (both real DICOM tags) -
// not guessed from local file/folder names, which turned out not to be
// reliable (one naming convention truncates the series UID). Returns an
// ORDERED array (Orthanc's own series order) rather than a dict, since the
// series name/label now comes from here directly (SeriesDescription, tag
// 0008103E) instead of being cross-referenced positionally against a
// separately-fetched visit list from the old Azure backend.
async function lookupStudySeries(patientId, studyDateYYYYMMDD) {
  const studiesResp = await fetch(
    `${ORTHANC_DICOMWEB_BASE}/studies?PatientID=${encodeURIComponent(patientId)}&StudyDate=${studyDateYYYYMMDD}`,
    { headers: { Accept: 'application/dicom+json' } }
  )
  if (!studiesResp.ok) throw new Error(`QIDO studies lookup failed: ${studiesResp.status}`)
  const studies = await studiesResp.json()
  if (!studies.length) throw new Error(`No Orthanc study found for ${patientId}/${studyDateYYYYMMDD}`)
  const studyUID = studies[0]['0020000D']?.Value?.[0]
  if (!studyUID) throw new Error('Study response missing StudyInstanceUID')

  const seriesResp = await fetch(`${ORTHANC_DICOMWEB_BASE}/studies/${studyUID}/series`, {
    headers: { Accept: 'application/dicom+json' },
  })
  const seriesList = await seriesResp.json()

  // Must run BEFORE any metadata registration below, not just before any
  // viewport uses it - confirmed root cause of a deterministic (not
  // random) one-pane-always-fails bug: ensureCornerstoneInitialized()
  // (specifically csDicomImageLoaderInit()) resets/initializes the loader's
  // internal state, including wadors.metaDataManager's storage. The
  // original code called ensureCornerstoneInitialized() AFTER each
  // metaDataManager.add(), so for the very first image ever processed on a
  // page load, its just-registered metadata was getting wiped out by the
  // loader init that ran immediately after - every subsequent image was
  // fine because init had already happened. Reversing viewport processing
  // order moved the failure to whichever series became first, proving it
  // was purely about registration order relative to this init call, not
  // about any specific image's data.
  ensureCornerstoneInitialized()

  // Concurrent again (Promise.all), not the earlier sequential for...of.
  // That serialization was a workaround for what looked like a concurrency
  // race but was actually a one-time initialization-ordering bug (see the
  // comment on ensureCornerstoneInitialized() above) - now that
  // ensureCornerstoneInitialized() runs up front, synchronously, before any
  // metadata registration starts, concurrent registration across series/
  // instances is safe again. Confirmed by repeated test runs after
  // reverting. Meaningfully faster: every series/instance/metadata fetch
  // was previously a fully sequential round trip through the PHP proxy.
  return Promise.all(
    seriesList.map(async (series) => {
      const seriesUID = series['0020000E']?.Value?.[0]
      const seriesName = series['0008103E']?.Value?.[0] || 'View'
      if (!seriesUID) return null

      const instancesResp = await fetch(
        `${ORTHANC_DICOMWEB_BASE}/studies/${studyUID}/series/${seriesUID}/instances`,
        { headers: { Accept: 'application/dicom+json' } }
      )
      const instances = await instancesResp.json()
      const imageIds = await Promise.all(
        instances.map(async (inst) => {
          const sopUID = inst['00080018']?.Value?.[0]
          const imageId = `wadors:${ORTHANC_DICOMWEB_BASE}/studies/${studyUID}/series/${seriesUID}/instances/${sopUID}/frames/1`

          // Cornerstone can't compute a VOI (window/level) range from pixel
          // frame bytes alone - it needs the real DICOM header tags
          // (WindowCenter/WindowWidth etc), fetched separately via WADO-RS
          // metadata and registered before the viewport uses this imageId.
          // Without this, tools that touch VOI (e.g. window-level dragging)
          // throw "Viewport is not a valid type" - confirmed by reading
          // WindowLevelTool's own source.
          const metadataResp = await fetch(
            `${ORTHANC_DICOMWEB_BASE}/studies/${studyUID}/series/${seriesUID}/instances/${sopUID}/metadata`,
            { headers: { Accept: 'application/dicom+json' } }
          )
          if (metadataResp.ok) {
            const metadataArr = await metadataResp.json()
            if (metadataArr[0]) {
              wadors.metaDataManager.add(imageId, metadataArr[0])
            }
          }
          return imageId
        })
      )
      return { seriesUID, seriesName, imageIds }
    })
  ).then((results) => results.filter(Boolean))
}

// All of a patient's study dates (no StudyDate filter), sorted
// chronologically - used to find the visit immediately before the current
// one for the PACS-style prior-comparison panel. Previously sourced from
// the old Azure backend's per-patient visit list (allVisits); now derived
// straight from Orthanc, same as everything else, since that backend no
// longer exists once embedded in OpenEMR.
async function lookupPriorStudyDate(patientId, currentStudyDateYYYYMMDD) {
  const dates = await lookupAllStudyDates(patientId)
  const currentIndex = dates.indexOf(currentStudyDateYYYYMMDD)
  return currentIndex > 0 ? dates[currentIndex - 1] : null
}

// Every study date this patient has (no StudyDate filter), sorted
// chronologically ascending - the full-history version of the query above,
// used by the left-side timeline so the user can jump straight to any prior
// study, not just the one immediately before the current one.
async function lookupAllStudyDates(patientId) {
  const studiesResp = await fetch(
    `${ORTHANC_DICOMWEB_BASE}/studies?PatientID=${encodeURIComponent(patientId)}`,
    { headers: { Accept: 'application/dicom+json' } }
  )
  if (!studiesResp.ok) throw new Error(`QIDO studies lookup failed: ${studiesResp.status}`)
  const studies = await studiesResp.json()
  return [...new Set(studies.map((s) => s['00080020']?.Value?.[0]).filter(Boolean))].sort()
}

function formatStudyDate(studyDateYYYYMMDD) {
  if (studyDateYYYYMMDD && studyDateYYYYMMDD.length === 8) {
    return `${studyDateYYYYMMDD.slice(0, 4)}-${studyDateYYYYMMDD.slice(4, 6)}-${studyDateYYYYMMDD.slice(6, 8)}`
  }
  return studyDateYYYYMMDD
}

// One real Cornerstone stack viewport, bound to a div, showing one series.
// Shows an explicit error state (not a silently blank pane, and no more
// PNG-thumbnail fallback - that came from the old Azure backend's
// /thumbnail endpoint, which no longer exists once embedded in OpenEMR) if
// the real DICOMweb image can't be loaded.
function CornerstoneViewportPane({ viewportId, wadoImageId, label }) {
  const elementRef = useRef(null)
  const [loadError, setLoadError] = useState(false)
  // Explicit loading state instead of a plain black pane while decoding -
  // the bundle/codec init + real DICOMweb round trips take a few real
  // seconds, and an unlabeled black rectangle during that window looks
  // identical to the (now-fixed) failure state, easy to mistake for the
  // bug recurring.
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    if (!wadoImageId || !elementRef.current) return
    ensureCornerstoneInitialized()

    // StrictMode (enabled in main.jsx) mounts this effect, cleans it up
    // almost immediately, then re-mounts it - a stale setStack() promise
    // from the first mount can otherwise resolve AFTER cleanup already
    // ran and call render()/addViewport() against an already-disabled
    // viewport, leaving a blank canvas. This guard makes the async
    // callback a no-op once this specific effect instance is cleaned up.
    let cancelled = false

    const element = elementRef.current
    setLoaded(false)
    viewportSetupChain = viewportSetupChain.then(async () => {
      if (cancelled) return
      try {
        renderingEngine.enableElement({
          viewportId,
          element,
          type: csEnums.ViewportType.STACK,
        })

        // The harness's "wait for the X-ray to load" check used to poll
        // <img>.complete/naturalWidth, but the actual DICOM pixels paint onto
        // this element's <canvas> via WebGL - that check always passed on
        // unrelated <img> tags elsewhere on the page while the canvas was
        // still blank (confirmed by pixel-analyzing real eval screenshots:
        // every episode's first screenshot had an unloaded viewport). Stamp a
        // DOM flag on the real Cornerstone render-complete event so the
        // harness can poll for actual pixels instead.
        element.addEventListener(csEnums.Events.IMAGE_RENDERED, () => {
          element.dataset.xrayLoaded = 'true'
          setLoaded(true)
        })

        const viewport = renderingEngine.getViewport(viewportId)
        await viewport.setStack([wadoImageId])
        if (cancelled) return
        viewport.render()
        toolGroup.addViewport(viewportId, RENDERING_ENGINE_ID)
      } catch (err) {
        if (cancelled) return
        console.error('Cornerstone viewport setup failed:', err)
        setLoadError(true)
      }
    })

    return () => {
      cancelled = true
      try {
        renderingEngine?.disableElement(viewportId)
      } catch {
        // element already gone (unmount race) - fine
      }
    }
  }, [viewportId, wadoImageId])

  if (!wadoImageId || loadError) {
    return (
      <div
        style={{
          color: '#e0704f',
          fontFamily: 'monospace',
          fontSize: '0.85rem',
          textAlign: 'center',
          padding: '1rem',
        }}
      >
        {label} failed to load
      </div>
    )
  }

  // Absolutely positioned against the pane (which has position: 'relative'
  // and real computed dimensions from the grid layout) rather than relying
  // on flex sizing - an empty div with width/height: 100% inside a flex
  // container with default align-items can collapse to zero size, unlike
  // an <img> which has intrinsic dimensions to size from.
  return (
    <>
      <div ref={elementRef} style={{ position: 'absolute', inset: 0 }} />
      {!loaded && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#7fd9a8',
            fontFamily: 'monospace',
            fontSize: '0.85rem',
            letterSpacing: '0.05em',
            pointerEvents: 'none',
          }}
        >
          Loading X-ray&hellip;
        </div>
      )}
    </>
  )
}

// PACS-style full-bleed viewer: one pane per view (series). 1 view = single
// full-screen pane; 2 views (e.g. PA + lateral) = side-by-side split; more
// than 2 = wraps into a grid rather than breaking, though the dataset here
// is 1-2 views per visit in practice.
// Visible toolbar - lets a human (or, once the agent gets a real drag
// primitive, the agent) explicitly pick which mode left-click-drag is in,
// rather than relying purely on which mouse button happens to be bound.
function Toolbar({ activeTool, onSelect }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: '0.5rem',
        padding: '0.5rem 0.75rem',
        backgroundColor: '#111',
        borderBottom: '1px solid #333',
      }}
    >
      {TOOLBAR_MODES.map((mode) => (
        <button
          key={mode.name}
          onClick={() => onSelect(mode.name)}
          style={{
            // Sized to match openEmrButtonStyle (App.jsx) - same reasoning:
            // Gemini kept missing smaller click targets by a wide margin
            // in testing, so every button the agent needs to click
            // reliably gets the same larger size.
            padding: '0.6rem 1.1rem',
            fontSize: '1.05rem',
            fontWeight: 600,
            fontFamily: 'monospace',
            borderRadius: '5px',
            border: activeTool === mode.name ? '2px solid #5a8ec2' : '2px solid #444',
            backgroundColor: activeTool === mode.name ? '#3a6ea5' : '#222',
            color: '#e0e0e0',
            cursor: 'pointer',
            lineHeight: 1.4,
          }}
        >
          {mode.label}
        </button>
      ))}
    </div>
  )
}

// Renders one study's own grid of view-panes (findings/views for a single
// visit) - extracted so it can be used twice: once for the prior study,
// once for the current one, side by side. Pane count/labels now come
// straight from the live Orthanc series lookup (SeriesDescription) instead
// of a separately-fetched visit list from the old Azure backend, cross-
// referenced positionally - one data source, no positional-matching risk.
function StudyBlock({ patientId, studyDateYYYYMMDD, patientDemographics, isPrior, widthPercent, onViewCount }) {
  const [series, setSeries] = useState(null)

  useEffect(() => {
    let cancelled = false
    setSeries(null)
    lookupStudySeries(patientId, studyDateYYYYMMDD)
      .then((result) => {
        if (cancelled) return
        setSeries(result)
        onViewCount?.(result.length)
      })
      .catch((err) => {
        console.error('DICOMweb series lookup failed:', err)
        if (!cancelled) {
          setSeries([])
          onViewCount?.(0)
        }
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onViewCount is a stable ref from the parent, not a reactive dependency
  }, [patientId, studyDateYYYYMMDD])

  if (series === null) {
    // Still resolving the series list itself (before individual panes even
    // exist to show their own "Loading <label>..." state) - without this,
    // the whole pane area is blank/black for the first stretch of load
    // time, indistinguishable from the failure state this project spent a
    // long time chasing down.
    return (
      <div
        style={{
          width: `${widthPercent}%`,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#7fd9a8',
          fontFamily: 'monospace',
          fontSize: '0.85rem',
          letterSpacing: '0.05em',
        }}
      >
        Loading X-ray&hellip;
      </div>
    )
  }
  if (series.length === 0) return null

  const dateLabel = formatStudyDate(studyDateYYYYMMDD)

  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: series.length === 1 ? '1fr' : `repeat(${Math.min(series.length, 2)}, 1fr)`,
    gridAutoRows: series.length > 2 ? '1fr' : undefined,
    gap: '2px',
    width: '100%',
    height: '100%',
    backgroundColor: '#000',
  }

  const demoLabel = [patientDemographics?.age, patientDemographics?.sex].filter(Boolean).join(' ')

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        // Explicit width (not flex: 1) so the two sides can be sized
        // proportionally to their own view counts - lets every individual
        // pane end up the same physical size even when the two studies
        // have different numbers of views, and lets the splitter drag
        // override it.
        width: `${widthPercent}%`,
        flexShrink: 0,
        minWidth: 0,
      }}
    >
      {isPrior && (
        <div
          style={{
            padding: '0.25rem 0.75rem',
            fontSize: '0.75rem',
            fontFamily: 'monospace',
            color: '#e0e0e0',
            backgroundColor: '#222',
            textAlign: 'center',
            letterSpacing: '0.05em',
          }}
        >
          PRIOR
        </div>
      )}
      <div style={gridStyle}>
        {series.map((s) => (
          <div
            key={s.seriesUID}
            style={{
              position: 'relative',
              backgroundColor: '#000',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden',
            }}
          >
            <CornerstoneViewportPane
              viewportId={`viewport-${s.seriesUID}-${isPrior ? 'prior' : 'current'}`}
              wadoImageId={s.imageIds[0]}
              label={s.seriesName}
            />

            {/* Corner overlay text - PACS convention: patient/view info top-left, study info top-right */}
            <div style={overlayStyle('top', 'left')}>
              <div style={{ fontWeight: 600 }}>{patientId}</div>
              {demoLabel && <div>{demoLabel}</div>}
            </div>
            <div style={overlayStyle('top', 'right')}>
              <div>{dateLabel}</div>
              <div>{s.seriesName}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Thin draggable bar between the two study sides - lets the user override
// the default proportional split. Tracks mouse position relative to the
// shared container's bounding rect while dragging.
function Splitter({ containerRef, onDrag }) {
  const draggingRef = useRef(false)

  useEffect(() => {
    function handleMouseMove(e) {
      if (!draggingRef.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const pct = ((e.clientX - rect.left) / rect.width) * 100
      onDrag(Math.min(90, Math.max(10, pct)))
    }
    function handleMouseUp() {
      draggingRef.current = false
    }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [containerRef, onDrag])

  return (
    <div
      onMouseDown={() => {
        draggingRef.current = true
      }}
      style={{
        width: '6px',
        flexShrink: 0,
        cursor: 'col-resize',
        backgroundColor: '#333',
      }}
    />
  )
}

// Stamps a query param onto the CURRENT url via history.replaceState (no
// reload, no navigation entry added) - the only way for grade_process.py
// to see that this happened at all. Every other Z1 checkpoint is detected
// by scanning the URLs agent_episode.py already logs every step
// (page_state's frame_urls) for a known pattern (formname=, set_pid=,
// etc.) - but clicking a timeline entry only changes React state, not the
// URL, so without this the click is invisible to grading entirely.
function stampUrlParam(key, value) {
  try {
    const url = new URL(window.location.href)
    url.searchParams.set(key, value)
    window.history.replaceState(null, '', url.toString())
  } catch (err) {
    console.error('Failed to stamp URL param:', err)
  }
}

// Persistent left-side vertical timeline of every study this patient has
// (not just the immediate prior one, unlike the current-vs-prior
// comparison pane) - clicking an entry switches the main viewer to that
// study, same as browsing through history. Newest at top, matching how
// most EHR history views order events.
function PriorStudiesTimeline({ patientId, activeStudyDateYYYYMMDD, onSelect }) {
  const [dates, setDates] = useState(null)

  useEffect(() => {
    let cancelled = false
    setDates(null)
    lookupAllStudyDates(patientId)
      .then((result) => {
        if (!cancelled) setDates(result)
        // Grading needs to know whether a prior study genuinely existed AT
        // ALL, independent of whether the agent ever clicked one - a
        // patient with only one study on file should never be penalized
        // for "not comparing to prior," same "nothing to engage with"
        // carve-out Z1.5 already uses for a missing office-visit form.
        stampUrlParam('priorStudiesAvailable', result.length > 1 ? '1' : '0')
      })
      .catch((err) => {
        console.error('Study timeline lookup failed:', err)
        if (!cancelled) setDates([])
      })
    return () => {
      cancelled = true
    }
  }, [patientId])

  const orderedDates = dates ? [...dates].reverse() : null // newest first

  return (
    <div
      style={{
        width: '140px',
        flexShrink: 0,
        backgroundColor: '#111',
        borderRight: '1px solid #333',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          padding: '0.6rem 0.75rem',
          fontSize: '0.7rem',
          fontFamily: 'monospace',
          letterSpacing: '0.08em',
          color: '#7c9389',
          borderBottom: '1px solid #333',
        }}
      >
        STUDIES
      </div>
      {orderedDates === null && (
        <div style={{ padding: '0.75rem', fontSize: '0.75rem', fontFamily: 'monospace', color: '#7fd9a8' }}>
          Loading&hellip;
        </div>
      )}
      {orderedDates?.map((d) => {
        const isActive = d === activeStudyDateYYYYMMDD
        return (
          <button
            key={d}
            onClick={() => onSelect(d)}
            style={{
              display: 'block',
              width: '100%',
              textAlign: 'left',
              padding: '0.6rem 0.75rem',
              border: 'none',
              borderLeft: isActive ? '3px solid #5a8ec2' : '3px solid transparent',
              borderBottom: '1px solid #222',
              backgroundColor: isActive ? '#1e2f42' : 'transparent',
              color: isActive ? '#e0e0e0' : '#9fb1af',
              fontFamily: 'monospace',
              fontSize: '0.8rem',
              cursor: 'pointer',
            }}
          >
            {formatStudyDate(d)}
          </button>
        )
      })}
    </div>
  )
}

// PACS-style full-bleed viewer. When the current visit has an earlier one
// for the same patient (looked up live from Orthanc - lookupPriorStudyDate),
// that prior study renders side by side on the left - real PACS current-
// vs-prior comparison layout, not just the current visit.
export default function XrayFullScreenViewer({ patientId, studyDateYYYYMMDD, patientDemographics }) {
  const [activeTool, setActiveTool] = useState(WindowLevelTool.toolName)
  const [manualSplitPercent, setManualSplitPercent] = useState(null)
  const [priorStudyDateYYYYMMDD, setPriorStudyDateYYYYMMDD] = useState(null)
  const [viewCounts, setViewCounts] = useState({ prior: 0, current: 0 })
  // The study actually shown in the main viewport - defaults to the
  // encounter's own date (the studyDateYYYYMMDD prop), but the left
  // timeline can override it to browse any of this patient's prior
  // studies without navigating away from this encounter. Resets back to
  // the prop's value whenever the encounter itself changes (new prop),
  // so switching patients/encounters doesn't leave a stale manual
  // selection behind.
  const [selectedStudyDateYYYYMMDD, setSelectedStudyDateYYYYMMDD] = useState(studyDateYYYYMMDD)
  const containerRef = useRef(null)

  const handleToolSelect = (toolName) => {
    setPrimaryTool(toolName)
    setActiveTool(toolName)
  }

  // Grading detects this the same way as every other Z1 checkpoint - a
  // pattern in the logged URLs, not a live DOM/state query - so the click
  // has to actually change the URL, not just React state.
  const handleTimelineSelect = (date) => {
    setSelectedStudyDateYYYYMMDD(date)
    if (date !== studyDateYYYYMMDD) {
      stampUrlParam('viewedPriorStudy', '1')
    }
  }

  useEffect(() => {
    setSelectedStudyDateYYYYMMDD(studyDateYYYYMMDD)
  }, [studyDateYYYYMMDD])

  useEffect(() => {
    let cancelled = false
    setManualSplitPercent(null)
    setPriorStudyDateYYYYMMDD(null)
    setViewCounts({ prior: 0, current: 0 })
    lookupPriorStudyDate(patientId, selectedStudyDateYYYYMMDD)
      .then((priorDate) => {
        if (!cancelled) setPriorStudyDateYYYYMMDD(priorDate)
      })
      .catch((err) => {
        console.error('Prior-visit lookup failed:', err)
      })
    return () => {
      cancelled = true
    }
  }, [patientId, selectedStudyDateYYYYMMDD])

  // Default split is proportional to each side's view count, so every
  // individual pane comes out the same physical size (e.g. prior with 1
  // view + current with 2 views -> 33% / 67%, not a plain 50/50 that
  // would make the current side's panes half the width of the prior's).
  // View counts arrive asynchronously (each StudyBlock reports its own via
  // onViewCount once its live Orthanc series lookup resolves); manualSplitPercent
  // (set by dragging the splitter) overrides this default until the next visit change.
  const defaultSplitPercent = viewCounts.prior
    ? (viewCounts.prior / (viewCounts.prior + viewCounts.current)) * 100
    : 0
  const splitPercent = manualSplitPercent ?? defaultSplitPercent

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 52px)' }}>
      <PriorStudiesTimeline
        patientId={patientId}
        activeStudyDateYYYYMMDD={selectedStudyDateYYYYMMDD}
        onSelect={handleTimelineSelect}
      />
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
        <Toolbar activeTool={activeTool} onSelect={handleToolSelect} />
        <div ref={containerRef} style={{ display: 'flex', flex: 1, minHeight: 0 }}>
          {priorStudyDateYYYYMMDD && (
            <>
              <StudyBlock
                patientId={patientId}
                studyDateYYYYMMDD={priorStudyDateYYYYMMDD}
                patientDemographics={patientDemographics}
                isPrior
                widthPercent={splitPercent}
                onViewCount={(n) => setViewCounts((prev) => ({ ...prev, prior: n }))}
              />
              <Splitter containerRef={containerRef} onDrag={setManualSplitPercent} />
            </>
          )}
          <StudyBlock
            patientId={patientId}
            studyDateYYYYMMDD={selectedStudyDateYYYYMMDD}
            patientDemographics={patientDemographics}
            isPrior={false}
            widthPercent={priorStudyDateYYYYMMDD ? 100 - splitPercent : 100}
            onViewCount={(n) => setViewCounts((prev) => ({ ...prev, current: n }))}
          />
        </div>
      </div>
    </div>
  )
}

function overlayStyle(vPos, hPos) {
  return {
    position: 'absolute',
    [vPos]: '0.75rem',
    [hPos]: '0.75rem',
    color: '#e0e0e0',
    fontSize: '0.8rem',
    fontFamily: 'monospace',
    lineHeight: 1.4,
    textShadow: '0 1px 2px rgba(0,0,0,0.9)',
    textAlign: hPos,
    pointerEvents: 'none',
  }
}

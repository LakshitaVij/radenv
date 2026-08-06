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

const API_BASE = 'http://localhost:7071/api'
// Relative path, not an absolute localhost:8042 URL - goes through the Vite
// dev server's proxy (vite.config.js server.proxy) so the browser sees a
// same-origin request instead of hitting a CORS wall talking to Orthanc
// directly.
const ORTHANC_DICOMWEB_BASE = '/dicom-web'
const RENDERING_ENGINE_ID = 'chartr-rendering-engine'
const TOOL_GROUP_ID = 'chartr-tool-group'

let cornerstoneInitialized = false
let renderingEngine = null
let toolGroup = null

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

// Looks up the real DICOM identifiers for one visit's X-ray via Orthanc's
// QIDO-RS, keyed on PatientID + StudyDate (both real DICOM tags) - not
// guessed from local file/folder names, which turned out not to be
// reliable (one naming convention truncates the series UID).
async function lookupWadoImageIds(patientId, studyDateYYYYMMDD) {
  console.error('[diag] lookupWadoImageIds called for', patientId, studyDateYYYYMMDD)
  const studiesResp = await fetch(
    `${ORTHANC_DICOMWEB_BASE}/studies?PatientID=${encodeURIComponent(patientId)}&StudyDate=${studyDateYYYYMMDD}`,
    { headers: { Accept: 'application/dicom+json' } }
  )
  console.error('[diag] studies response status', studiesResp.status)
  if (!studiesResp.ok) throw new Error(`QIDO studies lookup failed: ${studiesResp.status}`)
  const studies = await studiesResp.json()
  console.error('[diag] studies found:', studies.length)
  if (!studies.length) throw new Error(`No Orthanc study found for ${patientId}/${studyDateYYYYMMDD}`)
  const studyUID = studies[0]['0020000D']?.Value?.[0]
  if (!studyUID) throw new Error('Study response missing StudyInstanceUID')
  console.error('[diag] studyUID', studyUID)

  const seriesResp = await fetch(`${ORTHANC_DICOMWEB_BASE}/studies/${studyUID}/series`, {
    headers: { Accept: 'application/dicom+json' },
  })
  const seriesList = await seriesResp.json()
  console.error('[diag] series found:', seriesList.length)

  const imageIdsBySeriesUID = {}
  for (const series of seriesList) {
    const seriesUID = series['0020000E']?.Value?.[0]
    if (!seriesUID) continue
    const instancesResp = await fetch(
      `${ORTHANC_DICOMWEB_BASE}/studies/${studyUID}/series/${seriesUID}/instances`,
      { headers: { Accept: 'application/dicom+json' } }
    )
    const instances = await instancesResp.json()
    console.error('[diag] series', seriesUID, 'instances found:', instances.length)
    imageIdsBySeriesUID[seriesUID] = await Promise.all(
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
            ensureCornerstoneInitialized()
            const resolvedPixelModule = csMetaData.get('imagePixelModule', imageId)
            console.error('[diag] registered metadata for', imageId, 'tags:', Object.keys(metadataArr[0]).length, 'has 00280002 (SamplesPerPixel):', '00280002' in metadataArr[0], '| provider resolves imagePixelModule:', JSON.stringify(resolvedPixelModule))
          } else {
            console.error('[diag] metadata response was empty array for', imageId)
          }
        } else {
          console.error('[diag] metadata fetch failed', metadataResp.status, 'for', imageId)
        }
        return imageId
      })
    )
  }
  return imageIdsBySeriesUID
}

function extractBlobNameFromSasUrl(sasUrl) {
  try {
    const url = new URL(sasUrl)
    const path = decodeURIComponent(url.pathname)
    return path.replace('/dicom-patients/', '')
  } catch {
    return ''
  }
}

function parseVisitDate(visitFolder) {
  const parts = visitFolder.split('_')
  const dateStr = parts[1]
  if (dateStr && dateStr.length === 8) {
    return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`
  }
  return visitFolder
}

// One real Cornerstone stack viewport, bound to a div, showing one series.
// Falls back to the existing PNG-thumbnail <img> if the real DICOMweb
// image can't be loaded (Orthanc down, visit not uploaded yet, etc.) -
// visible failure, not a silent blank pane.
function CornerstoneViewportPane({ viewportId, wadoImageId, fallbackImgUrl, label }) {
  const elementRef = useRef(null)
  const [loadError, setLoadError] = useState(false)

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
    try {
      renderingEngine.enableElement({
        viewportId,
        element,
        type: csEnums.ViewportType.STACK,
      })
      const viewport = renderingEngine.getViewport(viewportId)
      viewport
        .setStack([wadoImageId])
        .then(() => {
          if (cancelled) return
          viewport.render()
          toolGroup.addViewport(viewportId, RENDERING_ENGINE_ID)
        })
        .catch((err) => {
          if (cancelled) return
          console.error('Cornerstone failed to load real DICOM image, falling back to PNG:', err)
          setLoadError(true)
        })
    } catch (err) {
      console.error('Cornerstone viewport setup failed, falling back to PNG:', err)
      setLoadError(true)
    }

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
      <img
        src={fallbackImgUrl}
        alt={label}
        style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
      />
    )
  }

  // Absolutely positioned against the pane (which has position: 'relative'
  // and real computed dimensions from the grid layout) rather than relying
  // on flex sizing - an empty div with width/height: 100% inside a flex
  // container with default align-items can collapse to zero size, unlike
  // an <img> which has intrinsic dimensions to size from.
  return <div ref={elementRef} style={{ position: 'absolute', inset: 0 }} />
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
// once for the current one, side by side.
function StudyBlock({ visit, patientId, patientDemographics, isPrior, widthPercent }) {
  const [wadoImageIdsBySeries, setWadoImageIdsBySeries] = useState(null)

  const views = visit.series
    .filter((s) => s.images.length > 0)
    .map((s) => ({
      label: s.series_name.split('_')[0],
      imgUrl: s.images[0],
    }))

  const dateLabel = parseVisitDate(visit.visit_folder)
  const studyDateYYYYMMDD = dateLabel.replaceAll('-', '')

  useEffect(() => {
    let cancelled = false
    setWadoImageIdsBySeries(null)
    lookupWadoImageIds(patientId, studyDateYYYYMMDD)
      .then((result) => {
        if (!cancelled) setWadoImageIdsBySeries(result)
      })
      .catch((err) => {
        console.error('Real DICOMweb lookup failed, panes will fall back to PNG thumbnails:', err)
        if (!cancelled) setWadoImageIdsBySeries({})
      })
    return () => {
      cancelled = true
    }
  }, [patientId, studyDateYYYYMMDD])

  if (views.length === 0) return null

  const seriesUIDs = Object.keys(wadoImageIdsBySeries || {})

  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: views.length === 1 ? '1fr' : `repeat(${Math.min(views.length, 2)}, 1fr)`,
    gridAutoRows: views.length > 2 ? '1fr' : undefined,
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
        {views.map((view, i) => {
          // Best-effort positional match: view order here mirrors the
          // series order Orthanc returns for this study, since our own
          // backend and Orthanc were both loaded from the same source
          // files. Falls back to PNG per-pane if this doesn't line up.
          const wadoImageId = wadoImageIdsBySeries?.[seriesUIDs[i]]?.[0]
          const fallbackImgUrl = `${API_BASE}/thumbnail?blob_name=${encodeURIComponent(
            extractBlobNameFromSasUrl(view.imgUrl)
          )}&size=1600`

          return (
            <div
              key={view.label}
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
                viewportId={`viewport-${patientId}-${studyDateYYYYMMDD}-${i}-${isPrior ? 'prior' : 'current'}`}
                wadoImageId={wadoImageId}
                fallbackImgUrl={fallbackImgUrl}
                label={view.label}
              />

              {/* Corner overlay text - PACS convention: patient/view info top-left, study info top-right */}
              <div style={overlayStyle('top', 'left')}>
                <div style={{ fontWeight: 600 }}>{patientId}</div>
                {demoLabel && <div>{demoLabel}</div>}
              </div>
              <div style={overlayStyle('top', 'right')}>
                <div>{dateLabel}</div>
                <div>{view.label}</div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function countViews(visit) {
  return visit ? visit.series.filter((s) => s.images.length > 0).length : 0
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

// PACS-style full-bleed viewer. When the selected visit has an earlier
// visit for the same patient (allVisits, sorted chronologically by the
// backend), that prior study renders side by side on the left - real
// PACS current-vs-prior comparison layout, not just the current visit.
export default function XrayFullScreenViewer({ visit, patientId, patientDemographics, allVisits }) {
  const [activeTool, setActiveTool] = useState(WindowLevelTool.toolName)
  const [manualSplitPercent, setManualSplitPercent] = useState(null)
  const containerRef = useRef(null)

  const handleToolSelect = (toolName) => {
    setPrimaryTool(toolName)
    setActiveTool(toolName)
  }

  const currentIndex = (allVisits || []).findIndex((v) => v.visit_folder === visit.visit_folder)
  const priorVisit = currentIndex > 0 ? allVisits[currentIndex - 1] : null

  // Default split is proportional to each side's view count, so every
  // individual pane comes out the same physical size (e.g. prior with 1
  // view + current with 2 views -> 33% / 67%, not a plain 50/50 that
  // would make the current side's panes half the width of the prior's).
  // Resets to that proportional default whenever the visit changes;
  // manualSplitPercent (set by dragging the splitter) overrides it until
  // the next visit change.
  const priorViewCount = countViews(priorVisit)
  const currentViewCount = countViews(visit)
  const defaultSplitPercent = priorViewCount
    ? (priorViewCount / (priorViewCount + currentViewCount)) * 100
    : 0
  const splitPercent = manualSplitPercent ?? defaultSplitPercent

  useEffect(() => {
    setManualSplitPercent(null)
  }, [patientId, visit.visit_folder])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 52px)' }}>
      <Toolbar activeTool={activeTool} onSelect={handleToolSelect} />
      <div ref={containerRef} style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {priorVisit && (
          <>
            <StudyBlock
              visit={priorVisit}
              patientId={patientId}
              patientDemographics={patientDemographics}
              isPrior
              widthPercent={splitPercent}
            />
            <Splitter containerRef={containerRef} onDrag={setManualSplitPercent} />
          </>
        )}
        <StudyBlock
          visit={visit}
          patientId={patientId}
          patientDemographics={patientDemographics}
          isPrior={false}
          widthPercent={priorVisit ? 100 - splitPercent : 100}
        />
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

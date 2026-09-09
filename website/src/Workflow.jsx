import { useEffect, useRef, useState } from 'react';

const STATIC_MEDIA = '(max-width: 900px), (prefers-reduced-motion: reduce), print';
const STEP_DURATION = 4200;
const steps = [
  {title: 'The original MEDIC-AD path', text: 'Visual tokens and text reach the language model directly; the anomaly branch supplies Ano tokens through the original sigmoid-based processor.'},
  {title: 'Add image-level supervision', text: 'Before sigmoid, raw A/N logits feed LCE and MIL; a normal/abnormal image label supervises the pooled evidence.'},
  {title: 'Budget the auxiliary gradient', text: 'CABG uses the auxiliary gradient and the LM gradient reference to set the LCE weight and cap its norm on their shared trainable support.'},
  {title: 'Update existing parameters', text: 'The combined gradients update existing prompts and attention projections; inference keeps both original paths, with no new module or trainable parameter.'},
];

function Box({x, y, width, lines, added = false, active = false}) {
  return <g className={`diagram-box${added ? ' added' : ''}${active ? ' is-active' : ''}`}>
    <rect x={x} y={y} width={width} height="66" rx="3" />
    <text x={x + width / 2} y={y + 25} textAnchor="middle">{lines.map((line, i) =>
      <tspan key={line} x={x + width / 2} dy={i ? 19 : 0} className={i ? 'diagram-small' : ''}>{line}</tspan>)}</text>
  </g>;
}

function MovingDot({path, blue}) {
  const motion = useRef(null);
  const opacity = useRef(null);
  useEffect(() => {
    // Start on user-driven mount, not at the SVG document's initial time.
    motion.current.beginElement();
    opacity.current.beginElement();
  }, []);
  return <circle r="3" opacity="0" className={`flow-dot${blue ? ' blue-dot' : ''}`} aria-hidden="true">
    <animateMotion ref={motion} path={path} begin="indefinite" dur="1.2s" repeatCount="1" fill="freeze" />
    <animate ref={opacity} attributeName="opacity" values="1;1;0" keyTimes="0;0.85;1" begin="indefinite" dur="1.2s" repeatCount="1" fill="freeze" />
  </circle>;
}

function Link({path, blue = false, active = false, pulse, dashed = false, name}) {
  return <g data-connection={name}>
    <path className={`diagram-line${blue ? ' blue-line' : ''}${active ? ' is-active' : ''}`} d={path} strokeDasharray={dashed ? '4 3' : undefined} />
    {active && <MovingDot key={pulse} path={path} blue={blue} />}
  </g>;
}

export default function Workflow() {
  const [step, setStep] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [pulse, setPulse] = useState(0);
  const [staticMode, setStaticMode] = useState(() => window.matchMedia(STATIC_MEDIA).matches);

  useEffect(() => {
    const media = window.matchMedia(STATIC_MEDIA);
    function sync() {
      setStaticMode(media.matches);
      if (media.matches) { setPlaying(false); setStep(null); }
    }
    function onVisibility() { if (document.hidden) setPlaying(false); }
    media.addEventListener('change', sync);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      media.removeEventListener('change', sync);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  useEffect(() => {
    if (!playing || staticMode) return;
    const timer = window.setTimeout(() => {
      if (step === steps.length - 1) { setPlaying(false); return; }
      setStep(step + 1);
      setPulse(value => value + 1);
    }, STEP_DURATION);
    return () => window.clearTimeout(timer);
  }, [playing, staticMode, step]);

  function selectStep(next) {
    setPlaying(false);
    setStep(next);
    setPulse(value => value + 1);
  }
  function playOnce() {
    if (playing) { setPlaying(false); return; }
    setStep(0);
    setPulse(value => value + 1);
    setPlaying(true);
  }
  const active = (...indices) => !staticMode && indices.includes(step);

  return <figure className={`workflow${staticMode ? ' static-workflow' : ''}`} data-step={step === null ? 'static' : step + 1} data-playing={playing}>
    <div className="workflow-heading"><strong>Where my method enters MEDIC-AD</strong><span><i className="key original" /> Original pathway <i className="key added" /> My training addition</span></div>
    <svg className="workflow-desktop" viewBox="0 0 1000 382" role="img" aria-labelledby="workflow-title workflow-description">
      <title id="workflow-title">Original inference pathway with a training-only LCE and CABG branch</title>
      <desc id="workflow-description">Image features feed both the language model and the anomaly processor. The original processor uses sigmoid anomaly-minus-normal attention, feature weighting, pooling and a Q-Former to produce Ano tokens. My auxiliary branch takes raw A and N logits before sigmoid, applies softsign contrast and MIL supervision, and uses CABG to balance training updates. Inference retains the original path.</desc>
      <defs>
        <marker id="arrow-grey" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8b949b" /></marker>
        <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#345f8b" /></marker>
      </defs>
      <text x="18" y="25" className="diagram-label">ORIGINAL INFERENCE PATH</text>
      {['M108 99H133', 'M285 99H317', 'M459 99H491', 'M645 99H677', 'M819 99H852'].map((path, index) => <Link key={path} path={path} name={`original-${index}`} active={active(0)} pulse={pulse} />)}
      <Box x={18} y={66} width={90} lines={['MRI image', 'input']} active={active(0)} />
      <Box x={137} y={66} width={148} lines={['Visual features', 'encoder + prompts']} active={active(0)} />
      <Box x={321} y={66} width={138} lines={['A / N attention', 'raw logits']} active={active(0, 1)} />
      <Box x={495} y={66} width={150} lines={['Sigmoid contrast', 'weight visual features']} active={active(0)} />
      <Box x={681} y={66} width={138} lines={['Anomaly processor', 'pool → Q-Former → Ano']} active={active(0)} />
      <Box x={856} y={66} width={126} lines={['Language model', 'generated answer']} active={active(0, 2)} />
      <Link path="M211 66V43H919V62" name="direct-visual-text" active={active(0)} pulse={pulse} />
      <rect x="585" y="32" width="248" height="21" fill="white" />
      <text x="709" y="46" textAnchor="middle" className="diagram-small">Projected visual tokens + text</text>
      <Link path="M390 132V207" name="pre-sigmoid-branch" blue active={active(1)} pulse={pulse} />
      <text x="405" y="173" className="diagram-blue-note">Branch before sigmoid</text>
      <text x="18" y="235" className="diagram-label blue-text">ADDED FOR TRAINING</text>
      <text x="18" y="258" className="diagram-small">Image-level supervision</text>
      <text x="593" y="188" textAnchor="middle" className={`diagram-blue-note${active(1) ? ' active-label' : ''}`}>Normal / abnormal label</text>
      <Link path="M593 194V209" name="image-label" blue active={active(1)} pulse={pulse} />
      <Link path="M919 132V209" name="lm-gradient-reference" blue dashed active={active(2)} pulse={pulse} />
      <rect x="781" y="164" width="126" height="24" fill="white" />
      <text x="899" y="180" textAnchor="end" className="diagram-blue-note">LM loss reference</text>
      <Box x={258} y={213} width={201} lines={['LCE · logit contrast', 'softsign(A − N)']} added active={active(1)} />
      <Link path="M459 246H491" name="lce-to-mil" blue active={active(1)} pulse={pulse} />
      <Box x={495} y={213} width={197} lines={['MIL · image supervision', 'smooth pooling + softplus']} added active={active(1, 2)} />
      <Link path="M692 246H724" name="auxiliary-gradient" blue active={active(2)} pulse={pulse} />
      <Box x={728} y={213} width={254} lines={['CABG · gradient budget', 'balance against the language loss']} added active={active(2, 3)} />
      <Link path="M855 279V316H457" name="parameter-update" blue active={active(3)} pulse={pulse} />
      <rect x="18" y="297" width="436" height="42" rx="3" className={`diagram-update${active(3) ? ' is-active' : ''}`} />
      <text x="236" y="323" textAnchor="middle" className="diagram-blue-note">Update existing prompts and attention projections</text>
      <text x="18" y="368" className="diagram-small">No new inference module · No additional trainable parameters · Image-level labels, no lesion masks</text>
    </svg>
    <div className="workflow-mobile">
      <p className="mini-label">Original inference</p>
      <ol><li>MRI image → visual encoder + prompts</li><li><strong>A / N attention: raw logits</strong></li><li>Sigmoid contrast → feature weighting</li><li>Pooling → Q-Former → Ano tokens</li><li>Language model → answer</li></ol>
      <p className="flow-bypass">Projected visual tokens and text also enter the language model directly.</p>
      <div className="mobile-branch"><p className="mini-label">My branch · before sigmoid · training only</p><ol><li><strong>LCE</strong> — softsign(A − N) from raw logits</li><li><strong>MIL</strong> — smooth pooling, supervised by a normal/abnormal image label</li><li><strong>CABG</strong> — the auxiliary gradient and LM gradient reference determine the budget</li><li>Update existing prompts and attention projections; both original inference paths stay unchanged</li></ol></div>
    </div>
    <div className="workflow-walkthrough">
      <div className="workflow-controls" role="group" aria-label="Workflow explanation controls">
        <span className="workflow-position">{step === null ? 'Full diagram' : `Step ${step + 1} of 4`}</span>
        <button type="button" onClick={() => selectStep(step - 1)} disabled={step === null || step === 0}>← Previous</button>
        <button type="button" onClick={() => selectStep(step === null ? 0 : step + 1)} disabled={step === 3}>Next →</button>
        <button type="button" className="workflow-play" onClick={playOnce}>{playing ? 'Stop playback' : 'Play once'}</button>
        <button type="button" className="workflow-reset" onClick={() => selectStep(null)} disabled={step === null}>Full diagram</button>
      </div>
      <div className="workflow-explanation" aria-live="polite" aria-atomic="true">
        {step === null ? <p>Use Next to explain one step at a time, or Play once to walk through all four steps.</p> : <p><strong>{steps[step].title}.</strong> {steps[step].text}</p>}
      </div>
    </div>
    <ol className="workflow-static-explanation">{steps.map(item => <li key={item.title}><strong>{item.title}.</strong> {item.text}</li>)}</ol>
    <figcaption>Redrawn from MEDIC-AD, Fig. 3 / §3.2, and the v1.2 implementation. Only the auxiliary training branch uses softsign; the original answer-generation pathway remains in place.</figcaption>
  </figure>;
}

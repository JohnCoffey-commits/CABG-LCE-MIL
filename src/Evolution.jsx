import { useEffect, useRef, useState } from 'react';
import evidence from './data/evolution.json';

function Fields({node}) {
  return <dl className="evidence-fields">{[
    ['Idea', node.idea], ['Test', node.test], ['What broke', node.broke],
    ['What I learned', node.learned], ['Status', node.statusDetail],
  ].map(([label, text]) => <div key={label}><dt>{label}</dt><dd>{text}</dd></div>)}</dl>;
}

export default function Evolution() {
  const [selected, setSelected] = useState(null);
  const modal = useRef(null);
  const dialogActive = useRef(false);
  const opener = useRef(null);
  const closeButton = useRef(null);
  const current = selected === null ? null : evidence.nodes[selected];

  useEffect(() => {
    if (selected === null) return;
    dialogActive.current = true;
    if (!modal.current.open) modal.current.showModal();
    document.body.classList.add('modal-open');
    modal.current.scrollTop = 0;
    closeButton.current.focus({preventScroll: true});
  }, [selected]);

  function close() {
    // A queued native close event must not clear a newly reopened dialog.
    if (modal.current.open || !dialogActive.current) return;
    dialogActive.current = false;
    document.body.classList.remove('modal-open');
    setSelected(null);
    opener.current?.focus({preventScroll: true});
  }

  function dismiss() {
    modal.current.close();
    close();
  }

  function keepFocusInDetails(event) {
    if (event.key !== 'Tab') return;
    const targets = [...modal.current.querySelectorAll('button:not(:disabled), a[href], summary, [tabindex="0"]')]
      .filter(element => element.getClientRects().length > 0);
    const first = targets[0];
    const last = targets[targets.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return <section id="evolution" aria-labelledby="evolution-heading">
    <h2 id="evolution-heading">3. Research process: how the idea evolved</h2>
    <p>The project progressed through successive falsification: each controlled failure exposed a specific limitation and directly motivated the next mechanism.</p>
    <div className="effort-summary" aria-label="Verified research activity"><span><strong>13 runs</strong> FB-MAQ pilot</span><span><strong>2 calibrations</strong> LAD-MIL</span><span><strong>12 images / 36 records</strong> mechanism diagnosis</span><span><strong>24 blocks</strong> per full trajectory</span></div>
    <div className="timeline-key"><span><i className="red" /> Experiment / gate failure</span><span><i className="amber" /> Rejected / inconclusive / blocked</span><span><i className="green" /> Mechanism insight</span><span><i className="blue" /> Retained candidate</span></div>
    <div className="timeline">{evidence.nodes.map((node, index) => <article key={node.id} className={`timeline-card tone-${node.tone}`}>
      <div className="card-top"><span className="node-number">{String(index + 1).padStart(2, '0')}</span><span className="status">{node.status}</span></div>
      <h3>{node.title}</h3><p className="node-subtitle">{node.subtitle}</p><p className="node-short">{node.short}</p>
      {node.secondaryStatus && <p className="secondary-status">{node.secondaryStatus}</p>}
      <button className="node-open" aria-label={`Read details: ${node.title}`} aria-haspopup="dialog" aria-controls="evolution-dialog" aria-expanded={selected === index}
        onClick={event => {opener.current = event.currentTarget; setSelected(index);}}>{node.button}<span aria-hidden="true">↗</span></button>
    </article>)}</div>
    <p className="small-note">8 turning points, covering 13 ideas and interventions. Design rejections were not executed trials; mechanism and engineering checks are not performance experiments.</p>
    <details className="speaking-notes"><summary>Detailed Explanation</summary><div><p>The project developed through controlled tests that exposed different limitations: multi-scale fusion, a fixed loss weight, gradient dead zones, a saturated evidence path and late spatial concentration. Each finding changed the next design decision.</p><p>I first tried multi-scale fusion within a fixed token budget. The implementation worked, but the internal pilot did not support further development. Direct supervision of anomaly evidence came next; two calibrations exposed unstable loss weighting. Same-forward M0/M1/M2 diagnostics then separated dead gradients, an incorrect support boundary and sigmoid saturation. I rejected projection and temperature search at the design stage before developing LCE. Full training revealed late concentration, and three repair attempts missed the spatial gate. The key insight came from a shared-checkpoint branch: keeping optimizer state and switching off later LCE prevented concentration under those conditions. After correcting backward reproducibility, the retained comparisons supported score gains from two starting points on the development set. These results establish an implemented, internally reproducible method and conditional mechanism insights; actual answer gains on independent cases remain unmeasured.</p></div></details>
    <dialog id="evolution-dialog" ref={modal} onClose={close} onCancel={event => {event.preventDefault(); dismiss();}} onKeyDown={keepFocusInDetails} aria-labelledby="detail-title" onClick={event => {
      if (event.target !== modal.current) return;
      const r = modal.current.getBoundingClientRect();
      if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dismiss();
    }}>
      <div className="dialog-top"><span>Idea evolution / {selected === null ? '' : String(selected + 1).padStart(2, '0')}</span><button ref={closeButton} onClick={dismiss} aria-label="Close details">×</button></div>
      {current && <div className={`dialog-content tone-${current.tone}`} key={current.id}>
        <h2 id="detail-title">{current.title}</h2><p className="detail-kind"><span className="status">{current.status}</span>{current.kind}</p>
        <Fields node={current} />
        {current.parts && <div className="sub-ideas">{current.parts.map(part => <details key={part.title} className={`tone-${part.tone}`}><summary>{part.title}<span className="status">{part.status}</span></summary><Fields node={part} /></details>)}</div>}
        {current.note && <p className="detail-note">{current.note}</p>}
        <div className="detail-sources"><strong>Evidence</strong>{current.sources.map(id => {const source = evidence.sources.find(s => s.id === id);return <p key={id}>[{id}] {source.title} — {source.section}</p>;})}</div>
      </div>}
      <div className="dialog-bottom"><button onClick={() => setSelected(i => i - 1)} disabled={selected === null || selected === 0}>← Previous</button><span>{selected === null ? '' : selected + 1} / 8</span><button onClick={() => setSelected(i => i + 1)} disabled={selected === null || selected === 7}>Next →</button></div>
    </dialog>
  </section>;
}

export function Sources() {
  return <details className="sources" id="sources"><summary>References, evidence and disclosure</summary>
    <p>Original paper: Park et al., <a href="https://arxiv.org/abs/2603.27176v2" target="_blank" rel="noreferrer">MEDIC-AD: Towards Medical Vision-Language Model’s Clinical Intelligence</a>, arXiv:2603.27176v2 (2026), especially Fig. 3 and §3.2.</p>
    <ol>{evidence.sources.map(source => <li key={source.id}><strong>[{source.id}] {source.title}</strong> — {source.section}.</li>)}</ol>
    <p>Numerical verification refers to separate implementation and evidence recomputation by the same research operator, not an independent research group or clinical replication. Research snapshot: <code>cb5a33c</code>. AI-assisted implementation and writing were used; the presenter is responsible for reviewing the research claims and following course disclosure requirements.</p>
    <p>This site serves only reviewed aggregate values and diagrams. No MRI images, case identifiers or original research artifacts are included.</p>
  </details>;
}

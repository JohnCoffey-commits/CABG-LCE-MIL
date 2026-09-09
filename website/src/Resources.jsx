import evidence from './data/evolution.json';

const ideaCount = evidence.nodes.reduce((total, node) => total + (node.parts?.length ?? 1), 0);
const activity = [
  {count: 13, label: 'FB-MAQ pilot runs', detail: 'Frozen-adapter evaluations'},
  {count: 2, label: 'Gradient calibrations', detail: 'Independent LAD-MIL subsets'},
  {count: 36, label: 'Diagnostic records', detail: 'M0 / M1 / M2 on 12 images'},
  {count: ideaCount, label: 'Ideas & interventions', detail: `Across ${evidence.nodes.length} turning points`},
];
const outputs = [
  {title: 'Full training & recovery records', text: 'Complete trajectories with checkpoint restore checks.'},
  {title: 'Controlled comparisons', text: 'Matched branches, ablations and fixed-endpoint results.'},
  {title: 'Mechanism diagnostics', text: 'Same-forward gradient and evidence-path measurements.'},
  {title: 'Auditable evidence', text: 'Execution traces, verification results and completion audits.'},
];

export default function Resources() {
  return <section id="resources" aria-labelledby="resources-heading">
    <h2 id="resources-heading">6. Research effort &amp; resources</h2>
    <p>A documented process of experimentation, diagnosis and revision.</p>

    <dl className="resource-infrastructure">
      <div><dt>Cloud platform</dt><dd>JarvisLabs.ai</dd></div>
      <div><dt>GPU</dt><dd>NVIDIA L4</dd></div>
    </dl>

    <h3>Selected research activity</h3>
    <ul className="resource-activity" aria-label="Selected documented research activity">
      {activity.map(item => <li key={item.label}>
        <strong className="resource-count">{item.count}</strong>
        <h4>{item.label}</h4>
        <p>{item.detail}</p>
      </li>)}
    </ul>
    <p className="small-note">These are distinct activity categories, not a cumulative count of training runs.</p>

    <div className="resource-outputs">
      <h3>Research outputs</h3>
      <ul>{outputs.map(item => <li key={item.title}><strong>{item.title}</strong><p>{item.text}</p></li>)}</ul>
    </div>

    <details className="technical-note resource-breakdown">
      <summary>View activity breakdown and evidence</summary>
      <dl>
        <div><dt>13 pilot evaluations</dt><dd>Stage 2G ran six matched B0/A3 pairs and one contextual reference in fresh processes, using retained adapters. These were evaluations, not 13 new training runs. [S03]</dd></div>
        <div><dt>2 gradient calibrations</dt><dd>The first LAD-MIL calibration used four images; the second used a separate 12-image subset. Neither selected an eligible loss weight, and formal matched training did not start. [S04, S05]</dd></div>
        <div><dt>36 same-forward diagnostic records</dt><dd>Twelve images were each examined under M0, M1 and M2 using the same forward graph: 12 × 3 = 36 records, with no optimizer update in this gate. These are mechanism records, not 36 independent cases. [S06]</dd></div>
        <div><dt>{ideaCount} ideas across {evidence.nodes.length} turning points</dt><dd>The research-process cards group method proposals, diagnostics and interventions. This includes two designs rejected before a performance experiment; it is not a count of executed experiments. [S01–S17]</dd></div>
      </dl>
      <p className="small-note">Full-training, recovery and controlled-comparison evidence is documented separately in [S10–S16].</p>
      <a href="#sources" onClick={() => {document.getElementById('sources').open = true;}}>Open the reference list →</a>
    </details>
    <p className="small-note">Selected documented activity from the MEDIC-AD / CABG-LCE-MIL v1.2 research record.</p>
  </section>;
}

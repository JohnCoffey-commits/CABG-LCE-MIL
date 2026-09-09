import { useState } from 'react';
import results from './data/results.json';

function Figure({id, title, image, alt, children, data, controls, source}) {
  return <figure className="result-figure" id={id}>
    <div className="figure-title"><h3>{title}</h3>{controls}</div>
    <a className="chart-open" href={`/figures/${image}.svg`} target="_blank" rel="noreferrer" aria-label={`Open larger chart: ${title}`}>
      <img className="data-chart" src={`/figures/${image}.svg`} alt={alt} width="590" height="360" />
    </a>
    <figcaption>{children}</figcaption>
    <div className="figure-source"><span>{source}</span><span><a href={`/figures/${image}.svg`} download>SVG</a><a href={`/data/${data}.csv`} download>Data</a></span></div>
  </figure>;
}

export default function Results() {
  const [metric, setMetric] = useState('auroc');
  const controls = <div className="metric-toggle" aria-label="Score metric"><button aria-pressed={metric === 'auroc'} onClick={() => setMetric('auroc')}>AUROC</button><button aria-pressed={metric === 'ap'} onClick={() => setMetric('ap')}>AP</button></div>;
  return <section id="results" aria-labelledby="results-heading">
    <h2 id="results-heading">4. Results and what they tell us</h2>
    <p>The completed comparisons show a stronger internal anomaly-ranking signal and reveal when spatial concentration can emerge. These are <strong>development-set findings</strong>, not estimates of clinical accuracy.</p>
    <div className="experiment-context"><span><strong>64</strong> unique training images</span><span><strong>32</strong> reused development images (20 abnormal / 12 normal)</span><span><strong>24</strong> steps per complete trajectory</span></div>
    <div className="figures-grid">
      <Figure id="score-figure" title="A. LCE strengthens the anomaly score" image={`full-none-${metric}`} data="history" controls={controls}
        source="Corrected execution · S15" alt={metric === 'auroc' ? 'Development AUROC. Initialization 43: LM-only 0.6875, full LCE 0.9917. Initialization 44: 0.4167 and 0.9708.' : 'Development average precision. Initialization 43: LM-only 0.8304, full LCE 0.9955. Initialization 44: 0.5651 and 0.9815.'}>
        <strong>Full exceeds none at both retained starts.</strong> Training labels, starting state and schedule are matched within each start. This is an objective ablation; LM-only is not the untouched original-paper model.
      </Figure>
      <Figure id="spatial-figure" title="B. Spatial spread is not performance" image="spatial-and-ranking" data="history" source="Corrected execution · S15"
        alt="Eight development endpoints. Full at initialization 43 combines high AUROC 0.9917 with narrow support 70.8; none at initialization 44 has broad support 733.6 but AUROC 0.4167. Early has high AUROC and remains stable at both starts.">
        <strong>A dispersed map can still rank poorly.</strong> Early-only retained high ranking and spatial stability at both starts; full concentrated severely only at start 43. The histories do not establish an optimal cutoff or a dose-matched timing law.
      </Figure>
      <Figure id="repair-figure" title="C. Three repairs missed the locked gate" image="stability-repairs" data="repairs" source="Historical D4 repair study · S10"
        alt="Median effective support: cosine taper 105.5, spatial guard 153.1, guard plus first-moment reset 153.1; all below the preregistered floor of 256.">
        <strong>Ranking gains survived, but spatial stability was unresolved.</strong> The guard helped descriptively; a first-moment reset added no material endpoint benefit. Tail criteria also failed. This historical execution is separate from panels A and B.
      </Figure>
      <Figure id="causal-figure" title="D. A controlled branch changed the outcome" image="shared-parent-causal" data="causal" source="Historical shared-parent study · S11"
        alt="From the same block-12 parent with median support 374.6: continuing LCE ends at 43.4, LCE off with optimizer state kept at 420.6, and LCE off with first moments reset at 500.6.">
        <strong>Removing future LCE prevented concentration from this parent.</strong> Keeping optimizer state was sufficient; AUROC/AP stayed at 0.9917/0.9955. This is local causal evidence for continued LCE, not proof that the dynamic controller is necessary.
      </Figure>
    </div>
    <p className="small-note">Click a chart to open its full-size vector version. Effective support describes how broadly spatial weights are distributed over 1,024 positions; it is not lesion-localization accuracy. Two selected initializations do not supply a population uncertainty estimate. No significance or external-validation claim is implied.</p>
    <details className="numeric-results"><summary>All four training histories: exact endpoint table</summary><div className="table-scroll"><table><caption>Same reused 32-image development set. Early = LCE at steps 1–12; late = steps 13–24; full = steps 1–24. LM training continues throughout.</caption><thead><tr><th>Start</th><th>LCE history</th><th>AUROC</th><th>AP</th><th>Support median</th><th>Spatial state</th></tr></thead><tbody>{results.history.map(row => <tr key={`${row.seed}-${row.history}`}><td>{row.seed}</td><th>{row.history}</th><td>{row.auroc.toFixed(4)}</td><td>{row.ap.toFixed(4)}</td><td>{row.support.toFixed(1)}</td><td>{row.stable ? 'Stable' : 'Severe'}</td></tr>)}</tbody></table></div></details>
    <div className="repro-note"><h3>What makes these results reviewable</h3><p>After fixing VPT backward nondeterminism, two full 24-step trajectories reproduced exactly in the corrected runtime, with fresh-process recovery checks. Matched comparisons retain checkpoints, optimizer state, raw traces and separate numerical verification. This establishes internal reproducibility within the tested execution setup.</p></div>
  </section>;
}

import Workflow from './Workflow.jsx';
import Evolution, { Sources } from './Evolution.jsx';
import Results from './Results.jsx';
import Math from './Math.jsx';
import Resources from './Resources.jsx';

export default function App() {
  return <>
    <a className="skip-link" href="#paper">Skip to content</a>
    <div className="page">
      <header className="site-header">
        <a className="site-name" href="#paper"><strong>CABG-LCE-MIL</strong><span>A research project built on MEDIC-AD</span></a>
        <nav aria-label="Primary navigation">
          <a className="github-link" href="https://github.com/JohnCoffey-commits/CABG-LCE-MIL" target="_blank" rel="noreferrer" aria-label="View the CABG-LCE-MIL source code on GitHub">GitHub <span aria-hidden="true">↗</span></a>
          <a href="#paper">Background</a><a href="#idea">My idea</a><a href="#evolution">Process</a><a href="#results">Results</a><a href="#conclusion">Conclusion</a>
        </nav>
      </header>
      <main>
        <section id="paper" className="paper-section" aria-labelledby="paper-heading">
          <p className="project-meta">UTS Master of IT · 6-credit Research Project · v1.2</p>
          <h1 id="paper-heading">Training better anomaly evidence in a medical VLM</h1>
          <div className="background-columns"><div>
            <h3>The original paper</h3>
            <p><a href="https://arxiv.org/abs/2603.27176v2" target="_blank" rel="noreferrer">MEDIC-AD (Park et al., 2026)</a> extends a medical vision-language model with three capabilities: detecting abnormalities using <code>&lt;Ano&gt;</code> tokens, comparing scans using <code>&lt;Diff&gt;</code> tokens, and generating visual heatmaps.</p>
            <p>Its anomaly processor contrasts normal and abnormal attention, uses the resulting map to weight visual features, and turns them into tokens for the language model. I focused on this first, anomaly-detection stage.</p>
          </div><aside className="research-question"><h3>The question I took forward</h3><p>Having an anomaly-aware architecture does not tell us whether its internal evidence is learning the signal we want.</p><p><strong>Can image-level supervision strengthen that evidence, while keeping MEDIC-AD’s answer-generation pathway unchanged?</strong></p></aside></div>
          <p className="scope-note">The gradient dead zones and saturation issues below were diagnosed in my proposed auxiliary objectives and local model setup. They are not presented as performance failures established by the original paper.</p>
        </section>

        <section id="idea" aria-labelledby="idea-heading">
          <h2 id="idea-heading">2. My idea: supervise the evidence, balance its influence</h2>
          <p><strong>CABG-LCE-MIL v1.2</strong> adds a training signal directly to normal-versus-abnormal evidence. It combines logit contrast, image-level multiple-instance learning and a measured gradient budget.</p>
          <Workflow />
          <div className="mechanism-explanation">
            <article><h3><span>01</span> LCE: a better gradient path</h3><p>Take the raw abnormal and normal logits before sigmoid, then compute <code>softsign(A − N)</code>. This bypasses the two sigmoid derivatives that attenuated the previous auxiliary signal.</p><p className="small-note">Softsign still weakens at extreme gaps; it does not guarantee stable training.</p></article>
            <article><h3><span>02</span> MIL: learn from image labels</h3><p>Treat the image as a bag of patches. Smooth LogSumExp pooling lets multiple locations contribute; a symmetric softplus loss gives both normal and abnormal examples a learning signal.</p><p className="small-note">No new lesion masks or per-patch labels are required.</p></article>
            <article><h3><span>03</span> CABG: control the update</h3><p>Measure class-balanced gradients from the language and auxiliary objectives. A moving estimate sets the LCE weight; a trust cap limits its norm relative to the language gradient.</p><p className="small-note">The update is balanced block by block, rather than relying on one fixed λ.</p></article>
          </div>
          <details className="technical-note"><summary>Mechanism details and what stays unchanged</summary>
            <div className="formula-grid">
              <div><strong>Auxiliary evidence</strong>
                <Math display tex={String.raw`c = \frac{A-N}{1+\lvert A-N\rvert}`} />
                <p>FP32 subtraction, averaged over four selected layers. Smooth pooling uses <Math tex={String.raw`\tau = 1/\log(1024)`} />; the image-label loss is:</p>
                <Math display tex={String.raw`\ell = \operatorname{softplus}(-y\cdot s)`} />
                <p>Here <Math tex="s" /> is the pooled score and <Math tex={String.raw`y\in\{-1,+1\}`} /> is the image label.</p>
              </div>
              <div><strong>Applied update</strong>
                <Math display tex={String.raw`g = g_{\mathrm{LM}} + \lambda_t\, g_{\mathrm{LCE}}`} />
                <p>On the nine-tensor shared support: a 0.10 EMA ratio target and a 0.20 norm cap, before global clipping. All 21 trainable tensors remain available to the LM objective.</p>
              </div>
            </div>
            <p>The original sigmoid-difference inference map, anomaly-token count and model architecture remain unchanged. The added objective introduces no trainable parameter or new contrast temperature. “Balanced” describes gradient allocation; controller necessity is not established by these experiments. [S08, S09]</p>
          </details>
        </section>

        <Evolution />
        <Results />

        <section id="conclusion" aria-labelledby="conclusion-heading">
          <h2 id="conclusion-heading">5. What this project achieved</h2>
          <p className="final-statement">I developed and tested a way to directly train MEDIC-AD’s anomaly evidence, found a consistent ranking gain in the two retained development comparisons, and identified a conditional failure mode of sustained auxiliary training.</p>
          <ol className="contributions">
            <li><strong>A method contribution.</strong> CABG-LCE-MIL combines a logit-contrast objective, smooth MIL supervision and an adaptive gradient budget, implemented without adding an inference module.</li>
            <li><strong>A scientific finding.</strong> LCE exposure improved internal score discrimination over matched LM-only training at both starts. Controlled branching showed that continuing LCE can participate in forming spatial concentration from a specific state; wider spatial support alone does not imply a better detector.</li>
            <li><strong>A reproducible experimental system.</strong> The work progressed from gradient diagnostics to complete training trajectories, recovery checks, matched ablations and independently recomputed evidence. Failed and inconclusive attempts remain part of the research record.</li>
          </ol>
          <div className="conclusion-boundary"><h3>The final boundary</h3><p>This is a completed method-development and mechanism-investigation project with internal empirical support. It does <strong>not</strong> yet establish better anomaly answers on independent patients, superiority to a simple supervised head, clinical usefulness or an optimal early cutoff.</p><p>The generation pipeline and one frozen linear baseline are ready. Their four-image development check was already at the answer ceiling. A bounded review of eight public MRI source families did not establish a cohort meeting the combined independence, permission, visible-slice label and precision requirements.</p><p><strong>Current conclusion:</strong> retain the method and evidence as the course outcome; freeze further development-set optimization. The next meaningful extension is a qualified independent-case comparison, not another timing or coefficient search.</p></div>
          <details className="speaking-notes"><summary>Detailed Explanation</summary><p>Taken as the final outcome, this project contributes an implemented method for directly supervising anomaly evidence, a consistent direction of score improvement across two retained development-set starting points, and controlled evidence linking sustained LCE to spatial concentration under the tested conditions. I also established reproducible full training and recovery. Better anomaly answers on independent cases remain unmeasured. The completed contribution is therefore method development, failure diagnosis and mechanism investigation; clinical effectiveness has not been established.</p></details>
        </section>
        <Resources />
      </main>
      <footer><div className="footer-line"><span>MEDIC-AD / CABG-LCE-MIL v1.2</span><span>Evidence through 8 September 2026</span><button onClick={() => window.print()}>Print / save PDF</button></div><Sources /></footer>
    </div>
  </>;
}

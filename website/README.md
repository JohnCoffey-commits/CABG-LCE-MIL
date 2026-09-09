# CABG-LCE-MIL research briefing

A compact academic website for presenting the development of **CABG-LCE-MIL v1.2**, a training objective investigated on top of MEDIC-AD. It explains the research question, mechanism, controlled failures, retained evidence, limitations and research effort.

**Live website:** https://medic-ad-briefing.vercel.app/

The briefing is designed for a short supervisor presentation. Scientific claims are deliberately bounded: the reported capability results are internal development-set evidence, while independent-cohort capability remains unmeasured because a qualifying cohort has not been established.

## Run locally

Requirements: Node.js 22 or newer.

```sh
npm ci
npm run dev
```

Open http://localhost:5173/. To test the production build:

```sh
npm run build
npm run preview
```

## Repository contents

- `src/App.jsx` — paper context, research question and conclusions.
- `src/Workflow.jsx` — interactive original-path and training-objective diagram.
- `src/Math.jsx` — accessible equations rendered with bundled KaTeX.
- `src/Evolution.jsx` — eight clickable research-evolution cards.
- `src/Results.jsx` — result figures, metric switcher and numerical table.
- `src/Resources.jsx` — research effort and compute resources.
- `src/data/` — reviewed, case-free presentation content.
- `public/figures/` — static research figures used by the website.
- `public/data/` — aggregate CSV downloads without patient or image identifiers.

This repository contains the briefing website source and reviewed aggregate presentation assets. It does not contain the MEDIC-AD research repository, MRI data, case manifests, protected-test material, model checkpoints, optimizer state, raw experiment artifacts, credentials or private infrastructure paths.

## Interaction and accessibility

The workflow can be presented in four controlled steps and does not auto-loop. Every evolution card opens a detailed evidence view using keyboard-accessible controls. The site includes responsive layouts, print rules and reduced-motion behaviour. KaTeX fonts and mathematical rendering are bundled locally; there is no runtime analytics or external data service.

## Deployment

The project is built by Vercel from `main` using `npm ci` and `npm run build`; the static output is `dist/`. Pushes to `main` update the production deployment, while pull requests can receive preview deployments through Vercel's Git integration.

## Research attribution

The starting point is [MEDIC-AD (Park et al., 2026)](https://arxiv.org/abs/2603.27176v2). CABG-LCE-MIL v1.2 and the evidence synthesis shown here were developed as a University of Technology Sydney Master of IT Research Project. Original and modified mechanisms are distinguished throughout the briefing.

The dependencies retain their respective upstream licences. No licence is granted here for external reuse of the research content or source beyond what applicable law permits.

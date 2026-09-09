import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export default function Math({tex, display = false}) {
  const html = useMemo(() => katex.renderToString(tex, {
    displayMode: display,
    output: 'htmlAndMathml',
    throwOnError: true,
    strict: 'error',
    trust: false,
  }), [tex, display]);
  const Element = display ? 'div' : 'span';
  // Only KaTeX output from the fixed, reviewed expressions is inserted here.
  return <Element className={display ? 'math-display' : 'math-inline'} dangerouslySetInnerHTML={{__html: html}} />;
}

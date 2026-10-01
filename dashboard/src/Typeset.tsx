import { useMemo } from "react";
import { renderToString } from "katex";
import "katex/dist/katex.min.css";

export default function Typeset({
  tex,
  displayMode = false,
}: {
  tex: string;
  displayMode?: boolean;
}) {
  // Only source-controlled equations enter this renderer.
  const html = useMemo(
    () =>
      renderToString(tex, {
        displayMode,
        output: "htmlAndMathml",
        throwOnError: true,
        strict: "error",
        trust: false,
      }),
    [tex, displayMode],
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

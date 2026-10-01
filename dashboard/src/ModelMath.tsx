import { lazy, Suspense } from "react";

const Typeset = lazy(() => import("./Typeset"));

export function InlineMath({ tex }: { tex: string }) {
  return (
    <span className="method-inline-math">
      <Suspense fallback={null}>
        <Typeset tex={tex} />
      </Suspense>
    </span>
  );
}

export function Equation({ label, tex }: { label: string; tex: string }) {
  return (
    <figure className="method-equation">
      <figcaption>{label}</figcaption>
      <div
        className="method-math"
        role="region"
        aria-label={label}
        tabIndex={0}
      >
        <Suspense fallback={null}>
          <Typeset tex={tex} displayMode />
        </Suspense>
      </div>
    </figure>
  );
}

export function RatingMath() {
  return (
    <>
      <Equation
        label="Seconds per reference lap"
        tex={String.raw`r_k^{(m)} = 0.9\left(x_k^{(m)} - \frac{1}{N}\sum_{j=1}^{N}x_j^{(m)}\right)`}
      />
      <p>
        <InlineMath tex="x_k^{(m)}" /> is entry <InlineMath tex="k" />
        ’s percentage-point pace in joint posterior draw <InlineMath tex="m" />.{" "}
        <InlineMath tex="N" /> is the number of rated entries in that field.
        Drivers and cars are centred separately in every draw. The factor 0.9
        converts one percentage point to seconds on a 90-second lap. This is a
        linear reference-lap conversion of log pace.
      </p>
      <Equation
        label="Pace estimate and 90% interval"
        tex={String.raw`\begin{aligned}
          \widehat r_k &= Q_{0.50}(r_k) \\
          I_{90,k} &= [Q_{0.05}(r_k),\,Q_{0.95}(r_k)]
        \end{aligned}`}
      />
      <Equation
        label="Rank probabilities"
        tex={String.raw`\begin{aligned}
          R_k^{(m)} &= 1 + \sum_{j\ne k}\mathbf{1}\{r_j^{(m)} > r_k^{(m)}\} \\
          P(\text{fastest }k) &\approx \frac{1}{M}\sum_{m=1}^{M}\mathbf{1}\{R_k^{(m)}=1\} \\
          P(\text{top 3 }k) &\approx \frac{1}{M}\sum_{m=1}^{M}\mathbf{1}\{R_k^{(m)}\le 3\}
        \end{aligned}`}
      />
      <p>
        <InlineMath tex="Q_p" /> is a posterior quantile;{" "}
        <InlineMath tex={String.raw`\mathbf{1}\{\cdot\}`} /> is 1 when the
        condition holds and 0 otherwise. Each of the <InlineMath tex="M" />{" "}
        shared draws gives a complete ordering. Rank intervals use its 5th and
        95th percentiles; numerical ties are ordered consistently. The displayed
        table sorts median pace, which can differ from median posterior rank.
      </p>
    </>
  );
}

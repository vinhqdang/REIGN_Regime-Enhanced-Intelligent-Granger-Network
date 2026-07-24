# ANTE: Anytime-valid Nonparametric Test of synErgy

*Sequential e-processes for group (synergistic) causal effects.*

This note sketches the test statistic and the martingale/e-process construction
for a sequential test of *synergistic* (group) causality: the hypothesis that a
set of causes acts on an effect only through the *sum of its parts*, against the
alternative that the joint effect exceeds that sum.

---

## 1. Problem setup

Let observations arrive as a stream $Z_1, Z_2, \dots$ with $Z_t = (X_t, \mathbf{T}_t, C_t)$:

* $C_t \in [0,1]$ — the effect (binary outcome or bounded score at a target node);
* $\mathbf{T}_t = (T_t^{(1)}, \dots, T_t^{(k)})$ — $k$ binary candidate causes;
* $X_t$ — optional covariates.

We work in the potential-outcomes model with $C_t(\mathbf{t})$ the outcome that
would be observed under the joint intervention $\mathrm{do}(\mathbf{T}=\mathbf{t})$,
and define cell means $\mu_{\mathbf t} = \mathbb{E}[C(\mathbf t)]$.

Identification assumptions (per time step): **consistency**, **positivity**
($\pi(\mathbf t \mid X) \ge \pi_{\min} > 0$), and **unconfoundedness**
$\mathbf T_t \perp \{C_t(\mathbf t)\} \mid X_t$. Under a randomized/known design
(our experiments) $\pi(\mathbf t\mid X)=\pi(\mathbf t)$ is known and all three hold
by construction; under observational streams $\pi$ is estimated (§6).

### The synergy null on two scales

For two causes $A,B$ the **additive interaction contrast** (the causal-inference
form of Rothman's synergy / the population RERI on the risk-difference scale) is

$$\theta \;=\; \mu_{11} - \mu_{10} - \mu_{01} + \mu_{00}.$$

$\theta=0$ is exactly *"the joint effect of $(A,B)$ equals the sum of the two
individual effects"* — no synergy. $\theta>0$ is super-additive synergy
(a causal-pie / AND mechanism); $\theta<0$ is sub-additive/antagonistic.

The **information-theoretic** synergy of Partial Information Decomposition (and
its effective-information variant used by SURD/PEID) targets the same idea on a
different scale:

$$\mathrm{Syn}(A,B\!\to\!C) \;=\; I(A,B;C) - \big[\,\mathrm{Red}(A,B;C) + U(A;C) + U(B;C)\,\big],$$

or, under a maximum-entropy intervention that renders the sources independent,
$\mathrm{Syn}_{EI} = EI(A,B\!\to\!C) - \sum_i EI(T^{(i)}\!\to\!C)$. Both reduce to
the same qualitative null — **joint $=$ sum of parts** — measured additively or
in bits. §2–§4 develop the additive-scale test in full (it is exactly
identified, bounded, and needs no density estimation); §5 gives the
information-theoretic instantiation via the same betting scaffold.

---

## 2. A per-sample score that is centred under the null

Under a known design the inverse-probability-weighted **interaction score**

$$\psi_t \;=\; c(A_t,B_t)\,\frac{C_t}{\pi(A_t,B_t)}, \qquad
c(a,b) = (-1)^{(1-a)+(1-b)} = \begin{cases} +1 & (a,b)\in\{(1,1),(0,0)\}\\ -1 & (a,b)\in\{(1,0),(0,1)\}\end{cases}$$

is unbiased for the interaction contrast:

$$\mathbb{E}[\psi_t] = \sum_{a,b} c(a,b)\,\pi(a,b)\frac{\mu_{ab}}{\pi(a,b)}
= \mu_{11}-\mu_{10}-\mu_{01}+\mu_{00} = \theta.$$

Hence **under $H_0:\theta=0$, $\mathbb{E}[\psi_t]=0$.** With $C\in[0,1]$ and
$\pi(a,b)\ge \pi_{\min}$, the score is bounded, $|\psi_t|\le c_\psi := 1/\pi_{\min}$
(for a balanced $2\times2$ design $\pi_{\min}=1/4$, so $\psi_t\in[-4,4]$).
Boundedness is what makes the betting martingale below exactly valid — no
asymptotics, no variance assumptions.

*Doubly-robust variant.* Replacing $\psi_t$ by the AIPW score
$\phi_t=\sum_{a,b}c(a,b)\big[\frac{\mathbb 1\{A_t=a,B_t=b\}}{\pi(a,b\mid X_t)}(C_t-\hat\mu_{ab}(X_t))+\hat\mu_{ab}(X_t)\big]$
keeps $\mathbb E[\phi_t\mid X_t]=\theta(X_t)$ Neyman-orthogonal, so slowly-learned
nuisances $(\hat\mu,\hat\pi)$ do not break centering — the key to the
observational extension (§6).

---

## 3. The betting e-process (test martingale)

Fix a significance level $\alpha$. Start with unit capital $E_0=1$ and bet a
**predictable** fraction $\lambda_t$ (a function of $\psi_1,\dots,\psi_{t-1}$ only):

$$\boxed{\;E_t \;=\; \prod_{s=1}^{t}\bigl(1 + \lambda_s\,\psi_s\bigr)\;}
\qquad |\lambda_s|\le \frac{\gamma}{c_\psi}\;(\gamma<1)\ \text{so } 1+\lambda_s\psi_s>0.$$

**Claim (validity).** Under any $P\in H_0$, $(E_t)$ is a nonnegative martingale
with $\mathbb E[E_t]=1$.

*Proof.* $\lambda_s$ is $\mathcal F_{s-1}$-measurable and $\mathbb E[\psi_s\mid\mathcal F_{s-1}]=0$, so
$\mathbb E[E_t\mid\mathcal F_{t-1}] = E_{t-1}\bigl(1+\lambda_t\,\mathbb E[\psi_t\mid\mathcal F_{t-1}]\bigr)=E_{t-1}.$
Non-negativity holds because each factor is positive. $\square$

**Anytime-valid test.** By **Ville's inequality** for nonnegative supermartingales,

$$P_{H_0}\Bigl(\exists\,t\ge 1:\; E_t \ge \tfrac1\alpha\Bigr)\;\le\;\alpha.$$

So the rule *"reject the first time $E_t\ge 1/\alpha$"* controls the type-I error
**uniformly over all stopping times** — you may monitor after every observation,
stop early, or peek indefinitely, and the guarantee holds. $E_t$ is an
*e-value* at every $t$; $1/E_t \wedge 1$ is an anytime-valid p-value.

### Two-sided test
Synergy can be positive or negative (e.g. XOR gives $\theta=-2$). Run two capital
processes — $E_t^{+}$ betting $\lambda_s\ge0$ and $E_t^{-}$ betting
$\lambda_s\le0$ — and combine by averaging,
$E_t = \tfrac12(E_t^{+}+E_t^{-})$. An average of e-processes is an e-process, so
the two-sided test is valid at level $\alpha$ against $\theta\ne0$.

### Choosing $\lambda_t$ (betting strategy)
Any predictable rule is valid; a good one *grows capital fast* under $H_1$
(minimises expected detection time — the "GRO"/growth-optimal criterion). We use
a truncated plug-in Kelly fraction from the running moments of $\psi$:

$$\lambda_t = \mathrm{clip}\!\Bigl(\frac{\hat\mu_{t-1}}{\hat v_{t-1}},\,\pm\frac{\gamma}{c_\psi}\Bigr),\quad
\hat\mu_{t-1}=\tfrac1{t-1}\!\sum_{s<t}\psi_s,\ \ \hat v_{t-1}=\tfrac1{t-1}\!\sum_{s<t}\psi_s^2,$$

with a short warm-up ($\lambda_t=0$ for $t\le t_0$). This is the aGRAPA/GRAPA
strategy of Waudby-Smith & Ramdas specialised to the interaction score.

---

## 4. Subset refinement without the full combinatorial lattice

With $k>2$ candidate causes the target is the synergistic *subset*, and the PID
lattice over all subsets is exponential. Two facts make a sequential search cheap.

1. **Hierarchical additivity (à la PEID).** Synergy is additive across partition
   refinements: refining a partition $\mathcal P\to\mathcal P'$ adds a non-negative
   increment, $\mathrm{Syn}_{\mathcal P'}=\mathrm{Syn}_{\mathcal P}+\mathrm{Syn}_{\mathcal R}$.
   This telescoping structure is exactly the *increment* structure an e-process
   consumes: we can test a chain of refinement nulls and multiply the
   corresponding e-values (e-values multiply across sequential/independent tests).
2. **e-value multiple testing.** For a family of $m$ candidate subsets we obtain
   one e-value $E^{(j)}$ per subset. Because e-values (unlike p-values) can be
   merged by *averaging* under arbitrary dependence and *multiplied* under
   independence, family-wise error is controlled by either an e-value Bonferroni
   (threshold $m/\alpha$, i.e. run each stream at level $\alpha/m$) or e-BH for
   FDR control — both far less conservative than p-value corrections and, unlike
   them, still anytime-valid.

**Algorithm (greedy anytime refinement).** Maintain a frontier of subsets;
stream data; for each frontier subset run its e-process; expand a subset into its
refinements only when its running e-value exceeds a promotion threshold; declare
a synergistic subset when its calibrated e-value crosses $1/\alpha$. Coverage is
never violated because every reported e-value is anytime-valid by construction;
the greedy schedule only affects *power/latency*, not type-I error. Our
experiment E4 verifies that with three candidates the search localises to the
true interacting pair and fires on the innocent pairs at the nominal rate.

---

## 5. Information-theoretic instantiation

To test synergy on the PID/PEID scale with the same scaffold, replace $\psi_t$
with a centred, bounded plug-in of the synergy increment. Using the
maximum-entropy-intervention identity that source-side redundancy vanishes when
the interventions are independent, the per-batch statistic

$$g_t \;=\; \widehat{EI}(A,B\!\to\!C)_t \;-\; \sum_i \widehat{EI}(T^{(i)}\!\to\!C)_t$$

is computed on disjoint mini-batches (so successive $g_t$ are independent),
recentred by a within-batch permutation null $\bar g_t^{0}$, and bet on via
$E_t=\prod_s(1+\lambda_s(g_s-\bar g_s^{0}))$. The permutation recentring makes
$\mathbb E[g_s-\bar g_s^0\mid\mathcal F_{s-1}]\le0$ under the no-synergy null, so
$(E_t)$ is an e-process by the same argument. This gives, to our knowledge, the
first *sequential* synergy monitor on the information-theoretic scale — PID/SURD/
PEID currently report a single point value with no inferential guarantee.

---

## 6. Observational and graph (GNN) instantiations

* **Observational streams.** Use the AIPW score $\phi_t$ (§2) with cross-fitted
  nuisances $\hat\mu_{ab},\hat\pi$ trained on past data only (sample-splitting
  along the stream keeps $\lambda_t$ predictable and $\phi_t$ orthogonal). This
  is the interaction-contrast analogue of anytime-valid ATE confidence sequences.
* **Graph / GNN synergy.** For an anomaly/fraud target node $C$ with candidate
  source nodes $A,B$, obtain $\hat\mu_{ab}$ from interventional forward passes of
  a learned graph SCM ($\mathrm{do}$ on the source-node features) and stream
  $\psi_t$ over incoming edges/timestamps. The test then answers *"do nodes $A$
  and $B$ jointly, not individually, drive the anomaly at $C$?"* — a question no
  existing group-causality method addresses sequentially, and one that plugs
  directly into an existing graph causal-discovery pipeline.

---

## 7. What the experiments demonstrate

| Exp | Claim | Result (see `experiments/results/`) |
|-----|-------|--------------------------------------|
| E1 | Anytime-valid type-I control; classical fixed-$n$ test inflates under monitoring | e-process FPR $\approx\alpha$; peeking-Wald FPR $\gg\alpha$ |
| E2 | Capital grows only under genuine synergy | null paths flat; AND/XOR paths cross $1/\alpha$ |
| E3 | Power $\uparrow$ and detection time $\downarrow$ with synergy strength | monotone power curve; shrinking detection latency |
| E4 | Subset refinement localises the synergistic pair | fires on true pair, others at $\approx\alpha$ |

---

## 8. Positioning / novelty

The three existing literatures on group causality — sufficient-cause interaction
(RERI, PRISM, generalized synergy index), joint potential-outcome/causal
aggregation, and information-theoretic PID/SURD/PEID — are **all fixed-sample**.
None provides a sequential, anytime-valid, or e-process test of a synergy/joint-
causation null. The contribution here is (i) a single betting-martingale test
that instantiates on *both* the additive causal-pie scale and the
information-theoretic scale, reducing them to one null; (ii) an anytime-valid
subset-refinement search exploiting hierarchical additivity; and (iii) graph/GNN
and observational instantiations. (A focused prior-art search is in
`docs/literature_review.md`; the "GAP ASSESSMENT" section there records any
near-misses.)

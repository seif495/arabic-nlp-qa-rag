                                                    An elementary introduction to information geometry
                                                                                   Frank Nielsen
                                                                        Sony Computer Science Laboratoties Inc
                                                                                   Tokyo, Japan


                                                                                              Abstract
arXiv:1808.08271v2 [cs.LG] 6 Sep 2020




                                                  In this survey, we describe the fundamental differential-geometric structures of information manifolds,
                                              state the fundamental theorem of information geometry, and illustrate some use cases of these information
                                              manifolds in information sciences. The exposition is self-contained by concisely introducing the necessary
                                              concepts of differential geometry, but proofs are omitted for brevity.
                                        Keywords: Differential geometry; metric tensor; affine connection; metric compatibility; conjugate connec-
                                        tions; dual metric-compatible parallel transport; information manifold; statistical manifold; curvature and
                                        flatness; dually flat manifolds; Hessian manifolds; exponential family; mixture family; statistical divergence;
                                        parameter divergence; separable divergence; Fisher-Rao distance; statistical invariance; Bayesian hypothesis
                                        testing; mixture clustering; αembeddings; gauge freedom


                                        1     Introduction
                                        1.1    Overview of information geometry
                                        We present a concise and modern view of the basic structures lying at the heart of Information Geometry
                                        (IG), and report some applications of those information-geometric manifolds (herein termed “information
                                        manifolds”) in statistics (Bayesian hypothesis testing) and machine learning (statistical mixture clustering).
                                            By analogy to Information Theory (IT) (pioneered by Claude Shannon in his celebrated 1948’s pa-
                                        per [119]) which considers primarily the communication of messages over noisy transmission channels, we
                                        may define Information Sciences (IS) as the fields that study “communication” between (noisy/imperfect)
                                        data and families of models (postulated as a priori knowledge). In short, information sciences seek meth-
                                        ods to distill information from data to models. Thus information sciences encompass information theory
                                        but also include the fields of Probability & Statistics, Machine Learning (ML), Artificial Intelligence (AI),
                                        Mathematical Programming, just to name a few.
                                            We review some key milestones of information geometry and report some definitions of the field by its pi-
                                        oneers in §5.2. Professor Shun-ichi Amari, the founder of modern information geometry, defined information
                                        geometry in the preface of his latest textbook [8] as follows: “Information geometry is a method of exploring
                                        the world of information by means of modern geometry.” In short, information geometry geometrically
                                        investigates information sciences. It is a mathematical endeavour to define and bound the term geometry
                                        itself as geometry is open-ended. Often, we start by studying the invariance of a problem (eg., invariance
                                        of distance between probability distributions) and get as a result a novel geometric structure (eg., a “sta-
                                        tistical manifold”). However, a geometric structure is “pure” and thus may be applied to other application
                                        areas beyond the scope of the original problem (eg, use of the dualistic structure of statistical manifolds in
                                        mathematical programming [57]): the method of geometry [9] thus yields a pattern of abduction [103, 115].
                                            A narrower definition of information geometry can be stated as the field that studies the geometry of
                                        decision making. This definition also includes model fitting (inference) which can be interpreted as a decision
                                        problem as illustrated in Figure 1: Namely, deciding which model parameter to choose from a family of
                                        parametric models. This framework was advocated by Abraham Wald [131, 132, 36] who considered all
                                        statistical problems as statistical decision problems. Dissimilarities (also loosely called distances among


                                                                                                  1
                                                mθ̂n (D)
                                                               mθ2
                                                mθ1
                                           M


Figure 1: The parameter inference θ̂ of a model from data D can also be interpreted as a decision making
problem: Decide which parameter of a parametric family of models M = {mθ }θ∈Θ suits the “best” the data.
Information geometry provides a differential-geometric structure on manifold M which useful for designing
and studying statistical decision rules.


others) play a crucial role not only for measuring the goodness-of-fit of data to model (say, likelihood in
statistics, classifier loss functions in ML, objective functions in mathematical programming or operations
research, etc.) but also for measuring the discrepancy (or deviance) between models.
    One may ponder why adopting a geometric approach? Geometry allows one to study invariance of
“figures” in a coordinate-free framework. The geometric language (e.g., line, ball or projection) also provides
affordances that help us reason intuitively about problems. Note that although figures can be visualized
(i.e., plotted in coordinate charts), they should be thought of as purely abstract objects, namely, geometric
figures.
    Geometry also allows one to study equivariance: For example, the centroid c(T ) of a triangle is equivariant
under any affine transformation A: c(A.T ) = A.c(T ). In Statistics, the Maximum Likelihood Estimator
(MLE) is equivariant under a monotonic transformation g of the model parameter θ: b(g(θ)) = g(θ̂), where
the MLE of θ is denoted by θ̂.

1.2    Outline of the survey
This survey is organized as follows:
    In the first part (§2), we start by concisely introducing the necessary background on differential geometry
in order to define a manifold structure (M, g, ∇), ie., a manifold M equipped with a metric tensor field g
and an affine connection ∇. We explain how this framework generalizes the Riemannian manifolds (M, g)
by stating the fundamental theorem of Riemannian geometry that defines a unique torsion-free metric-
compatible Levi-Civita connection which can be derived from the metric tensor.
    In the second part (§3), we explain the dualistic structures of information manifolds: We present the
conjugate connection manifolds (M, g, ∇, ∇∗ ), the statistical manifolds (M, g, C) where C denotes a cubic
tensor, and show how to derive a family of information manifolds (M, g, ∇−α , ∇α ) for α ∈ R provided any
given pair (∇ = ∇−1 , ∇∗ = ∇1 ) of conjugate connections. We explain how to get conjugate connections ∇
and ∇∗ coupled to the metric g from any smooth (potentially asymmetric) distances (called divergences),
present the dually flat manifolds obtained when considering Bregman divergences, and define, when dealing
with parametric family of probability models, the exponential connection e ∇ and the mixture connection
m
  ∇ that are dual connections coupled to the Fisher information metric. We discuss the concept of statistical
invariance for the metric tensor and the notion of information monotonicity for statistical divergences [30, 8].
It follows that the Fisher information metric is the unique invariant metric (up to a scaling factor), and that
the f -divergences are the unique separable invariant divergences.
    In the third part (§4), we illustrate how to use these information-geometric structures in simple ap-
plications: First, we described the natural gradient descent method in §4.1 and its relationships with the


                                                       2
Riemannian gradient descent and the Bregman mirror descent. Second, we consider two applications in du-
ally flat spaces in §4.2: In the first application, we consider the problem of Bayesian hypothesis testing and
show how Chernoff information (which defines the best error exponent) can be geometrically characterized
on the dually flat structure of an exponential family manifold. In the second application, we show how
to cluster statistical mixtures sharing the same component distributions on the dually flat mixture family
manifold.
    Finally, we conclude in §5 by summarizing the important concepts and structures of information geometry,
and by providing further references and textbooks [25, 8] for further readings to more advanced structures
and applications of information geometry. We also mention recent studies of generic classes of principled
distances and divergences.
    In the Appendix §A, we show how to estimate the statistical f -divergences between two probability distri-
butions in order to ensure that the estimates are non-negative in §B, and report the canonical decomposition
of the multivariate Gaussian family, an example of exponential family which admits a dually flat structure.
    At the beginning of each part, we start by outlining its contents. A summary of the notations used
throughout this survey is provided page 47.


2     Prerequisite: Basics of differential geometry
In §2.1, we review the very basics of Differential Geometry (DG) for defining a manifold (M, g, ∇) equipped
with both a metric tensor field g and an affine connection ∇. We explain these two independent met-
ric/connection structures in §2.2 and in §2.3, respectively. From an affine connection ∇, we show how to
derive the notion of covariant derivative in §2.3.1, parallel transport in §2.3.2 and geodesics in §2.3.3. We
further explain the intrinsic curvature and torsion of manifolds induced by the connection in §2.3.4, and state
the fundamental theorem of Riemannian geometry in §2.4: The existence of a unique torsion-free Levi-Civita
connection LC ∇ compatible with the metric (metric connection) that can be derived from the metric tensor
g. Thus the Riemannian geometry (M, g) is obtained as a special case of the more general manifold structure
(M, g, LC ∇): (M, g) ≡ (M, g, LC ∇). Information geometry shall further consider a dual structure (M, g, ∇∗ )
associated to (M, g, ∇), and the pair of dual structures shall form an information manifold (M, g, ∇, ∇∗ ).

2.1     Overview of differential geometry: Manifold (M, g, ∇)
Informally speaking, a smooth D-dimensional manifold M is a topological space that locally behaves like the
D-dimensional Euclidean space RD . Geometric objects (e.g., points, balls, and vector fields) and entities (e.g.,
functions and differential operators) live on M , and are coordinate-free but can conveniently be expressed
in any local coordinate system of an atlas A = {(Ui , xi )}i of charts(Ui , xi )’s (fully covering the manifold)
for calculations. Historically, René Descartes (1596-1650) allegedly invented the global Cartesian coordinate
system while wondering how to locate a fly on the ceiling from his bed. In practice, we shall use the most
expedient coordinate system to facilitate calculations. In information geometry, we usually handle a single
chart fully covering the manifold.
    A C k manifold is obtained when the change of chart transformations are C k . The manifold is said smooth
when it is C ∞ . At each point p ∈ M , a tangent plane Tp locally best linearizes the manifold. On any smooth
manifold M , we can define two independent structures:
    1. a metric tensor g, and
    2. an affine connection ∇.
   The metric tensor g induces on each tangent plane Tp an inner product space that allows one to measure
vector magnitudes (vector “lengths”) and angles/orthogonality between vectors. The affine connection ∇ is
a differential operator that allows one to define:

    1. the covariant derivative operator which provides a way to calculate differentials of a vector field Y with
       respect to another vector field X: Namely, the covariant derivative ∇X Y ,


                                                        3
                              Q∇
  2. the parallel transport    c   which defines a way to transport vectors between tangent planes along any
     smooth curve c,
  3. the notion of ∇-geodesics γ∇ which are defined as autoparallel curves, thus extending the ordinary
     notion of Euclidean straightness,
  4. the intrinsic curvature and torsion of the manifold.

2.2    Metric tensor fields g
The tangent bundle of M is defined as the “union” of all tangent spaces:
                                    T M := ∪p Tp = {(p, v),          p ∈ M, v ∈ Tp }.                         (1)
Thus the tangent bundle T M of a D-dimensional manifold M is of dimension 2D. (The tangent bundle is
a particular example of a fiber bundle with base manifold M .)
     Informally speaking, a tangent vector v plays the role of a directional derivative, with vf informally
meaning the derivative of a smooth function f (belonging to the space of smooth functions F(M )) along
the direction v. Since the manifolds are abstract and not embedded in some Euclidean space, we do not
view a vector as an “arrow” anchored on the manifold. Rather, vectors can be understood in several ways
in differential geometry like directional derivatives or equivalent class of smooth curves at a point. That is,
tangent spaces shall be considered as the manifold abstract too.
     A smooth vector field X is defined as a “cross-section” of the tangent bundle: X ∈ X(M ) = Γ(T M ),
where X(M ) or Γ(T M ) denote the space of smooth vector fields. A basis B = {b1 , . . . , bD } of a finite D-
dimensional vector space is a maximal linearly independent set of vectors: A set of vectors B = {b1 , . . . , bD }
                                            PD
is linearly independent if and only if        i=1 λi bi = 0 iff λi = 0 for all i ∈ [D]. That is, in a linearly
independent vector set, no vector of the set can be represented as a linear combination of the remaining
vectors. A vector set is linearly independent maximal when we cannot add another linearly independent
vector. Tangent spaces carry algebraic structures of vector spaces. Furthermore, to any vector space V ,
we can associate a dual covector space V ∗ which is the vector space of real-valued linear mappings. We
do not enter into details here to preserve this gentle introduction to information geometry with as little
intricacy as possible. Using local coordinates on a chart (U, x), the vector field X can be expressed as
       PD         Σ                                                                                    Σ
X = i=1 X i ei = X i ei using Einstein summation convention on dummy indices (using notation =), where
(X)B := (X i ) denotes the contravariant vector components (manipulated as “column vectors” in algebra)
                                                                   ∂
in the natural basis B = {e1 = ∂1 , . . . , eD = ∂D } with ∂i :=: ∂x i
                                                                       . A tangent plane (vector space) equipped
with an inner product h·, ·i yields an inner product space. We define a reciprocal basis B ∗ = {e∗ i = ∂ i }i of
B = {ei = ∂i }i so that vectors can also be expressed using the covariant vector components in the natural
reciprocal basis. The primal and reciprocal basis are mutually orthogonal by construction as illustrated in
Figure 2.
     For any vector v, its contravariant components v i ’s (superscript notation) and its covariant components
vi ’s (subscript notation) can be retrieved from v using the inner product with the use of the reciprocal and
primal basis, respectively:
                                                   vi    = hv, e∗ i i,                                        (2)
                                                   vi    = hv, ei i.                                          (3)
   The inner product defines a metric tensor g and a dual metric tensor g ∗ :
                                                  gij    := hei , ej i,                                       (4)
                                                  ∗ ij          ∗i    ∗j
                                              g          := he , e i.                                         (5)
   Technically speaking, the metric tensor gp : Tp M × Tp M → R is a 2-covariant tensor field:
                                                     Σ
                                                  g = gij dxi ⊗ dxj ,                                         (6)


                                                            4
                                                           x2

                                                           hei , ej i = δij
                                               e2
                                      2
                                  e
                                                                e1                    x1
                                          e1

Figure 2: Primal basis (red) and reciprocal basis (blue) of an inner product h·, ·i space. The primal/reciprocal
basis are mutually orthogonal: e1 is orthogonal to e2 , and e1 is orthogonal to e2 .

where ⊗ is the dyadic tensor product performed on pairwise covector basis {dxi }i (the covectors correspond-
ing to the reciprocal vector basis). We do not describe tensors in details for sake of brevity. A tensor is
a geometric entity of a tensor space that can also be interpreted as a multilinear map. A contravariant
vector lives in a vector space while a covariant vector lives in the dual covector space. We recommend the
textbook [66] for a concise and well-explained description of tensors.
   Let G = [gij ] and G∗ = [g ∗ ij ] denote the D × D matrices It follows by construction of the reciprocal basis
that G∗ = G−1 . The reciprocal basis vectors e∗ i ’s and primal basis vectors ei ’s can be expressed using the
dual metric g ∗ and metric g on the primal basis vectors ej ’s and reciprocal basis vectors e∗ j ’s, respectively:
                                                            Σ
                                                    e∗ i   =        g ∗ ij ej ,                               (7)
                                                            Σ             ∗j
                                                     ei    =        gij e .                                   (8)
    The metric tensor field g (“metric tensor” or “metric” for short) defines a smooth symmetric positive-
definite bilinear form on the tangent bundle so that for u, v ∈ Tp , g(u, v) ≥ 0 ∈ R. We can also write
equivalently gp (u, v):=:hu, vip :=:hu, vig(p) :=:hu, vi. Two vectors u and v are said orthogonal, denoted by
                                                                                            p
u ⊥ v, iff hu, vi = 0. The length of a vector is induced from the norm kukp :=:kukg(p) = hu, uig(p) . Using
local coordinates of a chart (U, x), we get the vector contravariant/covariant components, and compute the
metric tensor using matrix algebra (with column vectors by convention) as follows:
                                                                     −1
                           g(u, v) = (u)>                     >
                                        B × Gx(p) × (v)B = (u)B ∗ × Gx(p) × (v)B ∗ ,                          (9)

since it follows from the primal/reciprocal basis that G × G∗ = I, the identity matrix. Thus on any tangent
plane Tp , we get a Mahalanobis distance:
                                                    v
                                                    uD D
                                                    uX X
                           MG (u, v) := ku − vkG = t         Gij (ui − v i )(uj − v j ).               (10)
                                                                i=1 j=1

   The inner product of two vectors u and v is a scalar (a 0-tensor) that can be equivalently calculated as:
                                                                    Σ             Σ
                                           hu, vi := g(u, v) = ui vi = ui v i .                              (11)
   A metric tensor g of manifold M is said conformal when h·, ·ip = κ(p)h·, ·iEuclidean . That is, when the
inner product is a scalar function κ(·) of the Euclidean dot product. More precisely, we define the notion of
a metric g 0 conformal to another metric g when these metrics define the same angles between vectors u and
v of a tangent plane Tp :
                                         gp0 (u, v)             gp (u, v)
                                 q             q         =p          p         .                         (12)
                                   gp0 (u, u) gp0 (v, v)   gp (u, u) gp (v, v)


                                                                5
                                                        vc(t)         vq
                                           vp             q
                                                  c(t)
                                                          Q∇
                                                p   v q =   c vp


                                          M

Figure 3: Illustration of the parallel transport of vectors on tangent planes along a smooth curve. For a
smooth curve c, with c(0) = p and c(1) = q, a vector vp ∈ Tp is parallel transported smoothly to a vector
vq ∈ Tq such that for any t ∈ [0, 1], we have vc(t) ∈ Tc(t) .


Usually g 0 is chosen as the Euclidean metric. In conformal geometry, we can measure angles between vectors
in tangent planes as if we were in an Euclidean space, without any deformation. This is handy for checking
orthogonality in charts. For example, Poincaré disk model of hyperbolic geometry is conformal but Klein
disk model is not conformal (except at the origin), see [89].

2.3     Affine connections ∇
An affine connection ∇ is a differential operator defined on a manifold that allows us to define (1) a covariant
derivative of vector fields, (2) a parallel transport of vectors on tangent planes along a smooth curve, and
(3) geodesics. Furthermore, an affine connection fully characterizes the curvature and torsion of a manifold.

2.3.1   Covariant derivatives ∇X Y of vector fields
A connection defines a covariant derivative operator that tells us how to differentiate a vector field Y
according to another vector field X. The covariant derivative operator is denoted using the traditional
gradient symbol ∇. Thus a covariate derivative ∇ is a function:

                                         ∇ : X(M ) × X(M ) → X(M ),                                          (13)

that has its own special subscript notation ∇X Y :=:∇(X, Y ) for indicating that it is differentiating a vector
field Y according to another vector field X.
    By prescribing D3 smooth functions Γkij = Γkij (p), called the Christoffel symbols of the second kind, we
define the unique affine connection ∇ that satisfies in local coordinates of chart (U, x) the following equations:

                                                 ∇∂i ∂j = Γkij ∂k .                                          (14)
    The Christoffel symbols can also be written as Γkij := (∇∂i ∂j )k , where (·)k denote the k-th coordinate.
                             k
The k-th component (∇X Y ) of the covariant derivative of vector field Y with respect to vector field X is
given by:
                                                              ∂Y k
                                                                               
                                      k Σ             Σ
                               (∇X Y ) = X i (∇i Y )k = X i         +  Γ k
                                                                         ij Y j
                                                                                  .                       (15)
                                                              ∂xi
   The Christoffel symbols are not tensors (fields) because the transformation rules induced by a change of
basis do not obey the tensor contravariant/covariant rules.




                                                         6
                                 Q∇
2.3.2    Parallel transport        c    along a smooth curve c
Since the manifold is not embedded1 in a Euclidean space, we cannot add a vector v ∈ Tp to a vector v 0 ∈ Tp0
as the tangent vector spaces are unrelated to each others without a connection.2 Thus a connection ∇ defines
how to associate vectors between infinitesimally close tangent planes Tp and Tp+dp . Then the connection
allows us to smoothly transport a vector v ∈ Tp by sliding it (with infinitesimal moves) along a smooth curve
c(t) (with c(0) = p and c(1) = q), so that the vector vp ∈ Tp “corresponds” to a vector vq ∈ Tq : This is called
the parallel transport. This mathematical prescription is necessary in order to study dynamics on manifolds
(e.g., study the motion of a particle3 on the manifold). We can express the parallel transport along the
smooth curve c as:
                                                                Y∇
                                ∀v ∈ Tp , ∀t ∈ [0, 1], vc(t) =        v ∈ Tc(t)                             (16)
                                                                             c(0)→c(t)

The parallel transport is schematically illustrated in Figure 3.

2.3.3    ∇-geodesics γ∇ : Autoparallel curves
A connection ∇ allows one to define ∇-geodesics as autoparallel curves, that are curves γ such that we have:

                                                           ∇γ̇ γ̇ = 0.                                               (17)

    That is, the velocity vector γ̇ is moving along the curve parallel to itself (and all tangent vectors on the
geodesics are mutually parallel): In other words, ∇-geodesics generalize the notion of “straight Euclidean”
lines. In local coordinates (U, x), γ(t) = (γ k (t))k , the autoparallelism amounts to solve the following second-
order Ordinary Differential Equations (ODEs):

                                       γ̈(t) + Γkij γ̇(t)γ̇(t) = 0,        γ l (t) = xl ◦ γ(t),                      (18)
where Γkij are the Christoffel symbols of the second kind, with:

                                                  Σ                          Σ
                                             Γkij = Γij,l g lk ,       Γij,k = glk Γlij ,                            (19)

where Γij,l the Christoffel symbols of the first kind. Geodesics are 1D autoparallel submanifolds and ∇-
hyperplanes are defined similarly as autoparallel submanifolds of dimension D − 1. We may specify in
subscript the connection that yields the geodesic γ: γ∇ .
   The geodesic equation ∇γ̇(t) γ̇(t) = 0 may be either solved as an Initial Value Problem (IVP) or as a
Boundary Value Problem (BVP):
   • Initial Value Problem (IVP): fix the conditions γ(0) = p and γ̇(0) = v for some vector v ∈ Tp .
   • Boundary Value Problem (BVP): fix the geodesic extremities γ(0) = p and γ(1) = q.

2.3.4    Curvature and torsion of a manifold
An affine connection ∇ defines a 4D4 curvature tensor R (expressed using components Rjkl
                                                                                     i
                                                                                         of a (1, 3)-tensor).
The coordinate-free equation of the curvature tensor is given by:

                                  R(X, Y )Z := ∇X ∇Y X − ∇Y ∇X Z − ∇[X,Y ] Z,                                        (20)
  1 Whitney embedding theorem states that any D-dimensional Riemannian manifold can be embedded into R2D .
  2 When embedded, we can implicitly use the ambient Euclidean connection Euc ∇, see [2].
  3 Elie Cartan introduced the notion of affine connections [27, 3] in the 1920’s motivated by the principle of inertia in

mechanics: A point particle, without any force acting on it, shall move along a straight line with constant velocity.
                                                                                                             2  2
  4 It follows from symmetry constraints that the number of independent components of the Riemann tensor is D (D −1)    in
                                                                                                               12
D dimensions.



                                                                   7
Figure 4: Parallel transport with respect to the metric connection: Curvature effect can be visualized as
the angle defect along the parallel transport on smooth (infinitesimal) loops. For a sphere manifold, a vector
parallel-transported along a loop does not coincide with itself, while it always conside with itself for a (flat)
manifold. Drawings are courtesy of c CNRS, http://images.math.cnrs.fr/Visualiser-la-courbure.
html

where [X, Y ](f ) = X(Y (f )) − Y (X(f )) (∀f ∈ F(M )) is the Lie bracket of vector fields. When the connection
is the metric Levi-Civita, the curvature is called Riemann-Christoffel curvature tensor. In a local coordinate
system, we have:
                                                            Σ  l
                                              R(∂j , ∂k )∂i = Rjki ∂l .                                    (21)
Informally speaking, the curvature tensor as defined in Eq. 20 quantifies the amount of non-commutativity
of the covariant derivative.
    A manifold M equipped with a connection ∇ is said flat (meaning ∇-flat) when R = 0. This holds in
particular when finding a particular5 coordinate system x of a chart (U, x) such that Γkij = 0, i.e., when all
connection coefficients vanish.
    A manifold is torsion-free when the connection is symmetric. A symmetric connection satisfies the
following coordinate-free equation:
                                          ∇X Y − ∇Y X = [X, Y ].                                         (22)
Using local chart coordinates, this amounts to check that Γkij = Γkji . The torsion tensor is a (1, 2)-tensor
defined by:
                                    T (X, Y ) := ∇X Y − ∇Y X − [X, Y ].                                  (23)
    For a torsion-free connection, we have the first Bianchi identity:

                                       R(X, Y )Z + R(Z, X)Y + R(Y, Z)X = 0,                                                (24)

and the second Bianchi identity:

                              (∇V R)(X, Y )Z + (∇X R)(Y, V )Z + (∇Y R)(V, X)Z = 0.                                         (25)

    In general, the parallel transport is path-dependent. The angle defect of a vector transported on an
infinitesimal closed loop (a smooth curve with coinciding extremities) is related to the curvature. However
for a flat connection, the parallel transport does not depend on the path, and yields absolute parallelism
geometry [133]. Figure 4 illustrates the parallel transport along a curve for a curved manifold (the sphere
manifold) and a flat manifold ( the cylinder manifold6 ).
    An affine connection is a torsion-free linear connection. Figure 5 summarizes the various concepts of
differential geometry induced by an affine connection ∇ and a metric tensor g.
   5 For example, the Christoffel symbols vanish in a rectangular coordinate system of a plane but not in the polar coordinate

system of it.
   6 The Gaussian curvature at of point of a manifold is the product of the minimal and maximal sectional curvatures: κ :=
                                                                                                                            G
κmin κmax . For a cylinder, since κmin = 0, it follows that the Gaussian curvature of a cylinder is 0. Gauss’s Theorema Egregium
(meaning “remarkable theorem”) proved that the Gaussian curvature is intrinsic and does not depend on how the surface is
embedded into the ambient Euclidean space.


                                                               8
                            Affine connection
                                    ∇


   Curvature          ParallelQtransport        Geodesic           Volume form                Covariant
    ∇ i                   ∇                         ∇               ∇
      R jkl                     c(t) v                 γ              ω = f dv                derivative
                                               (∇γ̇ γ̇ = 0)        (∇ ∇ ω = 0)                  ∇X Y
                                   Levi-civita connection
                                             g
                                               ∇
   Ricci curvature                              Metric tensor
   ∇
     Ric(Y, Z) = tr(X 7→ ∇ R(X, Y, Z))                g


         Scalar curvature
          Scal = g ij Rij

                                                                    Manifold (M, g, ∇)

          Divergence                        Gradient                        Hessian                   Laplacian
  div(X) = tr(Y 7→ g(∇Y X, Y ))       g(gradf, X) = df (X)          Hessf (x)[v] = ∇v gradf        ∆f = div(grad(f ))


  Figure 5: Differential-geometric concepts associated to an affine connection ∇ and a metric tensor g.


    Curvature is a fundamental concept inherent to geometry [22]: There are several notions of curvatures:
scalar curvature, sectional curvature, Gaussian curvature of surfaces to Riemannian-Christoffel 4-tensor,
Ricci symmetric 2-tensor, synthetic Ricci curvature in Alexandrov geometry, etc.

2.4    The fundamental theorem of Riemannian geometry: The Levi-Civita metric
       connection
By definition, an affine connection ∇ is said metric compatible with g when it satisfies for any triple (X, Y, Z)
of vector fields the following equation:

                                      XhY, Zi = h∇X Y, Zi + hY, ∇X Zi,                                            (26)

which can be written equivalently as:

                                     Xg(Y, Z) = g(∇X Y , Z) + g(Y, ∇X Z)                                          (27)

Using local coordinates and natural basis {∂i } for vector fields, the metric-compatibility property amounts
to check that we have:
                                      ∂k gij = h∇∂k ∂i , ∂j i + h∂i , ∇∂k ∂j i                            (28)
                                                                                      Q∇
    A property of using a metric-compatible connection is that the parallel transport      of vectors preserve
the metric:                                  * ∇                ∇
                                                                         +
                                                 Y             Y
                                hu, vic(0) =             u,            v       ∀t.                        (29)
                                               c(0)→c(t)      c(0)→c(t)   c(t)

That is, the parallel transport preserves angles (and orthogonality) and lengths of vectors in tangent planes
when transported along a smooth curve.
   The fundamental theorem of Riemannian geometry states the existence of a unique torsion-free metric
compatible connection:


                                                         9
Theorem 1 (Levi-Civita metric connection). There exists a unique torsion-free affine connection compatible
with the metric called the Levi-Civita connection LC ∇.
    The Christoffel symbols of the Levi-Civita connection can be expressed from the metric tensor g as
follows:
                                   LC k Σ 1 kl
                                      Γij = g (∂i gil + ∂j gil − ∂l gij ) ,                       (30)
                                             2
where g ij denote the matrix elements of the inverse matrix g −1 .
    The Levi-Civita connection can also be defined coordinate-free with the Koszul formula:
    2g(∇X Y, Z) = X(g(Y, Z)) + Y (g(X, Z)) − Z(g(X, Y )) + g([X, Y ], Z) − g([X, Z], Y ) − g([Y, Z], X).    (31)
   There exists metric-compatible connections with torsions studied in theoretical physics. See for example
the flat Weitzenböck connection [15].
   The metric tensor g induces the torsion-free metric-compatible Levi-Civita connection that determines
the local structure of the manifold. However, the metric g does not fix the global topological structure: For
example, although a cone and a cylinder have locally the same flat Euclidean metric, they exhibit different
global structures.

2.5     Preview: Information geometry versus Riemannian geometry
In information geometry, we consider a pair of conjugate affine connections ∇ and ∇∗ (often but not neces-
sarily torsion-free) that are coupled to the metric g: The structure is conventionally written as (M, g, ∇, ∇∗ ).
The key property is that those conjugate connections are metric compatible, and therefore the induced dual
parallel transport preserves the metric:
                                                                ∇∗
                                                 * ∇                    +
                                                     Y         Y
                                    hu, vic(0) =          u,          v     .                               (32)
                                                  c(0)→c(t)     c(0)→c(t)   c(t)

Thus the Riemannian manifold (M, g) can be interpreted as the self-dual information-geometric manifold ob-
                                                                                                              ∗
tained for ∇ = ∇∗ = LC ∇ the unique torsion-free Levi-Civita metric connection: (M, g) ≡ (M, g, LC ∇, LC ∇ =
LC
   ∇). However, let us point out that for a pair of self-dual Levi-Civita conjugate connections, the information-
geometric manifold does not induce a distance. This contrasts with the Riemannian modeling (M, g) which
provides a Riemmanian metric distance Dρ (p, q) defined by the length of the geodesic γ connecting the two
points p = γ(0) and q = γ(1):
                                          Z 1                    Z 1q
                         Dρ (p, q) :=         kγ 0 (t)kγ(t) dt =      gγ(t) (γ̇(t), γ̇(t))dt,               (33)
                                           0                      0
                                         Z 1q
                                     =          γ̇(t)> gγ(t) γ̇(t)dt.                                       (34)
                                           0

This geodesic length distance Dρ (p, q) can also be interpreted as the shortest path linking point p to point
                    R1
q: Dρ (p, q) = inf γ 0 kγ 0 (t)kγ(t) dt (with p = γ(0) and q = γ(1)).
   Usually, this Riemannian geodesic distance is not available in closed-form (and need to be approximated
or bounded) because the geodesics cannot be explicitly parameterized (see geodesic shooting methods [11]).
   We are now ready to introduce the key geometric structures of information geometry.


3      Information manifolds
3.1     Overview
In this part, we explain the dualistic structures of manifolds in information geometry. In §3.2, we first present
the core Conjugate Connection Manifolds (CCMs) (M, g, ∇, ∇∗ ), and show how to build Statistical Manifolds


                                                        10
(SMs) (M, g, C) from a CCM in §3.3. From any statistical manifold, we can build a 1-parameter family
(M, g, ∇−α , ∇α ) of CCMs, the information α-manifolds. We state the fundamental theorem of information
geometry in §3.5. These CCMs and SMs structures are not related to any distance a priori but require at first
a pair (∇, ∇∗ ) of conjugate connections coupled to a metric tensor g. We show two methods to build an initial
pair of conjugate connections. A first method consists in building a pair of conjugate connections ( D ∇, D ∇∗ )
from any divergence D in §3.6. Thus we obtain self-conjugate connections when the divergence is symmetric:
D(θ1 : θ2 ) = D(θ2 : θ1 ). When the divergences are Bregman divergences (i.e., D = BF for a strictly convex
and differentiable Bregman generator), we obtain Dually Flat Manifolds (DFMs) (M, ∇2 F, F ∇, F ∇∗ ) in §3.7.
DFMs nicely generalize the Euclidean geometry and exhibit Pythagorean theorems. We further characterize
when orthogonal F ∇-projections and dual F ∇∗ -projections of a point on submanifold a is unique.7 A second
method to get a pair of conjugate connections ( e ∇, m ∇) consists in defining these connections from a regular
parametric family of probability distributions P = {pθ (x)}θ . In that case, these ‘e’xponential connection
e
  ∇ and ‘m’ixture connection m ∇ are coupled to the Fisher information metric P g. A statistical manifold
(P, P g, P C) can be recovered by considering the skewness Amari-Chentsov cubic tensor P C, and it follows a
1-parameter family of CCMs, (P, P g, P ∇−α , P ∇+α ), the statistical expected α-manifolds. In this parametric
statistical context, these information manifolds are called expected information manifolds because the various
quantities are expressed from statistical expectations E· [·]. Notice that these information manifolds can be
used in information sciences in general, beyond the traditional fields of statistics. In statistics, we motivate
the choice of the connections, metric tensors and divergences by studying statistical invariance criteria, in
§3.10. We explain how to recover the expected α-connections from standard f -divergences that are the only
separable divergences that satisfy the property of information monotonicity. Finally, in §3.11, the recall the
Fisher-Rao expected Riemannian manifolds that are Riemannian manifolds (P, P g) equipped with a geodesic
metric distance called the Fisher-Rao distance, or Rao distance for short.

3.2     Conjugate connection manifolds: (M, g, ∇, ∇∗ )
We begin with a definition:
Definition 1 (Conjugate connections). A connection ∇∗ is said to be conjugate to a connection ∇ with
respect to the metric tensor g if and only if we have for any triple (X, Y, Z) of smooth vector fields the
following identity satisfied:
                              XhY, Zi = h∇X Y, Zi + hY, ∇∗X Zi,          ∀X, Y, Z ∈ X(M ).                             (35)
   We can notationally rewrite Eq. 35 as:
                                        Xg(Y, Z) = g(∇X Y , Z) + g(Y, ∇∗X Z),                                          (36)
and further explicit that for each point p ∈ M , we have:
                                Xp gp (Yp , Zp ) = gp ((∇X Y )p , Zp ) + gp (Yp , (∇∗X Z)p ).                          (37)
We check that the right-hand-side is a scalar and that the left-hand-side is a directional derivative of a
real-valued function, that is also a scalar.
    Conjugation is an involution: (∇∗ )∗ = ∇.
Definition 2 (Conjugate Connection Manifold). The structure of the Conjugate Connection Manifold
(CCM) is denoted by (M, g, ∇, ∇∗ ), where (∇, ∇∗ ) are conjugate connections with respect to the metric
g.
   A remarkable property is that the dual parallel transport of vectors preserves the metric. That is, for any
smooth curve c(t), the inner product is conserved when we transport one of the vector u using the primal
                  Q∇                                                             Q∇∗
parallel transport c and the other vector v using the dual parallel transport c .
  7 In Euclidean geometry, the orthogonal projection of a point p onto an affine subspace S is proved to be unique using the

Pythagorean theorem.


                                                             11
                                                                           ∗
                                                          ∇              ∇
                                                     *                                 +
                                                          Y              Y
                                      hu, vic(0) =                u,               v          .           (38)
                                                      c(0)→c(t)        c(0)→c(t)       c(t)

Property 1 (Dual parallel transport preserves the metric). A pair (∇, ∇∗ ) of conjugate connections pre-
serves the metric g if and only if:
                                                   ∇∗
                                          * ∇              +
                                            Y      Y
                             ∀t ∈ [0, 1],      u,        v     = hu, vic(0) .                       (39)
                                              c(0)→c(t)    c(0)→c(t)        c(t)

Property 2. Given a connection ∇ on (M, g) (i.e., a structure (M, g, ∇)), there exists a unique conjugate
connection ∇∗ (i.e., a dual structure (M, g, ∇∗ )).
   We consider a manifold M equipped with a pair of conjugate connections ∇ and ∇∗ that are coupled with
the metric tensor g so that the dual parallel transport preserves the metric. We define the mean connection
¯
∇:                                                          ∗
                                                ∇¯ = ∇+∇ ,                                             (40)
                                                         2
with corresponding Christoffel coefficients denoted by Γ̄. This mean connection coincides with the Levi-Civita
metric connection:
                                                  ∇¯ = LC ∇.                                              (41)
                                ¯ is self-conjugate, and coincide with the Levi-Civita metric connection.
Property 3. The mean connection ∇

3.3    Statistical manifolds: (M, g, C)
Lauritzen introduced this corner structure [62] of information geometry in 1987. Beware that although it
bears the name “statistical manifold,” it is a purely geometric construction that may be used outside of the
field of Statistics. However, as we shall mention later, we can always find a statistical model P corresponding
to a statistical manifold [128]. We shall see how we can convert a conjugate connection manifold into such
a statistical manifold, and how we can subsequently derive an infinite family of CCMs from a statistical
manifold. In other words, once we have a pair of conjugate connections, we will be able to build a family of
pairs of conjugate connections.
    We define a totally symmetric8 cubic (0, 3)-tensor (i.e., 3-covariant tensor) called the Amari-Chentsov
tensor:

                                                  Cijk := Γkij − Γ∗ kij ,                                 (42)
or in coordinate-free equation:
                                          C(X, Y, Z) := h∇X Y − ∇∗X Y, Zi.                                (43)
   Using the local basis, this cubic tensor can be expressed as:

                                    Cijk = C(∂i , ∂j , ∂k ) = h∇∂i ∂j − ∇∗∂i ∂j , ∂k i                    (44)

Definition 3 (Statistical manifold [62]). A statistical manifold (M, g, C) is a manifold M equipped with a
metric tensor g and a totally symmetric cubic tensor C.
  8 This means that C
                     ijk = Cσ(i)σ(j)σ(k) for any permutation σ. The metric tensor is totally symmetric.




                                                            12
3.4    A family {(M, g, ∇−α , ∇α = (∇−α )∗ )}α∈R of conjugate connection manifolds
For any pair (∇, ∇∗ ) of conjugate connections, we can define a 1-parameter family of connections {∇α }α∈R ,
                                                                                           ¯ = LC ∇, ∇1 = ∇
called the α-connections such that (∇−α , ∇α ) are dually coupled to the metric, with ∇0 = ∇
       −1     ∗
and ∇ = ∇ . By observing that the scaled cubic tensor αC is also a totally symmetric cubic 3-covariant
tensor, we can derive the α-connections from a statistical manifold (M, g, C) as:

                                                               α
                                          Γα
                                           ij,k    =   Γ0ij,k −  Cij,k ,                                   (45)
                                                               2
                                                               α
                                          Γ−α
                                           ij,k    =   Γ0ij,k + Cij,k ,                                    (46)
                                                               2
                                                                   Σ
where Γ0ij,k are the Levi-Civita Christoffel symbols, and Γki,j = Γlij glk (by index juggling).
   The α-connection ∇α can also be defined as follows:
                                                       α
                        g(∇αX Y, Z) = g(
                                         LC
                                            ∇X Y, Z) + C(X, Y, Z), ∀X, Y, Z ∈ X(M ).                       (47)
                                                       2
Theorem 2 (Family of information α-manifolds). For any α ∈ R, (M, g, ∇−α , ∇α = (∇−α )∗ ) is a conjugate
connection manifold.
   The α-connections ∇α can also be constructed directly from a pair (∇, ∇∗ ) of conjugate connections by
taking the following weighted combination:
                                                  1+α         1−α ∗
                                       Γα
                                        ij,k =        Γij,k +    Γij,k .                                   (48)
                                                   2           2

3.5    The fundamental theorem of information geometry: ∇ κ-curved ⇔ ∇∗ κ-
       curved
We now state the fundamental theorem of information geometry and its corollaries:
Theorem 3 (Dually constant curvature manifolds). If a torsion-free affine connection ∇ has constant
curvature κ then its conjugate torsion-free connection ∇∗ has necessarily the same constant curvature κ.
   The proof is reported in [25] (Proposition 8.1.4, page 226).
   A statistical manifold (M, g, C) is said α-flat if its induced α-connection is flat. It can be shown that
Rα = −R−α .
   We get the following two corollaries:
Corollary 1 (Dually α-flat manifolds). A manifold (M, g, ∇−α , ∇α ) is ∇α -flat if and only if it is ∇−α -flat.
Corollary 2 (Dually flat manifolds (α = ±1)). A manifold (M, g, ∇, ∇∗ ) is ∇-flat if and only if it is ∇∗ -flat.
   (See Theorem 3.3 of [9])
   Let us now define the notion of constant curvature of a statistical structure [46]:
Definition 4 (Constant curvature κ). A statistical structure (M, g, ∇) is said of constant curvature κ when
                       R∇ (X, Y )Z = κ{g(Y, Z)X − g(X, Z)Y },          ∀X, Y, Z ∈ Γ(T M ),
where Γ(T M ) denote the space of smooth vector fields.
    It can be proved that the Riemann-Christoffel (RC) 4-tensors of conjugate α-connections [25] are related
as follows:                                                          
                             g R(α) (X, Y )Z, W + g Z, R(−α) (X, Y )W = 0.                              (49)
                     ∗
Thus we have g R∇ (X, Y )Z, W = −g Z, R∇ (X, Y )W .
                                                       

    Thus once we are given a pair of conjugate connections, we can always build a 1-parametric family of
manifolds. Manifolds with constant curvature κ are interesting from the computational viewpoint as dual
geodesics have simple closed-form expressions.


                                                        13
                                                                                                                           ∗
3.6     Conjugate connections from divergences: (M, D) ≡ (M, D g, D ∇, D ∇∗ = D ∇)
Loosely speaking, a divergence D(· : ·) is a smooth distance [138], potentially asymmetric. In order to
                                                                                                                    ∂
define precisely a divergence, let us first introduce the following handy notations: ∂i,· f (x, y) = ∂x               i f (x, y),
                 ∂                               ∂2     ∂                                 ∂    ∂2
∂·,j f (x, y) = ∂yj f (x, y), ∂ij,k f (x, y) = ∂xi ∂xj ∂yk f (x, y) and ∂i,jk f (x, y) = ∂xi ∂yj ∂yk f (x, y), etc.
Definition 5 (Divergence). A divergence D : M × M → [0, ∞) on a manifold M with respect to a local
chart Θ ⊂ RD is a C 3 -function satisfying the following properties:
   1. D(θ : θ0 ) ≥ 0 for all θ, θ0 ∈ Θ with equality holding iff θ = θ0 (law of the indiscernibles),
   2. ∂i,· D(θ : θ0 )|θ=θ0 = ∂·,j D(θ : θ0 )|θ=θ0 = 0 for all i, j ∈ [D],
   3. −∂·,i ∂·,j D(θ : θ0 )|θ=θ0 is positive-definite.
    The dual divergence is defined by swapping the arguments:
                                                      D∗ (θ : θ0 ) := D(θ0 : θ),                                            (50)
and is also called the reverse divergence (reference duality in information geometry). Reference duality of
divergences is an involution: (D∗ )∗ = D.
    The Euclidean distance is a metric distance but not a divergence. The squared Euclidean distance is a
non-metric symmetric divergence. The metric tensor g yields Riemannian metric distance Dρ but it is never
a divergence.
    From any given divergence D, we can define a conjugate connection manifold following the construction
of Eguchi [42, 43] (1983):
                                                                            ∗
Theorem 4 (Manifold from divergence). (M, D g, D ∇, D ∇) is an information manifold with:
                                                                                       ∗
                                              D
                                                  g       := −∂i,j D(θ : θ0 )|θ=θ0 = D g,                                   (51)
                                         D                                      0
                                             Γijk         := −∂ij,k D(θ : θ )|θ=θ0 ,                                        (52)
                                        D∗                                      0
                                             Γijk         := −∂k,ij D(θ : θ )|θ=θ0 .                                        (53)
    The associated statistical manifold is (M, D g, D C) with:
                                                                       ∗
                                                      D
                                                          C ijk = D Γijk − D Γijk .                                         (54)
   Since αD C is a totally symmetric cubic tensor for any α ∈ R, we can derive a one-parameter family of
conjugate connection manifolds:
                                                       −α     −α
                         n            α                                 α
                                                                          o
                          (M, D g, D C ) ≡ (M, D g, D ∇ , (D ∇ )∗ = D ∇ )       .                   (55)
                                                                                             α∈R

  In the remainder, we use the shortcut (M, D) to denote the divergence-induced information manifold
                 ∗
(M, D g, D ∇, D ∇ ). Notice that it follows from construction that:
                                                              D    ∗        ∗
                                                                  ∇ = D ∇.                                                  (56)

3.7     Dually flat manifolds (Bregman geometry): (M, F ) ≡ (M, BF g, BF ∇, BF ∇∗ =
        BF ∗
          ∇)
We consider dually flat manifolds that satisfy asymmetric Pythagorean theorems. These flat manifolds can
be obtained from a canonical Bregman divergence.
   Consider a strictly convex smooth function F (θ) called a potential function, with θ ∈ Θ where Θ is an
open convex domain. Notice that the function convexity does not change by an affine transformation. We
associate to the potential function F a corresponding Bregman divergence (parameter divergence):
                                    BF (θ : θ0 ) := F (θ) − F (θ0 ) − (θ − θ0 )> ∇F (θ0 ).                                  (57)


                                                                       14
   We write also the Bregman divergence between point P and point Q as D(P : Q) := BF (θ(P ) : θ(Q)),
where θ(P ) denotes the coordinates of a point P .
   The information-geometric structure induced by a Bregman generator9 is (M, F g, F C) := (M, BF g, BF C)
with:

                                      F
                                          g       :=          BF
                                                                   g = − [∂i ∂j BF (θ : θ0 )|θ0 =θ ] = ∇2 F (θ),                 (58)
                                      F                       BF
                                          Γ       :=               Γij,k (θ) = 0,                                                (59)
                                 F                            BF
                                     Cijk         :=               Cijk = ∂i ∂j ∂k F (θ).                                        (60)

   Since all coefficients of the Christoffel symbols vanish (Eq. 59), the information manifold is F ∇-flat.
The Levi-Civita connection LC ∇ is obtained from the metric tensor F g (usually not flat), and we get the
conjugate connection ( F ∇)∗ = F ∇1 from (M, F g, F C).
   The Legendre-Fenchel transformation yields the convex conjugate F ∗ that is interpreted as the dual
potential function:
                                         F ∗ (η) := sup{θ> η − F (θ)}.                                (61)
                                                                          θ∈Θ

Theorem 5 (Fenchel-Moreau biconjugation [52]). If F is a lower semicontinuous10 and convex function,
then its Legendre-Fenchel transformation is involutive: (F ∗ )∗ = F (biconjugation).

    In a dually flat manifold, there exists two global dual affine coordinate systems η = ∇F (θ) and θ =
∇F ∗ (η), and therefore the manifold can be covered by a single chart. Thus if a probability family belongs
to an exponential family then its natural parameters cannot belong to, say, a spherical space (that requires
at least two charts).
    We have the Crouzeix [32] identity relating the Hessians of the potential functions:

                                                                   ∇2 F (θ)∇2 F ∗ (η) = I,                                       (62)

where I denote the D × D identity matrix. This Crouzeix identity reveals that B = {∂i }i and B ∗ = {∂ j }j
are the primal and reciprocal basis, respectively.
   The Bregman divergence can be reinterpreted using Young-Fenchel (in)equality as the canonical diver-
gence AF,F ∗ [12]:

                         BF (θ : θ0 ) = AF,F ∗ (θ : η 0 ) = F (θ) + F ∗ (η 0 ) − θ> η 0 = AF ∗ ,F (η 0 : θ).                     (63)

    The dual Bregman divergence BF ∗ (θ : θ0 ) := BF (θ0 : θ) = BF ∗ (η : η 0 ) yields

                                                                                                        ∂
                                                  F ij
                                                   g (η)            =    ∂ i ∂ j F ∗ (η),    ∂ l :=:                             (64)
                                                                                                       ∂η l
                                                       ijk
                                              F
                                                  Γ∗         (η)    =    0,     F
                                                                                    C ijk = ∂ i ∂ j ∂ k F ∗ (η)                  (65)

    Thus the information manifold is both F ∇-flat and F ∇∗ -flat: This structure is called a dually flat
manifold (DFM). In a DFM, we have two global affine coordinate systems θ(·) and η(·) related by the
Legendre-Fenchel transformation of a pair of potential functions F and F ∗ . That is, (M, F ) ≡ (M, F ∗ ), and
the dual atlases are A = {(M, θ)} and A∗ = {(M, η)}.
    In a dually flat manifold, any pair of points P and Q can either be linked using the ∇-geodesic (that is
θ-straight) or the ∇∗ -geodesic (that is η-straight). In general, there are 23 = 8 types of geodesic triangles in
a dually flat manifold.
   9 Here, we define a Bregman generator as a proper, lower semi-continuous, and strictly convex and C 3 differentiable real-

valued function.
  10 A function f is lower semicontinous (lsc) at x iff f (x ) ≤ lim
                                                   0        0        x→x0 inf f (x). A function f is lsc if it is lsc at x for all x in
the function domain.



                                                                              15
                     γ ∗ (P, Q) ⊥F γ(Q, R)
                                                                                                   γ(P, Q) ⊥F γ ∗ (Q, R)


                   P                                                                         P




                                                      R                                                                            R
             Q                                                                           Q
             D(P : R) = D(P : Q) + D(Q : R)                                          D∗ (P : R) = D∗ (P : Q) + D∗ (Q : R)
        BF (θ(P ) : θ(R)) = BF (θ(P ) : θ(Q)) + BF (θ(Q) : θ(R))                BF ∗ (η(P ) : η(R)) = BF ∗ (η(P ) : η(Q)) + BF ∗ (η(Q) : η(R))



                              Figure 6: Dual Pythagorean theorems in a dually flat space.


   on a Bregman manifold, the primal parallel transport of a vector does not change the contravariant vector
components, and the dual parallel transport does not change the covariant vector components. Because the
dual connections are flat, the dual parallel transports are path-independent.
   Moreover, the dual Pythagorean theorems [76] illustrated in Figure 6 holds. Let γ(P, Q) = γ∇ (P, Q)
denote the ∇-geodesic passing through points P and Q, and γ ∗ (P, Q) = γ∇∗ (P, Q) denote the ∇∗ -geodesic
passing through points P and Q. Two curves γ1 and γ2 are orthogonal at point p = γ1 (t1 ) = γ2 (t2 ) with
respect to the metric tensor g when g(γ̇1 (t1 ), γ̇2 (t2 )) = 0.
Theorem 6 (Dual Pythagorean identities).
                                                                                       Σ
     γ ∗ (P, Q) ⊥ γ(Q, R) ⇔ (η(P ) − η(Q))> (θ(Q) − θ(R)) = (ηi (P ) − ηi (Q))(θi (Q) − θi (R)) = 0,
                                                                                       Σ
     γ(P, Q) ⊥ γ ∗ (Q, R) ⇔ (θ(P ) − θ(Q))> (η(Q) − η(R)) = (θi (P ) − θi (Q))> (ηi (Q) − ηi (R)) = 0.

    We can define dual Bregman projections and characterize when these projections are unique: A subman-
ifold S ⊂ M is said ∇-flat (∇∗ -flat) iff. it corresponds to an affine subspace in the θ-coordinate system (in
the η-coordinate system, respectively).
Theorem 7 (Uniqueness of projections). The ∇-projection PS of P on S is unique if S is ∇∗ -flat and
minimizes the divergence D(θ(P ) : θ(Q)):

                                         ∇-projection:         PS = arg min D(θ(P ) : θ(Q)).                                                     (66)
                                                                             Q∈S

   The dual ∇∗ -projection PS∗ is unique if M ⊆ S is ∇-flat and minimizes the divergence D(θ(Q) : θ(P )):

                                        ∇∗ -projection:            PS∗ = arg min D(θ(Q) : θ(P )).                                                (67)
                                                                              Q∈S

   Let S ⊂ M and S 0 ⊂ M , then we define the divergence between S and S 0 as

                                                   D(S : S 0 ) :=         min        D(s : s0 ).                                                 (68)
                                                                       s∈S,s0 ∈S 0

    When S is a ∇-flat submanifold and S 0 ∇∗ -flat submanifold, the divergence D(S : S 0 ) between submani-
fold S and submanifold S 0 can be calculated using the method of alternating projections [8]. Let us remark


                                                                        16
that Kurose [61] reported a Pythagorean theorem for dually constant curvature manifolds that generalizes
the Pythagorean theorems of dually flat spaces.
    We shall concisely explain the space of Bregman spheres explained in details in [20]. Let D denote the
dimension of Θ. We define the lifting of primal coordinates θ to the primal potential function F = {θ̂ =
(θ, θD+1 = F (θ)) : θ ∈ Θ} using an extra dimension θD+1 . A Bregman ball Σ

                 BallF (C : r) := {P such that F (θ(P )) + F ∗ (η(C)) − hθ(P ), η(C)i ≤ r}                        (69)

can then be lifted to F: Σ̂ = {θ̂(P ) : P ∈ σ}. The boundary Bregman sphere σ = ∂Σ is lifted to ∂ Σ̂ = σ̂,
and the lifted points are all supported by a supporting (D + 1)-dimensional hyperplane (of dimension D):

                                  Hσ̂ : θD+1 = hθ − θ(C), η(C)i + F (θ(C)) + r.                                   (70)

Let Hσ̂− denotes the halfspaces bounded by Hσ̂ and containing θ̂(C) = (θ(C), F (θ(C))). A point P belongs
to a Bregman ball Σ iff θ(P   ˆ ) ∈ H − , see [20]. Reciprocally, a (D + 1)-dimensional hyperplane H : θD+1 =
                                        σ̂
hθ, ηa i + b cutting the potential function F yields a Bregman sphere σH of center C with θ(C) = ∇F ∗ (ηa )
and radius r = h∇F ∗ (ηa ), ηa i − F (θa ) + b = F ∗ (ηa ) + b, where θa = ∇F ∗ (ηa ). It follows that the intersection
of k Bregman balls is a (D − k)-dimensional Bregman ball, and that a Bregman sphere can be defined by
D + 1 points in general position since an hyperplane in the augmented space is defined by D + 1 points.
We can test whether a point P belongs to a Bregman ball with bounding Bregman sphere passing through
D + 1 points P1 , . . . , PD+1 or not by checking the sign of a (D + 2) × (D + 2) determinant:
                                                                                                         
                                                               1          ... 1                 1
           InBregmanBallF (P1 , . . . , Pd+1 ; P ) := sign  θ(P1 )       . . . θ(PD+1 )        θ(P )     .       (71)
                                                               F (θ(P1 )) . . . F (θ(PD+1 )) F (θ(P ))

We have:
                                                                P ∈ InBregmanBall◦F (P1 , . . . , PD+1 ; P )
                                                
                                                 = −1 ⇔
      InBregmanBallF (P1 , . . . , Pd+1 ; P ) :   =0   ⇔        P ∈ ∂InBregmanBallF (P1 , . . . , PD+1 ; P )      (72)
                                                  = +1 ⇔        P 6∈ InBregmanBallF (P1 , . . . , PD+1 ; P )
                                                

   Similarly, a dual-type Bregman ball Σ∗ can be defined by

                 Ball∗F (C : r) := {P such that F (θ(C)) + F ∗ (η(P )) − hθ(C), η(P )i ≤ r},                      (73)

and be lifted to the dual potential function F ∗ . Notice that Ball∗F (C : r) = BallF ∗ (C : r).
   In general, we have the following quadrilateral relation for Bregman divergences:
Property 4 (Bregman 4-parameter property [37]). For any four points P1 , P2 , Q1 , Q2 , we have the following
identity:

            BF (θ(P1 ) : θ(Q1 )) + BF (θ(P2 ) : θ(Q2 ))        −BF (θ(P1 ) : θ(Q2 )) − BF (θ(P2 ) : θ(Q1 ))
                                                               −(θ(P2 ) − θ(P1 ))> (η(Q1 ) − η(Q2 )) = 0.         (74)

    In summary, to define a dually flat space, we need a convex Bregman generator. When the α-geometries
are neither dually flat (eg., Cauchy manifolds [79], we may still build a dually flat structure on the manifold
by considering some Bregman generator (eg., Bregman-Tsallis generator for the dually flat Cauchy mani-
fold [79]). The dually flat geometry can be investigated under the wider scope of Hessian manifolds [120]
which consider locally potential functions. In general, a dually flat space can be built from any smooth
strictly convex generator F . For example, a dually flat geometry can be built on homogeneous cones with
the characteristic function F of the cone [120]. Figure 7 illustrates several common constructions of dually
flat spaces.


                                                          17
                  Exponential family
                 F : cumulant function

                                                           strictly convex and                    Dually flat space
           Homogeneous convex cone                     differentiable C 3 function               (Hessian structure)
           F : characteristic function                                                             (M, g, ∇, ∇∗ )


                   Mixture family
                 Shannon negentropy                                                             Bregman divergence



         Figure 7: Common dually flat spaces associated to smooth and strictly convex generators.


3.8     Hessian α-geometry: (M, F, α) ≡ (M, F g, F ∇−α , F ∇α
The dually flat manifold is also called a manifold with a Hessian structure [120] induced by a convex potential
                                                                                                   ∗
function F . Since we built two dual affine connections BF ∇ = F ∇ and BF ∇∗ = F ∇∗ = F ∇, we can
build a family of α-geometry as follows:
                                         F                                 F ij
                                             gij (θ) = ∂i ∂j F (θ),         g (η) = ∂ i ∂ j F (η),                     (75)
and
                 F       1−α                  F α ∗         ∗           1+α i j k ∗
                     Γα
                      ijk (θ) =
                              ∂i ∂j ∂k F (θ),  Γijk (η) = F Γαijk (η) =     ∂ ∂ ∂ F (η).                               (76)
                          2                                              2
   Thus when α = ±1, the Hessian α-geometry is dually flat.
   We now consider information manifolds induced by parametric statistical models.

3.9     Expected α-manifolds of a family of parametric probability distributions:
        (P, P g, P ∇−α , P ∇α )
Informally speaking, an expected manifold is an information manifold built on a regular parametric family
of distributions. It is sometimes called “expected” manifold or “expected” geometry in the literature [140]
because the components of the metric tensor g and the Amari-Chentsov cubic tensor C are expressed using
statistical expectations E· [·].
    Let P be a parametric family of probability distributions:

                                                          P := {pθ (x)}θ∈Θ ,                                           (77)

with θ belonging to the open parameter space Θ. The order of the family is the dimension of its parameter
space. We define the likelihood function11 L(θ; x) := pθ (x) as a function of θ, and its corresponding log-
likelihood function:
                                    l(θ; x) := log L(θ; x) = log pθ (x).                              (78)
The score vector:
                                                          sθ = ∇θ l = (∂i l)i ,                                        (79)
                                                     ∂
indicates the sensitivity of the likelihood ∂i l:=: ∂θ i
                                                         l(θ; x).
   The Fisher information matrix (FIM) of D × D for dim(Θ) = D is defined by:

                                                     P I(θ) := Eθ [∂i l∂j l]ij  0,                                    (80)

where  denotes the Löwner order. That is, for two symmetric positive-definite matrices A and B, A  B
if and only if matrix A − B is positive semidefinite. For regular models [25], the FIM is positive definite:
P I(θ)  0, where A  B if and only if matrix A − B is positive-definite.
 11 The likelihood function is an equivalence class of functions defined modulo a positive scaling factor.




                                                                      18
    The FIM is invariant by reparameterization of the sample space X , and covariant by reparameterization
of the parameter space Θ, see [25]. That is, let p̄(x; η) = p(θ(η); x). Then we have:
                                                             >               
                                                    ¯ = ∂θi
                                                    I(η)          I(θ(η))
                                                                            ∂θi
                                                                                   .                                                  (81)
                                                           ∂ηj ij           ∂ηj ij
               h         i
                   ∂θi
Matrix Jij =       ∂ηj        is the Jacobian matrix.
                         ij

Example 1. For example, consider the family

                                                  (x − µ)2
                                                                                
                                       1
                  N = p(x; µ, σ) = √        exp(−          )) : (µ, σ) ∈ R × R++                                                      (82)
                                       2πσ          2σ 2

of univariate normal distributions. The 2D parameter vector is λ = (µ, σ) with µ denoting the mean and
σ the standard deviation. Another common parameterization of the normal family is λ0 = (µ, σ 2 ). The λ0
parameterization extends naturally to d-variance normal distributions with λ0 = (µ, Σ), where Σ denotes the
covariance matrix (with Σ = σ 2 when d = 1). For multivariate normal distributions, the λ-parameterization
can be interpreted as λ = (µ, L> ) where L> is the upper triangular matrix in the Cholesky decomposition
(when d = 1, L> = σ). We have the following Fisher information matrices in the λ-parameterization and
λ0 -parameterization:                        " 1      # 
                                                    0          1
                                                                      
                                               λ22            σ 2  0
                                    Iλ (λ) =            =                                              (83)
                                               0 λ22          0 σ22
                                                                         2

and                                                         "                      #
                                                                1                                  1
                                                                         0
                                                                                                                   
                                                     0          λ2                                 σ2       0
                                               Iλ0 (λ ) =                1             =                 1                            (84)
                                                                0       2λ22                       0    2σ 4

   Since the FIM is covariant, we have the following the change of transformation:

                                                    Iλ0 (λ0 ) = Jλ,λ
                                                                 >            0
                                                                    0 Iλ (λ (λ )) Jλ,λ0 ,                                             (85)

with                                                                                          
                                                                               1        0
                                                             Jλ0 ,λ =                                                                 (86)
                                                                               0       2σ
   Thus we check that
                                                                1                                                       1
                                                                                                                           
                                                1    0          σ2      0               1           0                   σ2   0
                                Iλ (λ) =                              1                                     =                2        (87)
                                                0   2σ          0    2σ 4               0          2σ                   0    σ2

   Notice that the infinitesimal length elements are invariant: dsλ = dsλ0 .
    As a corollary, notice that we can recognize the Euclidean metric in any other coordinate system if the
                                  >
metric tensor g can be written Jλ,λ  0 Jλ,λ0 . For example, the Riemannian geometry induced by a dually flat

space with a separable potential function is Euclidean [49].
    In statistics, the FIM plays a role in the attainable precision of unbiased estimators. For any unbiased
estimator, the Cramér-Rao lower bound [71] on the variance of the estimator is:
                                                                                   1 −1
                                                         Varθ [θ̂n (X)]             PI (θ).                                          (88)
                                                                                   n
    Figure 8 illustrates the Cramér-Rao lower bound (CRLB) for the univariate distributions: At regular
grid locations (µ, σ) of the upper space of normal parameters, we repeat 200 runs (trials) of estimating the
normal parameters (µ,\  σ) using the MLE on 100 iid samples x1 , . . . , xn ∼ N (µ, σ). The sample mean and the
sample covariance matrix are calculated for the number of trials and displayed as back ellipses. The Fisher


                                                                         19
Figure 8: Visualizing the Cramér-Rao lower bound: The red ellipses display the Fisher information matrix
of normal distributions at grid locations. The black ellipses are sample covariance matrices centered at the
sample means calculated by repeating 200 runs of sampling 100 iid variates for the normal parameters of the
grid.




                                                    20
information matrix is plotted as red ellipses at the grid locations: The red ellipses have semi-axis parallel to
the coordinate system since the parameters µ and σ are orthogonal (diagonal FIM). This is not true anymore
for the sample covariance matrix of the MLE estimator, and the centers of the sample covariance matrices
deviate from the grid locations.
    We report the expression of the FIM for two important generic parametric family of probability dis-
tributions: (1) an exponential family (with its prominent multivariate normal family), and (2) a mixture
family.
Example 2 (FIM of an exponential family E). An exponential family [84] E is defined for a sufficient
statistic vector t(x) = (t1 (x), . . . , tD (x)), and an auxiliary carrier measure k(x) by the following canonical
density:                  (                                                !               )
                                                 XD
                      E = pθ (x) = exp              ti (x)θi − F (θ) + k(x) such that θ ∈ Θ ,                (89)
                                           i=1

where F is the strictly convex cumulant function (also called log-normalizer, and log partition function or
free energy in statistical mechanics). Exponential families include the Gaussian family, the Gamma and Beta
families, the probability simplex ∆, etc. The FIM of an exponential family is given by:
                                                         2          2 ∗     −1
                          E I(θ) = CovX∼pθ (x) [t(x)] = ∇ F (θ) = (∇ F (η))     0.                          (90)
Natural parameters beyond vector types can also be used in the canonical decomposition of the density of an
exponential family: For example, we may use a matrix type for defining the zero-centered multivariate Gaus-
sian family or the Wishart family, a complex
                                          PD numbers for defining the complex-valued Gaussian distribution
family, etc. We then replace the term i=1 ti (x)θi in Eq. 89 by an inner product defined for the natural
parameter type (e.g., dot product for vectors, matrix product trace for matrices, etc). Furthermore, natural
parameters can be of compound types: For example, the multivariate Gaussian distribution can be written
using θ = (θv , θM ) where θv is a vector part and θM a matrix part, see [84].
    Let Σ = [σij ] denote the covariance matrix and Σ−1 = [σ ij ] the precision matrix of a multivariate normal
distribution. The Fisher information matrix of the multivariate Gaussian [114, 121] N (µ, Σ) is given by

                                          µij          Σ = [σij ] 
                                I(µ, Σ) = σ                  0              µ                                (91)
                                           0       σ il σ jk + σ ik σ jl Σ = [σkl ]

Notice that the lower right block matrix is a 4D tensor of dimension d × d × d × d. The zero subblock matrices
in the FIM indicate that the parameters µ and Σ are orthogonal to each other. In particular, when d = 1,
since σ 11 = σ12 , we recover the Fisher information matrix of the univariate Gaussian:
                                                       1         
                                                               0
                                             I(µ, Σ) = σ2                                                 (92)
                                                         0 2σ1 4
We refer to [63] for the FIM of a Gaussian distribution using other canonical parameterizations (natu-
ral/expectation parameters of exponential family).
Example 3 (FIM of a mixture family M). A mixture family is defined for D + 1 functions F1 , . . . , FD and
C as:                        (                                          )
                                     XD
                       M = pθ (x) =     θi Fi (x) + C(x) such that θ ∈ Θ ,                             (93)
                                             i=1
                                                                                              R
where the functions {Fi (x)}i Rare linearly independent on the common support X and satisfying Fi (x)dµ(x) =
0. Function C is such that C(x)dµ(x) = 1. Mixture families include statistical mixtures with prescribed
component distributions and the probability simplex ∆. The FIM of a mixture family is given by:
                                                             Z
                                                Fi (x)Fj (x)      Fi (x)Fj (x)
                        M I(θ) = EX∼pθ (x)                2
                                                              =                dµ(x)  0.               (94)
                                                 (pθ (x))       X     pθ (x)

                                                        21
   The family of Gaussian mixture model (GMM) with prescribed component distributions (ie., convex weight
combinations of D + 1 Gaussian densities) form a mixture family [93].
    Notice that the probability simplex of discrete distributions can be both modeled as an exponential family
or a mixture family [8].
    The expected α-geometry is built from the expected dual ±α-connections. The Fisher “information metric”
tensor is built from the FIM as follows:
                                                           >
                                           P g(u, v) := (u)θ P I(θ) (v)θ                                 (95)

   The expected exponential connection and expected mixture connection are given by
                                     e
                                     P∇     := Eθ [(∂i ∂j l)(∂k l)] ,                                    (96)
                                     m
                                     P∇     := Eθ [(∂i ∂j l + ∂i l∂j l)(∂k l)] .                         (97)

   The dualistic structure is denoted by (P, P g, m    e
                                                  P ∇, P ∇) with Amari-Chentsov cubic tensor called the
skewness tensor:
                                        Cijk := Eθ [∂i l∂j l∂k l] .                                (98)
   It follows that we can build a one-family of expected information α-manifolds:

                                          (P, P g, P ∇−α , P ∇+α ) α∈R ,
                                        
                                                                                                         (99)

with
                               α                                 1−α
                            P Γ ij,k (θ)   := Eθ [∂i ∂j l∂k l] +      Cijk (θ),                         (100)
                                                                 2               
                                                                1−α
                                            = Eθ     ∂i ∂j l +       ∂i l∂j l (∂k l) .                  (101)
                                                                 2
   The Levi-Civita metric connection is recovered as follows:
                                                −α
                                    ¯       + P ∇α
                                           P∇
                                  P∇ =             = LC
                                                     P ∇ :=
                                                            LC
                                                               ∇(P g)                                   (102)
                                           2
   The α-Riemann-Christoffel curvature tensor is:
                                        α          α        rs
                                                                 Γα     α       α     α
                                                                                          
                          P Rijkl = ∂i Γjk,l − ∂j Γik,l + g       ik,r Γjs,l − Γjk,r Γis,l ,            (103)
        α         −α
with Rijkl   = −Rijlk . We check that the expected ±α-connections are coupled with the metric: ∂i gjk =
 α        −α
Γij,k + Γik,j .
    In case of an exponential family E or a mixture family M equipped with the dual exponential/mixture
connection, we get dually flat manifolds (Bregman geometry).
    Indeed, for the exponential/mixture families, it is easy to check that the Christoffel symbols of ∇e and
  m
∇ vanish:
                                         e     m       e     m
                                         M Γ = M Γ = E Γ = E Γ = 0.                                    (104)

3.10    Criteria for statistical invariance
So far we have explained how to build an information manifold (or information α-manifold) from a pair
of conjugate connections. Then we reported two ways to obtain such a pair of conjugate connections: (1)
from a parametric divergence, or (2) by using the predefined expected exponential/mixture connections. We
now ask the following question: Which information manifold makes sense in Statistics? We can refine the
question as follows:

   • Which metric tensors g make sense in statistics?


                                                         22
                                      p1    p2 p3       p4 p5 p6 p7            p8   p

                                                    coarse graining
                                     p1 + p2 p3 + p4 + p5 p6 p7 + p8                pA


                                                                                       0
Figure 9: A divergence satisfies the property of information monotonicity iff D(θĀ : θĀ ) ≤ D(θ : θ0 ). Here,
parameter θ represents a discrete distribution.


   • Which affine connections ∇ make sense in statistics?
   • Which statistical divergences make sense in statistics (from which we can get the metric tensor and
     dual connections)?

   By definition, an invariant metric tensor g shall preserve the inner product under important statistical
mappings called Markov embeddings. Informally, we embed ∆D into ∆D0 with D0 > D and the induced
metric should be preserved (see [8], page 62).
Theorem 8 (Uniqueness of Fisher information metric [26, 129]). The Fisher information metric is the
unique invariant metric tensor under Markov embeddings up to a scaling constant.
   A D-dimensional parameter (discrete) divergence satisfies the information monotonicity12 if and only if:
                                                          0
                                                 D(θĀ : θĀ ) ≤ D(θ : θ0 )                                (105)

for any
      P coarse-grained  partition A = {Ai }E  i=1 of [D] = {1, . . . , D} (A-lumping [34]) with E ≤ D, where
 i            j
θĀ = j∈Ai θ for i ∈ [E]. This concept of coarse-graining is illustrated in Figure 9.
    A separable divergence D(θ1 : θ2 ) is a divergence that can be expressed as the sum of elementary scalar
divergences d(x : y):                                    X
                                          D(θ1 : θ2 ) :=   d(θ1i : θ2j ).                              (106)
                                                                  i

For example, the squared Euclidean distance D(θ1 : θ2 ) = i (θ1i − θ2i )2 is a separable
                                                               P
                                                                                          divergence for the
                                                                                         pP
scalar Euclidean divergence d(x : y) = (x − y)2 . The Euclidean distance DE (θ1 , θ2 ) =        i    i 2
                                                                                            i (θ1 − θ2 ) is not
separable because of the square root operation.
    The only invariant and decomposable divergences when D > 1 are f -divergences [56] defined for a convex
functional generator f :
                                                D        0
                                               X         θi
                               If (θ : θ0 ) :=     θi f      ≥ f (1), f (1) = 0                          (107)
                                               i=1
                                                         θ i

    The standard f -divergences are defined for f -generators satisfying f 0 (1) = 0 (choose fλ (u) := f (u) +
λ(u − 1) since Ifλ = If ), and f 00 (u) = 1 (scale fixed).
    Statistical f -divergences are invariant [108] under one-to-one/sufficient statistic transformations y = t(x)
of sample space: p(x; θ) = q(y(x); θ):

                                                                          p(x; θ0 )
                                                            Z                      
                             If [p(x; θ) : p(x; θ0 )]   =       p(x; θ)f              dµ(x),
                                                           X               p(x; θ)
                                                                          q(y; θ0 )
                                                          Z                        
                                                        =     q(y; θ)f                dµ(y),
                                                           Y              q(y; θ)
                                                        = If [q(y; θ) : q(y; θ0 )].
 12 This property could be renamed as the “distance coarse-binning inequality property.”




                                                             23
   The dual f -divergences for reference duality is

                     If ∗ [p(x; θ) : p(x; θ0 )] = If [p(x; θ0 ) : p(x; θ)] = If  [p(x; θ) : p(x; θ0 )]      (108)

for the standard conjugate f -generator (diamond f  generator) with:
                                                           
                                                           1
                                            f  (u) := uf     .                                              (109)
                                                           u
One can check that f  is a standard f -generator when f is standard.
  Let us report some common examples of f -divergences:

   • The family of α-divergences:
                                                                   Z                                  
                                                  4                          1−α          1+α
                                 Iα [p : q] :=              1−           p    2    (x)q    2    (x)dµ(x) ,   (110)
                                               1 − α2
                                         1+α
                            4
     obtained for f (u) = 1−α 2 (1 − u
                                       2 ). The α-divergences include:



        – the Kullback-Leibler when α → 1:
                                                                Z
                                                                                     p(x)
                                               KL[p : q] =              p(x) log          dµ(x),             (111)
                                                                                     q(x)
          for f (u) = − log u.
        – the reverse Kullback-Leibler α → −1:
                                                       Z
                                                                             q(x)
                                       KL∗ [p : q] =        q(x) log              dµ(x) = KL[q : p],         (112)
                                                                             p(x)
          for f (u) = u log u.
        – the symmetric squared Hellinger divergence:
                                                  Z p      p
                                     H 2 [p : q] = ( p(x) − q(x))2 dµ(x),                                    (113)
                       √
          for f (u) = ( u − 1)2 (corresponding to α = 0)
        – the Pearson and Neyman chi-squared divergences [90], etc.

   • the Jensen-Shannon divergence:
                                  Z                                             
                                1                2p(x)                  2q(x)
                    JS[p : q] =      p(x) log             + q(x) log               dµ(x),                    (114)
                                2             p(x) + q(x)            p(x) + q(x)

     for f (u) = −(u + 1) log 1+u
                               2 + u log u.

   • the Total Variation                                       Z
                                                           1
                                           TV[p : q] =              |p(x) − q(x)|dµ(x),                      (115)
                                                           2
     for f (u) = 21 |u − 1|. The total variation distance is the only metric f -divergence.

    The f -topology is the topology generated by open f -balls, open balls with respect to f -divergences.
AtopologyT is saidstrongerthan atopologyT 0 if T contains all the open sets of T 0 . Csiszar’s theorem [33]
states that when |α| < 1, the α-topology is equivalent to the topology induced by the total variation metric
distance. Otherwise, the α-topology is stronger than the TV topology.
    Let us state an important feature of f divergences:


                                                               24
Theorem 9. The f -divergences are invariant by diffeomorphisms m(x) of the sample space X : Let Y =
m(X), and Xi ∼ pi with Yi = m(Xi ) ∼ qi . Then we have If [q1 : q2 ] = If [p1 : p2 ].
Example 4. Consider the exponential distributions and the Rayleigh distributions which are related by:
                                                       √
                                                                                 
                                                                               1
                X ∼ Exponential(λ) ⇔ Y = m(X) = X ∼ Rayleigh σ = √                  .
                                                                               2λ
The densities of the exponential distributions are defined by

                                pλ (x) = λ exp(−λx) with support X = [0, ∞),

and the densities of the Rayleigh distributions are defined by

                                                  x2
                                                    
                                       x
                             qσ (x) = 2 exp − 2 with support X = [0, ∞).
                                      σ          2σ

We have
                                                                  λ22           σ12 − σ22
                                                                       
                                    DKL [qσ1 : qσ2 ] = log                  +             .
                                                                  λ21              σ22
It follows that
                                                                        
                                                          1          1
                              DKL [pλ1 : pλ2 ] = DKL q √       : q√
                                                          2λ1        2λ1
                                                                            
                                                     2λ1           1       1
                                               = log     + 2λ2         −
                                                     2λ2          2λ1     λ2
                                                      
                                                      λ1     λ2
                                               = log       +      − 1.
                                                      λ2     λ1

    A remarkable property is that invariant standard f -divergences yield the Fisher information matrix and
the α-connections. Indeed, the invariant standard f -divergences is related infinitesimally to the Fisher metric
as follows:

                                                         Z                                    
                                                                                p(x; θ + dθ)
                       If [p(x; θ) : p(x; θ + dθ)]   =        p(x; θ)f                             dµ(x)   (116)
                                                                                  p(x; θ)
                                                     Σ   1            i   j
                                                     =     F gij (θ)dθ dθ                                  (117)
                                                         2
   A statistical parameter divergence D on a parametric family of distributions P yields an equivalent
parameter divergence P D:
                                             0                       0
                                    P D(θ : θ ) := D[p(x; θ) : p(x; θ )].                        (118)
Thus we can build the information manifold induced by this parameter divergence P D(· : ·). For P D(· : ·) =
                                                  I                  (I )∗       ∗
If [· : ·], the induced ±1-divergence connections Pf ∇ := P If ∇ and P f ∇ := P If ∇ are precisely the expected
±α-connections (derived from the exponential/mixture connections) with:

                                                 α = 2f 000 (1) + 3.                                       (119)

   Thus the invariant connections which coincide with the connections induced by the invariant statistical
divergences are the expected α-connections. Note that the curvature of an expected α-connection depends
both on α and on the considered statistical model [64].




                                                         25
3.11    Fisher-Rao expected Riemannian manifolds: (P, P g)
Historically, a first manifold modeling of a regular parametric family of distributions P = {pθ (x)}θ was to
consider the Fisher Information Matrix (FIM) as the Riemannian metric tensor g (see [53, 111]), with:

                                            P I(θ) := Epθ [∂i l∂j l] ,                                   (120)
               ∂
where ∂i l:=: ∂θ i
                   log p(x; θ). Under some regularity conditions, we can rewrite the FIM:

                                           P I(θ) := −Epθ [∂i ∂j l] .                                    (121)

   The Riemannian geodesic metric distance Dρ is commonly called the Fisher-Rao distance:
                                                  Z 1q
                                Dρ (pθ1 , pθ2 ) =      γ̇(t)> gγ(t) γ̇(t)dt,                             (122)
                                                     0

where γ denotes the geodesic passing through γ(0) = θ1 and q      γ(1) = θ2 . The Fisher-Rao distance can also
                                                               R1
be defined as the shortest path length: Dρ (pθ1 , pθ2 ) = inf γ 0 γ̇(t)> gγ(t) γ̇(t)dt.

Definition 6 (Fisher-Rao distance). The Fisher-Rao distance is the geodesic metric distance of the Fisher-
Riemannian manifold (P, P g).
   Let us give some examples of Fisher-Riemannian manifolds:
   • The Fisher-Riemannian manifold of the family of categorical distributions (also called finite discrete
     distributions in [8]) amount to the spherical geometry [58] (spherical manifold).
   • The Fisher-Riemannian manifold of the family of bivariate location-scale families amount to hyperbolic
     geometry (hyperbolic manifold).
   • The Fisher-Riemannian manifold of the family of location families amount to Euclidean geometry
     (Euclidean manifold).
                                                                                Σ
    The first fundamental form of the Riemannian geometry is ds2 = hdx, dxi = gij dxi dxj where ds denotes
the line element.
    This Riemannian geometric structure applied to a family of parametric probability distributions was
first proposed by Harold Hotelling [53] (in a handwritten note of 1929, reprinted typeset in [123]) and
independently later by C. R. Rao [111] (1945, reprinted in [110]). In a similar vein, Jeffreys [55] proposed to
use the volume element of a manifold as an invariant prior: The eponym Jeffreys prior in 1946.
    Notice that for a parametric family of probability distributions P, the Riemannian structure (P, P g)
                                                                     I     I
coincides with the self-dual conjugate connection manifold (P, P g, Pf ∇, Pf ∇∗ ) induced by a symmetric f -
divergence like the squared Hellinger divergence.
    The exponential map expp at point p ∈ M provides a way to map back a vector v ∈ Tp to a point
expp (v) ∈ M (when well-defined). The exponential map can be used to parameterize a geodesic γ with
γ(0) = p and unit tangent vector γ̇(0) = v: t 7→ expp (tv). For geodesically complete manifolds, the
exponential map is defined everywhere.

3.12    The monotone α-embeddings and the metric gauge freedom
Another common mathematically equivalent expression of the FIM [25] is given by:
                                         Z p            p
                             Iij (θ) := 4 ∂i p(x; θ)∂j p(x; θ)dµ(x).                                     (123)

This form of the FIM is well-suited to prove that the FIM is always a positive semi-definite matrix [25]
(I(θ)  0). It turns out that we can define a family of equivalent representations of the FIM using the
α-embedding [139] of the parametric family.


                                                         26
   First, we define the α-representation of densities lα (x; θ) := kα (p(x; θ)) with:
                                                2 1−α
                                                  1−α u
                                                          2 ,    if α 6= 1,
                                     kα (u) :=                                                                     (124)
                                                  log u,         if α = 1.
    The function lα (x; θ) is called the α-likelihood function. Then the α-representation of the FIM, the α-FIM
for short, is expressed as:                      Z
                                        α
                                       Iij (θ) :=   ∂i lα (x; θ)∂j l−α (x; θ)dµ(x).                                (125)

                                           α
                                                                    ∂i lα ∂j l−α dµ(x). Expanding the α-FIM, we get:
                                                                R
   We can rewrite compactly the α-FIM, as Iij (θ) =
                                                      1−α          1+α
                                       1
                                           R
                     α              R1−α2 ∂i p(x; θ)
                                                         2 ∂ p(x; θ) 2 dµ(x)
                                                             j                         for α 6= ±1
                    Iij (θ) =                                                                                      (126)
                                       ∂i log p(x; θ)∂j p(x; θ)dµ(x)                   for α ∈ {−1, 1}

    The 1-representation of the density is called the logarithmic representation (or e-representation), the −1-
representation the mixture representation (or m-representation), and its 0-representation is called the square
root representation. The set of α-scores vectors Bα := {∂i lα }i are interpreted as the tangent basis vectors
of the α-base Bα . Thus the FIM is α-independent.
    Furthermore, the α-representation of the FIM can be rewritten under mild conditions [25] as:
                                                 Z
                                α            2             1+α
                               Iij (θ) = −          p(x; θ) 2 ∂i ∂j lα (x; θ)dµ(x).                       (127)
                                           1+α
   Since we have:                                                                      
                                                      1−α                   1−α
                                    ∂i ∂j lα (x; θ) = p 2       ∂i ∂j l +       ∂i l∂j l ,                         (128)
                                                                             2
it follows that:                                                                
                                 α             2                      1−α
                                Iij (θ) = −             −Iij (θ) +        Iij        = Iij (θ).                    (129)
                                              1+α                      2
   Notice that when α = 1, we recover the equivalent expression of the FIM (under mild conditions):
                                               1
                                              Iij (θ) = −E[∇2 log p(x; θ)].                                        (130)

In particular, when the family is an exponential family [84] with cumulant function F (θ) (satisfying the mild
conditions), we have:
                                               I(θ) = ∇2 F (θ).                                          (131)
    Zhang [139, 69] further discussed the representation/reference biduality which was confounded in the
α-geometry.
    Gauge freedom of the Riemannian metric tensor has been investigated under the framework of (ρ, τ )-
monotone embeddings [139, 98, 69] in information geometry: Let ρ and τ be two strictly increasing functions,
and f a strictly convex function such that f 0 (ρ(u)) = τ (u) (with f ∗ denoting its convex conjugate). Ob-
serve that the set of strictly increasing real-valued univariate functions has a group structure for the group
operation chosen as the functional composition ◦. Let us write pθ (x) = p(x; θ).
    The (ρ, τ )-metric tensor ρ,τ g(θ) = [ ρ,τ gij (θ)]ij can be derived from the (ρ, τ )-divergence:
                                     Z
                      Dρ,τ (p : q) = (f (ρ(p(x))) + f ∗ (τ (q(x))) − ρ(p(x))τ (q(x))) dν(x)              (132)




                                                            27
   We have:
                                           Z
                       ρ,τ
                             gij (θ)   =       (∂i ρ(pθ (x))) (∂j τ (pθ (x))) dν(x),                          (133)
                                           Z
                                       =       ρ0 (pθ (x))τ 0 (pθ (x)) (∂i pθ (x)) (∂j pθ (x)) dν(x),         (134)
                                           Z
                                       =       f 00 (ρ(pθ (x))) (∂i ρ(pθ (x))) (∂j ρ(pθ (x))) dν(x),          (135)
                                           Z
                                       =       (f ∗ )00 (τ (pθ (x))) (∂i τ (pθ (x))) (∂j τ (pθ (x))) dν(x).   (136)


3.13    Dually flat spaces and canonical Bregman divergences
We have described how to build a dually flat space from any strictly convex and smooth generator F : A
Hessian structure is built from F (θ) with Riemannian Hessian metric ∇2 F (θ), and the convex conjugate
F ∗ (η) (obtained by the Legendre-Fenchel duality) yields the dual Hessian structure with Riemannian Hessian
metric ∇2 F ∗ (η). The dual connections ∇ and ∇∗ are coupled with the metric. The connections are defined
by their respective Christoffel symbols Γ(θ) = 0 and Γ∗ (η) = 0, showing that they are flat connections.
     Conversely, it can be proved [8] that given two dually flat connections ∇ and ∇∗ , we can reconstruct two
dual canonical strictly convex potential functions F (θ) and F ∗ (η) such that η = ∇F (θ) and θ = ∇F ∗ (η).
The canonical divergence AF,F ∗ yields the dual Bregman divergences BF and BF ∗ .
                                                                                        2
     The only symmetric Bregman divergences are squared Mahalanobis distances MQ          [20] with the Maha-
lanobis distance defined by:                        q
                                           MQ (θ, θ0 ) =       (θ0 − θ)> Q(θ0 − θ).                           (137)
   Let Q = LL> be the Cholesky decomposition of a positive-definite matrix Q  0. It is well-known that
the Mahalanobis distance MQ amounts to the Euclidean distance on affinely transformed points:
                                  2
                                 MQ (θ, θ0 )    =     ∆θ> Q∆θ,                                                (138)
                                                =     ∆θ> LL> ∆θ,                                             (139)
                                                =     MI2 (L> θ, L> θ0 ) = kL> θ − L> θ0 k2 ,                 (140)
where ∆θ = θ0 − θ.
                                           2
    The squared Mahalanobis distance MQ      does not satisfy the triangle inequality, but the Mahalanobis
distance MQ is a metric distance. We can convert a Mahalanobis distance MQ1 into another Mahalanobis
distance MQ2 , and vice versa, as follows:
Proof. Let us write matrix Q = L> L  0 using the Cholesky decomposition. Then we have
                 MQ (θ1 , θ2 ) = MI (L> θ1 , L> θ2 ) ⇔ MI (θ1 , θ2 ) = MQ ((L> )−1 θ1 , ((L> )−1 θ2 ).        (141)
Then we have for two symmetric positive-definite matrices Q1 = L>                  >
                                                                1 L1  0 and Q2 = L2 L2  0:

                    MQ1 (θ1 , θ2 ) = MI (L>       >               > −1 >
                                          1 θ1 , L1 θ2 ) = MQ2 ((L2 ) L1 θ1 , (L>
                                                                                2)
                                                                                  −1 >
                                                                                    L1 θ2 ).                  (142)
It follows that we have:
                                   MQ1 (θ1 , θ2 ) = MQ2 ((L>
                                                           2)  L1 θ1 , (L>
                                                             −1 >
                                                                         2)
                                                                           −1 >
                                                                             L1 θ2 ).                         (143)


  We have MQ   2
                 (θ1 , θ2 ) = BF (θ1 , θ2 ) (Bregman divergence) with F (θ) = 12 θ> Qθ for a positive-definite
matrix Q  0. The convex conjugate F ∗ (η) = 12 η > Q−1 η (with Q−1  0). We have η = Q−1 θ and η = Qθ.
                                                                             2         2
We have the following identity between the dual Mahalanobis divergences MQ       and MQ  −1 :


                                                 2               2
                                                MQ (θ1 , θ2 ) = MQ −1 (η1 , η2 ).                             (144)


                                                                28
                                                                                             R                  
     When the Bregman generator is based on an integral, i.e., the log-normalizer
                                                                       R          F (θ) = log exp(ht(x), θidµ(x)
for exponential families E, or the negative Shannon entropy F (θ) = mθ (x) log m(η)dµ(x) for mixture fam-
ilies M, the associated Bregman divergences BF,E or BF,M can be relaxed and interpreted as a statistical
distance. We explain how to obtain the reconstruction below:

   • Consider an exponential family E of order D with densities defined according to a dominating measure
     µ:
                                E = {pθ (x) = exp(θ> t(x) − F (θ)) : θ ∈ Θ},                        (145)
     where the natural parameter θ and the sufficient statistic vector t(x) belong to RD . We have the
     integral-based Bregman generator:
                                                      Z                  
                               F (θ) = FE (pθ ) = log    exp(θ> t(x))dµ(x) ,                     (146)

     and the dual convex conjugate
                                                                 Z
                                         F ∗ (η) = −h(pθ ) =         p(x) log p(x)dµ(x),                               (147)
                   R
     where h(p) = − p(x) log p(x)dµ(x) denotes Shannon’s entropy.
     Let λ(i) denotes the i-th coordinates of vector λ, and let us calculate the inner product θ1> η2 =
     P
       i θ1 (i)η2 (i) of the Legendre-Fenchel divergence.      P       We have η2 (i) P    = Epθ2 [ti (x)]. Using the linear
     property
     P            of  the expectation    E[·], we    find that   i θ1 (i)η 2 (i) = Ep θ2
                                                                                         [  i θ1 (i)ti (x)]. Moreover, we have
       i θ 1 (i)t i (x) = (log pθ 1
                                    (x)) + F (θ 1 ).  Thus  we have:

                                 θ1> η2 = Epθ2 [log pθ1 + F (θ1 )] = F (θ1 ) + Epθ2 [log pθ1 ] .                       (148)

     It follows that we get

                              BF,E [pθ1 : pθ2 ]   =   F (θ1 ) + F ∗ (η2 ) − θ1> η2 ,                                   (149)
                                                  = F (θ1 ) − h(pθ2 ) − Epθ2 [log pθ1 ] − F (θ1 ),                     (150)
                                                                   
                                                               pθ
                                                  = Epθ2 log 2 =: DKL∗ [pθ1 : pθ2 ],                                   (151)
                                                               p θ1

     By relaxing the exponential family densities pθ1 and pθ2 to be arbitrary densities p1 and p2 , we obtain
     the reverse KL divergence between p1 and p2 from the dually flat structure induced by the integral-
     based log-normalizer of an exponential family:
                                                          Z
                                                       p2                 p2 (x)
                          DKL [p1 : p2 ] = Ep2 log
                              ∗                            = p2 (x) log          dµ(x),                 (152)
                                                       p1                 p1 (x)
                                         = DKL [p2 : p1 ].                                              (153)

     Thus we have recovered the reverse Kullback-Leibler divergence DKL∗ from BF,E .
     The dual divergence D∗ [p1 : p2 ] := D[p2 : p1 ] is obtained by swapping the distribution parameter
     orders. We have:
                                                                         
                         ∗                                             p1
                       DKL ∗ [p1 : p2 ] := DKL∗ [p2 : p1 ] = Ep1   log      =: DKL [p1 : p2 ],     (154)
                                                                       p2
                            ∗
     and DKL∗ [p1 : p2 ] = DKL∗ [p2 : p1 ] = DKL [p2 : p1 ].


     To summarize, the canonical Legendre-Fenchel divergence associated with the log-normalizer of an
     exponential family amounts to the statistical reverse Kullback-Leibler divergence between pθ1 and pθ1


                                                            29
  (or the KL divergence between the swapped corresponding densities): DKL [pθ1 : pθ2 ] = BF (θ2 : θ1 ) =
  AF,F ∗ (θ2 : η1 ). Notice that it is easy to check that DKL [pθ1 : pθ2 ] = BF (θ2 : θ1 ) [14, 16]. Here, we took
  the opposite direction by constructing DKL from BF .
  We may consider an auxiliary carrier term k(x) so that the densities write pθ (x) = exp(θ> t(x) − F (θ) +
  k(x)). Then the dual convex conjugate writes [88] as F ∗ (η) = −h(pθ ) + Epθ [k(x)].
  Notice that since the Bregman generator is defined up to an affine term, we may consider the equivalent
  generator F (θ) = − log pθ (ω) instead of the integral-based generator. This approach yields ways
  to build formula bypassing the explicit use of the log-normalizer for calculating various statistical
  distances [94].
• In this second example, we consider a mixture family
                                   (       D                   D
                                                                           )
                                          X                    X
                             M = mθ =         θi pi (x) + (1 −   θi )p0 (x) ,                                            (155)
                                                      i=1                     i=1

  where p0 , . . . , pD are D + 1 linearly independent probability densities. The integral-based Bregman
  generator F is chosen as Shannon negentropy:
                                                          Z
                             F (θ) = FM (mθ ) = −h(mθ ) = mθ (x) log mθ (x)dµ(x).                   (156)


  We have                                              Z
                                    ηi = [∇F (θ)]i =        (pi (x) − p0 (x)) log mθ (x)dµ(x),                           (157)

  and the dual convex potential function is
                                      Z
                          F ∗ (η) = − p0 (x) log mθ (x)dµ(x) = h× (p0 : mθ ),                                            (158)

  i.e., the cross-entropy between the density p0 and the mixture mθ . Let us calculate the inner product
  θ1> η2 of the Legendre-Fenchel divergence as follows:


             X             Z                                               Z X
                  θ1 (i)       (pi (x) − p0 (x)) log mθ2 (x)dµ(x)     =                 θ1 (i)pi (x) log mθ2 (x)dµ(x)
              i                                                                     i
                                                                               X
                                                                           −            θ1 (i)p0 (x) log mθ2 (x)dµ(x).   (159)
                                                                                i


  That is                                   Z X                           X
                                 θ1> η2 =         θ1 (i)pi log mθ2 dµ −        θ1 (i)p0 log mθ2 dµ.                      (160)
                                              i                           i




                                                              30
      Thus it follows that we have the following statistical distance:
         BF,M [mθ1 : mθ2 ]   := F (θ1 ) + F ∗ (η2 ) − θ1> η2 ,                                     (161)
                                              Z                     Z X
                             = −h(mθ1 ) − p0 (x) log mθ2 (x)dµ(x) −     θ1 (i)pi (x) log mθ2 (x)dµ(x)
                                                                                         i
                                      X
                                  +       θ1 (i)p0 (x) log mθ2 (x)dµ(x),                                              (162)
                                      i
                                                Z           X                      X
                             =    −h(mθ1 ) −        ((1 −        θ1 (i))p0 (x) +       θ1 (i)pi (x)) log mθ2 (x)dµ(x), (163)
                                                             i                     i
                                              Z
                             =    −h(mθ1 ) − mθ1 (x) log mθ2 (x)dµ(x),                                                (164)
                                  Z
                                                mθ1 (x)
                             =      mθ1 (x) log         dµ(x),                                                        (165)
                                                mθ2 (x)
                             =    DKL [mθ1 : mθ2 ].                                                                   (166)

      Thus we have DKL [mθ1 : mθ2 ] = BF (θ1 : θ2 ). By relaxing the mixture densities mθ1 and mθ2 to
      arbitrary densities m1 and m2 , we find that the dually flat geometry induced by the negentropy of
      densities of a mixture family induces a statistical distance which corresponds to the (forward) KL
      divergence. That is, we have recovered the statistical distance DKL from BF,M . Note that in general
      the entropy of a mixture is not available in closed-form (because of the log sum term), except when the
      component distributions have pairwise disjoint supports. This latter case includes the case of Dirac
      distributions whose mixtures represent the categorical distributions.
   Dually flat spaces can be built from any strictly convex C 3 generator F . Vinberg and Koszul [120] showed
how to obtain such a convex generator for homogeneous cones. A cone C in a vector space V yields a dual
cone of positive linear functionals in the dual vector space V ∗ :
                                      C ∗ := {ω ∈ V ∗ : ∀v ∈ C, ω(v) ≥ 0} .                                           (167)
The characteristic function of the cone is defined by
                                                Z
                                      χC (θ) :=     exp(−ω(θ))dω ≥ 0,                                                 (168)
                                                     C∗

and the function log χC (θ) defines a Bregman generator which induces a Hessian structure and a dually flat
space.
   Figure 10 displays the main types of information manifolds encountered in information geometry with
their relationships.


4     Some applications of information geometry
Information geometry [8] found broad applications in information sciences. For example, we can mention:
    • Statistics: Asymptotic inference, Expectation-Maximization (EM and the novel information-geometric
      em), time series (AutoRegressive Moving Average model, ARMA) models,
    • Machine learning: Restricted Boltzmann machines (RBMs), neuromanifolds and natural gradient [124],
    • Signal processing: Principal Component Analysis (PCA), Independent Component Analysis (ICA),
      Non-negative Matrix Factorization (NMF),
    • Mathematical programming: Barrier function of interior point methods,
    • Game theory: Score functions.
    Next, we shall describe a few applications, starting with the celebrated natural gradient descent.


                                                            31
                                                                        Smooth Manifolds
                                                                                          ∗
               Conjugate Connection Manifolds                                LC
                                                                                  ∇ = ∇+∇
                                                                                        2                              Riemannian Manifolds
                         (M, g, ∇, ∇∗ )
                                                                                                                         (M, g) = (M, g, LC ∇)
                      (M, g, C = Γ∗ − Γ)                                 Self-dual Manifold
                                                     ∇α = 1+α2
                                                                ∇ + 1−α
                                                                     2
                                                                        ∇∗
         (M, g, ∇−α , ∇α )                             ±α        α
                                                     Γ    = Γ̄ ∓ 2 C
         (M, g, αC)
                          Divergence Manifold                                                        Fisher-Riemannian            g = Fisher g
                                               ∗
                        (M, Dg , D ∇, D ∇∗ = D ∇)                                                                                 Fisher
                                                                                                          Manifold                       gij = E[∂i l∂j l]
                          D
                            ∇ − flat ⇔ D ∇∗ − flat
       I[pθ : p 0 ] = D(θ : θ 0 )
               θ
                                                                                                         Multinomial
     Parametric KL∗ on exponential families                                                                family                        Location-scale
      families      KL on mixture families
                                                                                                                                            family
                    Conformal divergences on deformed families
     f -divergences Etc.                        Bregman divergence




32
                                                                                                                       Location
       Expected Manifold                                   canonical
                                                                                                                        family
       (M, Fisher g, ∇−α , ∇α )                            divergence                 Spherical Manifold                               Hyperbolic Manifold
            α-geometry                         Dually flat Manifolds
                                                     (M, F, F ∗ )
                                                 (Hessian Manifolds)                Euclidean Manifold
          Cubic skewness tensor                Dual Legendre potentials
          Cijk = E[∂i l∂j l∂k l]            Bregman Pythagorean theorem
          αC = ∇αFisher g

                     Distance = Non-metric divergence                                                        Distance = Metric geodesic length

          Frank Nielsen



                         Figure 10: Overview of the main types of information manifolds with their relationships in information geometry.
4.1     Natural gradient in Riemannian space
The Natural Gradient [6] (NG) is an extension of the ordinary (Cartesian) gradient of Euclidean geometry
to the gradient in a Riemannian space analyzed in an arbitrary coordinate system. We explain the natural
gradient

4.1.1   The vanilla gradient descent method
Given a real-valued function Lθ (θ) parameterized by a a D-dimensional vector θ on parameter space θ ∈
Θ ⊂ RD , we wish to minimize Lθ , i.e., solve minθ∈Θ Lθ (θ). The gradient descent (GD) method, also called
the steepest descent method, is a first-order local optimization procedure which starts by initializing the
parameter to an arbitrary value (say, θ0 ∈ Θ), and then iteratively updates at stage t the current location
of θt to θt+1 as follows:
                                      GD : θt+1 = θt − αt ∇θ Lθ (θt ).                                (169)
    The scalar αt > 0 is called the step size or learning rate in machine learning. The ordinary gradient
(OG) ∇θ Fθ (θ) (vector of partial derivatives) represents the steepest vector at θ of the function graph Lθ =
{(θ, Lθ (θ)) : θ ∈ Θ}. The GD method was pioneered by Cauchy [28] (1847) and its convergence proof to a
stationary point was first reported in Curry [35] (1944).
    If we reparameterize the function Lθ using a one-to-one and onto differentiable mapping η = η(θ) (with
reciprocal inverse mapping θ = θ(η)), the GD update rule transforms as:

                                           ηt+1 = ηt − αt ∇η Lη (ηt ),                                  (170)

where
                                              Lη (η) := Lθ (θ(η)).                                      (171)
    Thus in general, the two gradient descent location sequences {θt }t and {ηt }t (initialized at θ0 = θ(η0 )
and η0 = η(θ0 )) are different (because usually η(θ) 6= θ), and the two GDs may potentially reach different
stationary points. In other words, the GD local optimization depends on the choice of the parameterization
of the function L (i.e., Lθ or Lη ). For example, minimizing with the gradient descent a temperature function
Lθ (θ) with respect to Celsius degrees θ may yield a different result than minimizing the same temperature
function Lη (η) = Lθ (θ(η)) expressed with respect to Fahrenheit degrees η. That is, the GD optimization
is extrinsic since it depends on the choice of the parameterization of the function, and does not take into
account the underlying geometry of the parameter space Θ.
    The natural gradient precisely addresses this problem and solves it by choosing intrinsically the steepest
direction with respect to a Riemannian metric tensor field on the parameter manifold. We shall explain the
natural gradient descent method and highlight its connections with the Riemannian gradient descent, the
mirror descent and even the ordinary gradient descent when the parameter space is dually flat.

4.1.2   Natural gradient and its connection with the Riemannian gradient
Let (M, g) be a D-dimensional Riemannian space [38] equipped with a metric tensor g, and L ∈ C ∞ (M )
a smooth function to minimize on the manifold M . The Riemannian gradient [21] uses the Riemannian
exponential map expp : Tp → M to update the sequence of points pt ’s on the manifold as follows:

                                    RG :     pt+1 = exppt (−αt ∇M L(pt )),                              (172)

where the Riemannian gradient ∇M is defined according to a directional derivative ∇v by:
                                                             
                                ∇M L(p) := ∇v L expp (v) v=0 ,                                          (173)

with
                                                      L(p + hv) − L(p)
                                     ∇v L(p) := lim                    .                                (174)
                                                  h→0        h


                                                       33
   However, the Riemannian exponential mapping expp (·) is often computationally intractable since it re-
quires to solve a system of second-order differential equations [38, 1]. Thus instead of using expp , we shall
rather use a computable Euclidean retraction R : Tp → RD of the exponential map expressed in a local
θ-coordinate system as:
                                   RetG : θt+1 = Rθt (−αt ∇θ Lθ (θt )) .                                (175)
   Using the retraction [1] Rp (v) = p + v which corresponds to a first-order Taylor approximation of the
exponential map, we recover the natural gradient descent [6]:

                                     NG : θt+1 = θt − αt gθ−1 (θt )∇θ Lθ (θt ).                              (176)

   The natural gradient [6] (NG)
                                          NG
                                               ∇Lθ (θ) := gθ−1 (θ)∇θ Lθ (θ)                                  (177)
encodes the Riemannian steepest descent vector, and the natural gradient descent method yields the following
update rule
                                     NG : θt+1 = θt − αt NG ∇Lθ (θt ).                                  (178)
    Notice that the natural gradient is a contravariant vector while the ordinary gradient is a covariant
vector. Recall that a covariant vector [vi ] is transformed into a contravariant vector [v i ] by v i = j g ij vi ,
                                                                                                       P

that is by using the dual Riemannian metric gη∗ (η) = gθ (θ)−1 . The natural gradient is invariant under an
invertible smooth change of parameterization. However, the natural gradient descent does not guarantee
that the locations θt ’s always stay on the manifold: Indeed, it may happen that for some t, θt 6∈ Θ when
Θ 6= RD .
Property 5 ([21]). The natural gradient descent approximates the intrinsic Riemannian gradient descent
using a contravariant gradient vector induced by the Riemannian metric tensor g. The natural gradient is
invariant to coordinate transformations.
   Next, we shall explain how the natural gradient descent is related to the mirror descent and the ordinary
gradient when the Riemannian space Θ is dually flat.

4.1.3   Natural gradient in dually flat spaces: Connections to Bregman mirror descent and
        ordinary gradient
Recall that a dually flat space (M, g, ∇, ∇∗ ) is a manifold M equipped with a pair (∇, ∇∗ ) of dual torsion-free
                                                                                                      ∗
flat connections which are coupled to the Riemannian metric tensor g [8, 77] in the sense that ∇+∇  2   = LC ∇,
        LC
where      ∇ denotes the unique metric torsion-free Levi-Civita connection.
    On a dually flat space, there exists a pair of dual global Hessian structures [120] with dual canonical
Bregman divergences [23, 8]. The dual Riemannian metrics can be expressed as the Hessians of dual convex
potential functions F and F ∗ . Examples of Hessian manifolds are the manifolds of exponential families or
the manifolds of mixture families [86]. On a dually flat space induced by a strictly convex and C 3 function F
(Bregman generator), we have two dual global coordinate system: θ(η) = ∇F ∗ (η) and η(θ) = ∇F (θ), where
F ∗ denotes the Legendre-Fenchel convex conjugate function [70, 71]. The Hessian metric expressed in the
primal θ-coordinate system is gθ (θ) = ∇2 F (θ), and the dual Hessian metric expressed in the dual coordinate
system is gη∗ (η) = ∇2 F ∗ (η). Crouzeix’s identity [32] shows that gθ (θ)gη (η) = I, where I denotes the D × D
matrix identity.
    The ordinary gradient descent method can be extended using a proximity function Φ(·, ·) as follows:
                                                                                   
                                                                         1
                              PGD : θt+1 = arg min hθ, ∇Lθ (θt )i + Φ(θ, θt ) .                            (179)
                                                   θ∈Θ                   αt

When Φ(θ, θt ) = 12 kθ − θt k2 , the PGD update rule becomes the ordinary GD update rule.




                                                         34
  Consider a Bregman divergence [23] BF for the proximity function Φ: Φ(p, q) = BF (p : q). Then the
PGD yields the following mirror descent (MD):
                                                                       
                                                             1
                         MD : θt+1 = arg min hθ, ∇L(θt )i + BF (θ : θt ) .                     (180)
                                           θ∈Θ              αt
   This mirror descent can be interpreted as a natural gradient descent as follows:
Property 6 ([112]). Bregman mirror descent on the Hessian manifold (M, g = ∇2 F (θ)) is equivalent to
natural gradient descent on the dual Hessian manifold (M, g ∗ = ∇2 F (η)), where F is a Bregman generator,
η = ∇F (θ) and θ = ∇F ∗ (η).
   Indeed, the mirror descent rule yields the following natural gradient update rule:

                              NG∗ : ηt+1    = ηt − αt (gη∗ )−1 (ηt )∇η Lθ (θ(ηt )),                    (181)
                                            =    ηt − αt (gη∗ )−1 (ηt )∇η Lη (ηt ),                    (182)

where gη∗ (η) = ∇2 F ∗ (η) = (∇2θ F (θ))−1 and θ(η) = ∇F ∗ (θ).
   The method is called mirror descent [24] because it performs that gradient step in the dual space (ie.,
mirror space) H = {η = ∇F (θ) : θ ∈ Θ}, and thus solves the inconsistency contravariant/covariant type
problem of subtracting a covariant vector from a contravariant vector of the ordinary GD (Eq. 169).
   Let us prove now the following property of the natural gradient in a dually flat space or Bregman
manifold [77]:
Property 7 ([137]). In a dually flat space induced by potential convex function F , the natural gradient
amounts to the ordinary gradient on the dually parameterized function: NG ∇Lθ (θ) = ∇η Lη (η) where η =
∇θ F (θ) and Lη (η) = Lθ (θ(η)).
Proof. Let (M, g, ∇, ∇∗ ) be a dually flat space. We have gθ (θ) = ∇2 F (θ) = ∇θ ∇θ F (θ) = ∇θ η since
η = ∇θ F (θ). The function to minimize can be written either as Lθ (θ) = Lθ (θ(η)) or as Lη (η) = Lη (η(θ)).
Recall the chain rule in the calculus of differentiation:

                               ∇θ Lθ (θ) = ∇θ (Lη (η(θ))) = (∇θ η)(∇η Lη (η)).                         (183)

   Thus we have:
                                 NG
                                      ∇Lθ (θ)   := gθ−1 (θ)∇θ Lθ (θ),                                  (184)
                                                =     (∇θ η)−1 (∇θ η)∇η Lη (η),                        (185)
                                                =     ∇η Lη (η).                                       (186)



   It follows that the natural gradient descent on a loss function Lθ (θ) amounts to an ordinary gradient
descent on the dually parameterized loss function Lη (η) := Lθ (θ(η)). In short, NG ∇θ Lθ = ∇η Lη .

4.1.4   An application of the natural gradient: Natural Evolution Strategies (NESs)
A nice family of applications of the natural gradient are the Natural Evolution Strategies (NESs) for black-
box minimization [19]: Let f (x) for x ∈ X ⊂ Rd be a real-valued function to minimize. Berny [18] proposed to
relax the optimization problem minx∈X f (x) by considering a parametric search distribution pλ , and minimize
instead:
                                                min Epλ [f (x)],                                        (187)
                                                λ∈Λ
                   D
where λ ∈ Λ ⊂ R denotes the parameter space of the search distributions. Let J(λ) = Epλ [f (x)]. Min-
imizing J(λ) instead of f (x) is particularly useful when X is a discrete space: Indeed, the combinatorial


                                                       35
                                   p0 (x)                                 p1 (x)



                                                                                    x
                                                    x1        x2


Figure 11: Statistical Bayesian hypothesis testing: The best Maximum A Posteriori (MAP) rule chooses to
classify an observation from the class that yields the maximum likelihood.


optimization [18] minx∈X f (x) is replaced by a continuous optimization minλ∈Λ J(λ) when Λ is a continuous
parameter, and the ordinary or natural GD methods can be used. The gradient ∇J(λ) is called the search
gradient, and it can be approximated stochastically using the log-likelihood trick [135] as
                                                n
                                 ∼           1X
                                 ∇ J(λ) :=         f (xi )∇ log pλ (xi ) ≈ ∇J(λ),                          (188)
                                             n i=1

where x1 , . . . , xn ∼ pλ . Similarly, the Fisher information matrix (FIM) may be approximated by the following
empirical FIM:
                                                  n
                                       ˜       1X
                                       I(λ) =        ∇λ lλ (xi )(∇λ lλ (xi ))> ≈ I(λ),                      (189)
                                               n i=1

where lλ (x) := log pλ (x) denote the log-likelihood function. Notice that the approximated FIM may
potentially be degenerated and may not respect the structure of the true FIM. For example, we have
                                 (x−µ)2                                                        ˜
∇µ l(x; µ, σ 2 ) = x−µ
                    σ 2 and ∇σ =
                              2
                                   2σ 4 − 2σ1 2 . The non-diagonal of the approximate  FIM I(λ) are close to
                                                                  2             1  1
but usually non-zero although the expected FIM is diagonal I(µ, σ ) = diag σ2 , 2σ4 . Thus we may estimate
the FIM until the non-diagonal elements have absolute values less than a prescribed  > 0. For multivariate
normals, we have ∇µ l(x; µ, Σ) = Σ−1 (x − µ) and ∇Σ l(x; µ, Σ) = 12 (∇µ l(x; µ, Σ)∇µ l(x; µ, Σ)> − Σ−1 ).

4.2    Some illustrating applications of dually flat manifolds
In this part, we describe how to use the dually flat structures for handling an exponential family E (in a
hypothesis testing problem detailed in §4.3) and the mixture family M (clustering statistical mixtures §4.4).
Note that for a general divergence, neither (E, D) nor (M, D) is dually flat. However, when D = KL,
the Kullback-Leibler divergence, we get dually flat spaces that are computationally attractive since the
primal/dual geodesics are straight lines in the corresponding global affine coordinate system.

4.3    Hypothesis testing in the dually flat exponential family manifold (E, KL∗ )
Given two probability distributions P0 ∼ p0 (x) and P1 ∼ p1 (x), we ask to classify a set of iid. observations
X1:n = {x1 , . . . , xn } as either sampled from P0 or from P1 ? This is a statistical decision problem [73].
For example, P0 can represent the signal distribution and P1 the noise distribution. Figure 11 displays
the probability distributions and the unavoidable error that is made by any statistical decision rule (on
observations x1 and x2 ).
   Assume that both distributions P0 ∼ Pθ0 and P1 ∼ Pθ1 belong to the same exponential family E =
{Pθ : θ ∈ Θ}, and consider the exponential family manifold with the dually flat structure (E, E g, E ∇e , E ∇m ).
That is, the manifold equipped with the Fisher information metric tensor field and the expected exponential
connection and conjugate expected mixture connection. More generally, the expected α-geometry of an



                                                         36
exponential family E with cumulant function F is given by:

                                        gij (θ)    = ∂i ∂j F (θ),                                       (190)
                                                     1−α
                                         Γα
                                          ij,k     =         ∂i ∂j ∂k F (θ).                            (191)
                                                        2
                                                        −1
When α = 1, Γα                1
                ij,k = 0 and ∇ is flat, and so is ∇        by using the fundamental theorem of information
geometry.
   The ±1-structure can also be derived from a divergence manifold structure by choosing the reverse
Kullback-Leibler divergence KL∗ :
                                     (E, E g, E ∇e , E ∇m ) ≡ (E, KL∗ ).                             (192)
   Therefore, the Kullback-Leibler divergence KL[Pθ : Pθ0 ] amounts to a Bregman divergence (for the
cumulant function of the exponential family):

                                  KL∗ [Pθ0 : Pθ ] = KL[Pθ : Pθ0 ] = BF (θ0 : θ).                        (193)
   The best exponent error α∗ of the best Maximum A Priori (MAP) decision rule is found by minimizing
the Bhattacharyya distance to get the Chernoff information [106]:
                                                  Z
                                                              1−α
                          C[P1 , P2 ] = − log min      pα
                                                        1 (x)p2   (x)dµ(x) ≥ 0.                 (194)
                                                 α∈(0,1)   x∈X

   On the exponential family manifold E, the Bhattacharyya distance:
                                                    Z
                                                             1−α
                              Bα [p1 : p2 ] = − log   pα
                                                       1 (x)p2   (x)dµ(x),                              (195)
                                                           x∈X

amounts to a skew Jensen parameter divergence [83] (also called Burbea-Rao divergence):

                         JFα (θ1 : θ2 ) = αF (θ1 ) + (1 − α)F (θ2 ) − F (θ1 + (1 − α)θ2 ).              (196)

  It can be shown that the Chernoff information (that minimizes α) is equivalent to a Bregman divergence:
Namely, the Bregman divergence for exponential families at the optimal exponent value α∗ .
Theorem 10 (Chernoff information [73]). The Chernoff information between two distributions belonging to
the same exponential family amount to a Bregman divergence:
                                                                 ∗             ∗
                                                          α              α
                                  C[Pθ1 : Pθ2 ] = B(θ1 : θ12 ) = B(θ2 : θ12 ),                          (197)
       α
where θ12 = (1 − α)θ1 + αθ2 , and α∗ denote the best exponent error.
                 ∗
         ∗      α
   Let θ12 := θ12  denote the best exponent error. The geometry [73] of the best error exponent can be
explained on the dually flat exponential family manifold as follows:

                                   P ∗ = Pθ12
                                           ∗ = Ge (P1 , P2 ) ∩ Bim (P1 , P2 ),                          (198)

where Ge denotes the exponential geodesic γ∇e and Bim the m-bisector:

                        Bim (P1 , P2 ) = {P : F (θ1 ) − F (θ2 ) + η(P )> (θ2 − θ1 ) = 0}.               (199)

    Figure 12 illustrates how to retrieve the best error exponent from an exponential arc (θ-geodesic) inter-
secting the m-bisector.
    Furthermore, instead of considering two distributions for this statistical binary decision problem, we may
consider a set of n distributions of P1 , . . . , Pn ∈ E. The geometry of the error exponent in this multiple
hypothesis testing setting has been investigated in [72]. On the dually flat exponential family manifold, it
corresponds to check the exponential arcs between natural neighbors (sharing Voronoi subfaces) of a Bregman
Voronoi diagram [20]. See Figure 13 for an illustration.


                                                           37
                                      m-bisector               η-coordinate system
                                       Bim (Pθ1 , Pθ2 )


                                    pθ12
                                      ∗  e-geodesic Ge (Pθ1 , Pθ2 )
                                          Pθ ∗      pθ2
                                             12




                              pθ1
                                                       ∗
                                 C(θ1 : θ2 ) = B(θ1 : θ12 )


Figure 12: Exact geometric characterization (not necessarily i closed-form) of the best exponent error rate
α∗ .




                      η-coordinate system




                                                    Chernoff distribution between
                                                    natural neighbours

Figure 13: Geometric characterization of the best exponent error rate in the multiple hypothesis testing
case.




                                                          38
                          0.5
                                                                                         M1
                                                                                         M2
                         0.45                                                 Gaussian(-2,1)
                                                                                Cauchy(2,1)
                                                                               Laplace(0,1)
                          0.4


                         0.35


                          0.3


                         0.25


                          0.2


                         0.15


                          0.1


                         0.05


                           0
                                -4         -2       0         2        4           6             8




Figure 14: Example of a mixture family of order D = 2 (3 components: Laplacian, Gaussian and Cauchy
prefixed distributions).


4.4    Clustering mixtures in the dually flat mixture family manifold (M, KL)
Given a set of k prescribed statistical distributions p0 (x), . . . , pk−1 (x), all sharing the same support X (say,
R), a mixture family M of order D = k − 1 consists of all strictly convex combinations of these component
distributions [93]:
                    (           k−1                  k−1
                                                            !                               k−1
                                                                                                      )
                                X                    X                                      X
              M := m(x; θ) =         θi pi (x) + 1 −      θi p0 (x) such that θi > 0,           θi < 1 .      (200)
                                     i=1                i=1                                    i=1

    Figure 14 displays two mixtures obtained as convex combinations of prescribed Laplacian, Gaussian
and Cauchy component distributions (D = 2). When considering a set of prescribed Gaussian component
distributions, we obtain a w-Gaussian Mixture Model, or w-GMM for short.
    We consider the expected information manifold (M, M g, M ∇m , M ∇e ) which is dually flat and equivalent
to (MΘ , KL). That is, the KL between two mixtures with prescribed components
                                                                     R            (w-mixtures, for short) is
equivalent to a Bregman divergence for F (θ) = −h(mθ ), where h(p) = p(x) log p(x)dµ(x) is the differential
Shannon information (negative entropy) [93]:

                                           KL[mθ1 : mθ2 ] = BF (θ1 : θ2 ).                                    (201)
    Consider a set {mθ1 , . . . , mθn } of n w-mixtures [93]. Because F (θ) = −h(m(x; θ)) is the negative differ-
ential entropy of a mixture (not available in closed form [95]), we approximate the untractable F by another
close tractable generator F̃ . We use Monte Carlo stochastic sampling to get Monte-Carlo convex F̃S for an
independent and identically distributed sample S.
    Thus we can build a nested sequence (M, F̃S1 ), . . . , (M, F̃Sm ) of tractable dually flat manifolds for nested
sample sets S1 ⊂ . . . ⊂ Sm converging to the ideal mixture manifold (M, F ): limm→∞ (M, F̃Sm ) = (M, F )
(where convergence is defined with respect to the induced canonical Bregman divergence). A key advantage
of this approach is that for a given sample S, all computations carried inside the dually flat manifold (M, F̃S )
are consistent, see [93].
    For example, we can apply Bregman k-means [87] on these Monte Carlo dually flat spaces [85] of w-GMMs
(Gaussian Mixture Models) to cluster a set of w-GMMs. Figure 15 displays the result of such a clustering.
    We have briefly described two applications using dually flat manifolds: (1) the dually flat exponential
manifold induced by the statistical reverse Kullback-Leibler divergence on an exponential family (structure
(E, KL∗ )), and (2) the dually flat mixture manifold induced by the statistical Kullback-Leibler divergence


                                                         39
                        0.25




                         0.2




                        0.15




                         0.1




                        0.05




                          0
                                -4          -2             0        2           4




                      Figure 15: Example of w-GMM clustering into k = 2 clusters.


on a mixture family (structure (M, KL)). There are many other dually flat structures that can be met in a
statistical context: For example, two other dually flat structures for the D-dimensional probability simplex
∆D are reported in Amari’s textbook [8]: (1) the conformally deforming of the α-geometry (page 88, Eq.
4.95 of [8]), and (2) the χ-escort geometry (page 91, Eq. 4.114 of [8]).


5     Conclusion: Summary, historical background, and perspectives
5.1    Summary
We explained the dualistic nature of information manifolds (M, g, ∇, ∇∗ ) in information geometry. The
dualistic structure is defined by a pair of conjugate connections coupled with the metric tensor that provides
a dual parallel transport that preserves the metric. We showed how to extend this structure to a 1-parameter
family of structures: From a pair of conjugate connections, the pipeline to build this 1-parameter family of
structures can be informally summarized as:

                  (M, g, ∇, ∇∗ ) ⇒ (M, g, C) ⇒ (M, g, αC) ⇒ (M, g, ∇−α , ∇α ),      ∀α ∈ R.              (202)

     We stated the fundamental theorem of information geometry on dual constant-curvature manifolds,
including the special but important case of dually flat manifolds on which there exists two potential functions
and global affine coordinate systems related by the Legendre-Fenchel transformation. Although, information
geometry historically started with the Riemannian modeling (P, P g) of a parametric family of probability
distributions P by letting the metric tensor be the Fisher information matrix, we have emphasized the
dualistic view of information geometry which considers non-Riemannian manifolds that can be derived from
any divergence, and not necessarily tied to a statistical context (e.g., information manifold can be used in
mathematical programming [101]). Let us notice that for any symmetric divergence (e.g. any symmetrized
f -divergence like the squared Hellinger divergence), the induced conjugate connections coincide with the
Levi-Civita connection but the Fisher-Rao metric distance does not coincide with the squared Hellinger
divergence.
     On one hand, a Riemannian metric distance Dρ is never a divergence because the rooted distance functions
fail to be smooth at the extremities but a squared Riemmanian metric distance is always a divergence. On the
other hand, taking the power δ of a divergence D (i.e., Dδ ) for some δ > 0 may yield a metric distance (e.g.,
the square root of the Jensen-Shannon divergence [44]), but this may not always be the case: The powered
Jeffreys divergence J δ is never a metric distance (see [127], page 889). Recently, the Optimal Transport


                                                      40
(OT) theory [130] gained interest in statistics and machine learning. But the optimal transport between two
members of a same elliptically-contoured family has the same optimal transport formula distance (see [40]
Eq. 16 and Eq. 17, although they have different Kullback-Leibler divergences). Another essential difference
is that the Fisher-Rao manifold of location-scale families is hyperbolic but the Wasserstein manifold of
location-scale families has positive curvature [40, 125].
    Notice that we may convert back and forth a similarity S(p, q) ∈ (0, 1] to a dissimilarity D(p, q) ∈ [0, ∞)
as follows:
                                       S(p, q)   =   exp(−D(p, q)) ∈ (0, 1]                                  (203)
                                      D(p, q)    =   − log S(p, q) ∈ [0, ∞)                                  (204)
When the dissimilarity satisfies the (additive) triangle inequality (i.e., D(p, q)+D(q, r) ≥ D(p, r) for any triple
(p, q, r)) then the corresponding similarity satisfies the multiplicative triangle inequality: S(p, q) × S(q, r) ≤
S(p, r). A metric transform on a metric distance D is a transformation T such that T (D(p, q)) is a metric.
                               1
The transformation T (u) = 1+u    is a metric transform which bounds potentially unbounded metric distances:
                                                           D(p,q)
That is, if D is an unbounded metric, then T (D(p, q)) = 1+D(p,q)  is a bounded metric distance. The
                       2
transformation S(u) = u is not a metric transform since the squared of the Euclidean metric distance is
not a metric distance.

5.2    A brief historical review of information geometry
The field of Information Geometry (IG) was historically motivated by providing some differential-geometric
structures to statistical models in order to reason geometrically about statistical problems with the endeavor
goal of geometrizing mathematical statistics [29, 5, 39, 67, 54, 58, 9, 47]: Professor Harold Hotelling [53]
first considered in the late 1920’s the Fisher Information Matrix (FIM) I as a Riemannian metric tensor g
(ie., the Fisher Information metric, FIm), and interpreted a parametric family of probability distributions
M as a Riemannian manifold (M, g). Historically speaking, Hotelling attended the American Mathemati-
cal Society’s Annual Meeting in Bethlehem (Pennsylvania, USA) on December 26–29, 1929, but left before
his scheduled talk on December 27. His handwritten notes on the “Spaces of Statistical Parameters” was
read by a colleague and are fully typeset in [123]. We warmly thank Professor Stigler for sending us the
scanned handwritten notes and for discussing by emails some historical aspects of the birth of information
geometry. In this pioneering work, Hotelling mentioned that location-scale probability families yield Rie-
mannian manifolds of constant non-positive curvatures. This Riemannian modeling of parametric family of
densities was further independently studied by Calyampudi Radhakrishna Rao (C.R. Rao) in his celebrated
paper [111] (1945) that also includes the Cramér-Rao lower bound [71] and the Rao-Blackwellization tech-
nique used in statistics. Nowadays the induced Riemannian metric distance is often called the Fisher-Rao
distance [122] or Rao distance [113]. Yet another use of Riemannian geometry in statistics was pioneered by
Harold Jeffreys [55] that proposed to use as an invariant prior the normalized volume element of the Fisher-
Riemannian manifold. In those seminal papers, there was no theoretical justification of using the Fisher
information matrix as a metric tensor (besides the fact that it is a well-defined positive-definite matrix for
regular identifiable models). Nowadays, this Riemmanian metric tensor is called the information metric for
short. Modern information geometry considers a generalization of this approach using a non-Riemannian
dualistic modeling (M, g, ∇, ∇∗ ) which coincides with the Riemannian manifold when ∇ = ∇∗ = LC ∇, the
Levi-Civita connection (the unique torsion-free affine connection compatible with the metric tensor). The
Fisher-Rao geometry has also been explored in thermodynamics yielding the Ruppeiner geometry [134], and
the geometry of thermodynamics is called nowadays called geometrothermodynamics [109].
    In the 1960’s, Nikolai Chentsov (also commonly written Čencov) studied the algebraic category of all
statistical decision rules with its induced geometric structures: Namely, the α-geometries (“equivalent differ-
ential geometry”) and the dually flat manifolds (“Nonsymmetric Pythagorean geometry” of the exponential
families with respect to the Kullback-Leibler divergence). In the preface of the english translation of his
1972’s russian monograph [29], the field of investigation is defined as “geometrical statistics.” However in
the original Russian monograph, Chentsov used the russian term geometrostatistics. According to Professor


                                                        41
Alexander Holevo, the geometrostatistics term was coined by Andrey Kolmogorov to define the field of dif-
ferential geometry of statistical models. In the monograph of Chentsov [29], the Fisher information metric
is shown to be the unique metric tensor (up to a scaling factor) yielding statistical invariance under Markov
morphisms (see [26] for a simpler proof that generalizes to positive measures).
    The dual nature of the information geometry was thoroughly investigated by Professor Shun-ichi Amari [4].
In the preface of his 1985’s monograph [5], Professor Amari coined the term information geometry as follows:
“The differential-geometrical method developed in statistics is also applicable to other fields of sciences such
as information theory and systems theory... They together will open a new field, which I would like to
call information geometry.” Professor Amari mentioned in [5] that he considered the Gaussian Riemannian
manifold as a hyperbolic manifold in 1959, and was strongly influenced by Efron’s paper on statistical curva-
ture [41] (1975) to study the family of α-connections in the 1980’s [4, 68]. Professor Amari prepared his PhD
under the supervision of Professor Kondo [31], an expert of differential geometry in touch with Professor
Kawaguchi [59]. The role of differential geometry in statistics has been discussed in [17].
    Note that the dual affine connections of information geometry have also been investigated independently
in affine differential geometry [99] which considers invariance under volume-preserving affine transformations
by defining a volume form instead of a metric form for Riemannian geometry. The notion of dual parallel
transport compatible with the metric is due to Aleksandr Norden [100] and Rabindra Nath Sen [116, 117, 118]
(See the Senian geometry in http://insaindia.res.in/detail/N54-0728).
    We summarize the main fundamental structures of information manifolds below:
 (M, g)                    Riemannian manifold
 (P, P g)                  Fisher-Riemannian (expected) Riemannian manifold
 (M, g, ∇)                 Riemannian manifold (M, g) with affine connection ∇
 (P, P g, P e ∇α )         Chentsov’s manifold with affine exponential α-connection
 (M, g, ∇, ∇∗ )            Amari’s dualistic information manifold
 (P, P g, P ∇−α , P ∇α )   Amari’s (expected) information α-manifold, α-geometry
 (M, g, C)                 Lauritzen’s statistical manifold [62]
                 ∗
 (M, D g, D ∇, D ∇)        Eguchi’s conjugate connection manifold induced by divergence D
     F    F
 (M, g, C)                 Chentsov/Amari’s dually flat manifold induced by convex potential F

  We use the ≡ symbol to denote the equivalence of geometric structures. For example, we have (M, g) ≡
                 ∗
(M, g, LC ∇, LC ∇ = LC ∇).

5.3    Perspectives
We recommend the two recent textbooks [25, 8] for an indepth covering of (parametric) information geometry,
and the book [48] for a thorough description of some infinite-dimensional statistical models. (Japanese
readers may refer to [7, 45].) We did not report the various coefficients of the metric tensors, Christoffel
symbols and skewness tensors for the expected α-geometry of common parametric models like the multivariate
Gaussian distributions, the Gamma/Beta distributions, etc. They can be found in [10, 25] and in various
articles dealing with less common family of distributions [64, 65, 141, 105, 142, 104, 10]. Although we have
focused on the finite parametric setting, information geometry is also considering non-parametric families of
distributions [107], and quantum information geometry [51].
    We have shown that we can always create an information manifold (M, D) from any divergence function
D. It is therefore important to consider generic classes of divergences in applications, that are ideally
axiomatized and shown to have exhaustive characteristics. Beyond the three main Bregman/Csiszár/Jensen
classes (theses classes overlap [102]), we may also mention the class of conformal divergences [98, 91], the
class of projective divergences [92, 97], etc. Figure 16 illustrates the relationships between the principal
classes of distances.
    There are many perspectives on information geometry as attested by the new Springer journal13 , and the
biannual international conference “Geometric Sciences of Information” (GSI) [80, 81, 82] with its collective
 13 ’Information Geometry’, https://www.springer.com/mathematics/geometry/journal/41884




                                                       42
                                             Dissimilarity measure


                                           C3     Divergence


                                                 v-Divergence Dv                           Projective divergence
                                                                                    double sided
                                                                                                              one-sided
                                      scaled conformal divergence CD,g (· : ·; ·)          γ-divergence
                                                                                     D(λp : λ0 p0 ) = D(p : p0 )
                                                                                                            Hÿvarinen SM/RM
                                                                                                           D(λp : p0 ) = D(p : p0 )


             conformal divergence CD,g (· : ·)                             scaled Bregman divergence BF (· : ·; ·)




   total Bregman divergence tB(· : ·)                  Bregman divergence BF (· : ·)             Csiszár f -divergence If (· : ·)



Dv (P : Q) = D(v(P ) : v(Q))                                   tBF (P : Q) = √ BF (P :Q)
                                                                               1+k∇F (Q)k2
                           
            R          q(x)
If (P : Q) = p(x)f ( p(x)     dν(x)                             CD,g (P : Q) = g(Q)D(P : Q)
                                                                                             
                                                                                         P  Q
BF (P : Q) = F (P ) − F (Q) − hP − Q, ∇F (Q)i                   BF,g (P : Q; W ) = W BF Q  :W



                                  Figure 16: Principled classes of distances/divergences




                                                                 43
post-conference edited books [75, 74]. We also mention the edited book [13] on the Occasion of Shun-ichi
Amari’s 80th birthday.
Acknowledgments:FN would like to thank the organizers of the Geometry In Machine Learning workshop in
2018 (GiMLi, http://gimli.cc/2018/) for their kind keynote talk invitation, and specially Professor Søren
Hauberg (Technical University of Denmark, DTU). This survey is based on the talk given at GiMLi. I am
very thankful to Professor Stigler (University of Chicago, USA) and Professor Holevo (Steklov Mathematical
Institute, Russia) for providing me feedback on some historical aspects of the field of information geometry.
Finally, I would like to express my sincere thanks to Gaëtan Hadjeres (Sony Computer Science Laboratories
Inc, Paris) for his careful proofreading and feedback.


A     Monte Carlo estimations of f -divergences
Let (X, F, µ) be a probability space [60] with X denoting the sample space, F the σ-algebra, and µ a
reference positive measure. The f -divergence [33, 90] between two probability measures P and Q both
absolutely continuous with respect to a positive measure µ for a convex generator f : (0, ∞) → R strictly
convex at 1 and satisfying f (1) = 0 is
                                                        Z            
                                                                 q(x)
                               If (P : Q) = If (p : q) = p(x)f          dµ(x),                     (205)
                                                                 p(x)
where P = pdµ and Q = qdµ (i.e., p and q denote the Radon-Nikodym derivatives with respect to µ). We
use the following conventions:
                 
                  0                                     a          a           f (u)
             0f      = 0, f (0) = lim f (u), ∀a > 0, 0f     = lim uf      = a lim        .      (206)
                  0              u→0+                    0   u→0+     u       u→∞ u

    When f (u) = − log u, we retrieve the Kullback-Leibler divergence (KLD):
                                                 Z
                                                             p(x)
                                    DKL (p : q) = p(x) log        dµ(x).                               (207)
                                                             q(x)
    The KLD is usually difficult to calculate in closed-form, say, for example, between statistical mixture
models [96]. A common technique is to estimate the KLD using Monte Carlo sampling using a proposal
distribution r:
                                                      n
                                                  1 X p(xi )       p(xi )
                                    KLn (p : q) =
                                    c                          log        ,                           (208)
                                                  n i=1 r(xi )     q(xi )
where x1 , . . . , xn ∼iid r. When r is chosen as p, the KLD can be estimated as:
                                                         n
                                        c n (p : q) = 1         p(xi )
                                                        X
                                        KL                  log        .                               (209)
                                                      n i=1     q(xi )

Monte Carlo estimators are consistent under mild conditions: limn→∞ KL c n (p : q) = KL(p : q).
    In practice, one problem when implementing Eq. 209, is that we may end up potentially with KL
                                                                                                c n (p : q) <
0. This may have disastrous consequences as algorithms implemented by programs consider non-negative
divergences
          Pto execute a correct
                             P workflow. The potential negative value problem of Eq. 209 comes from the
fact that i p(xi ) 6= 1 and i q(xi ) 6= 1. One way to circumvent this problem is to consider the extended
f -divergences:
Definition 7 (Extended f -divergence). The extended f -divergence for a convex generator f , strictly convex
at 1 and satisfying f (1) = 0 is defined by
                                      Z                             
                           e                    q(x)      0      q(x)
                          If (p : q) = p(x) f          − f (1)        −1   dµ(x).                      (210)
                                                p(x)             p(x)

                                                      44
   Indeed, for a strictly convex generator f , let us consider the scalar Bregman divergence [23]:
                                  Bf (a : b) = f (a) − f (b) − (a − b)f 0 (b) ≥ 0.                        (211)
               q(x)
   Setting a = p(x) and b = 1 in Eq. 211, and using the fact that f (1) = 0, we get
                                                        
                                       q(x)       q(x)
                                   f          −        − 1 f 0 (1) ≥ 0.                                   (212)
                                       p(x)       p(x)
   Therefore we define the extended f -divergences as
                                            Z                  
                                                        q(x)
                               Ife (p : q) = p(x)Bf          : 1 dµ(x) ≥ 0.                               (213)
                                                        p(x)
   That is, the formula for the extended f -divergences is
                                  Z                               
                       e                      q(x)       0     q(x)
                      If (p : q) = p(x) f            − f (1)        −1   dµ(x) ≥ 0.                       (214)
                                              p(x)             p(x)
    Then we estimate the extended f -divergence using importance sampling of the integral with respect to
distribution r, using n variates x1 , . . . , xn ∼iid p as:
                                              n                                  
                                           1X        q(xi )               q(xi )
                           Iˆf,n (p : q) =       f            − f 0 (1)          − 1 ≥ 0.                 (215)
                                           n i=1     p(xi )               p(xi )
   For example, for the KLD, we obtain the following Monte Carlo estimator:
                                            n                        
                           c n (p : q) = 1           p(xi ) q(xi )
                                           X
                           KL                    log       +       − 1 ≥ 0,                               (216)
                                         n i=1       q(xi ) p(xi )

since the extended KLD is
                                             Z                            
                                                         p(x)
                            DKLe (p : q) =      p(x) log      + q(x) − p(x) dµ(x).                        (217)
                                                         q(x)
Eq. 216 can be interpreted as a Psum of scalar Itakura-Saito divergences since the Itakura-Saito divergence is
                 c n (p : q) = 1 n DIS (p(xi ) : q(xi )) with the scalar Itakura-Saito divergence
scale-invariant: KL            n   i=1
                                                   a  a           a
                                 DIS (a : b) = DIS    : 1 = − log − 1 ≥ 0,                              (218)
                                                    b       b       b
a Bregman divergence obtained for the generator f (u) = − log u.
   Notice that the extended f -divergence is a f -divergence for the generator
                                          fe (u) = f (u) − f 0 (1)(u − 1).                                (219)
We check that the generator fe satisfies both f (1) = 0 and f 0 (1) = 0, and we have Ife (p : q) = Ife (p : q).
                                        e
Thus DKLe (p : q) = IfKL               KL (u) = − log u + u − 1.
                       e (p : q) with f

    Let us remark that we only need to have the scalar function strictly convex at 1 to ensure that Bf ab : 1 ≥
                                                                                                             

0. Indeed, we may use the definition of Bregman divergences extended to strictly convex functions but not
necessarily smooth functions [50, 126]:
                              Bf (x : y) =     max     {f (x) − f (y) − (x − y)g(y)},                     (220)
                                             g(y)∈∂f (y)

where ∂f (y) denotes the subderivative of f at y.
    Furthermore, noticing that Iλf (p : q) = λIf (p : q) for λ > 0, we may enforce that f 00 (1) = 1, and obtain
a standard f -divergence [8] which enjoys the property that If (pθ (x) : pθ+dθ (x)) = dθ> I(θ)dθ, where I(θ)
denotes the Fisher information matrix of the parameteric family {pθ }θ of densities.


                                                           45
B       The multivariate Gaussian family: An exponential family
We report the canonical decomposition of the multivariate Gaussian [136] family {N (µ, Σ) such that µ ∈
Rd , Σ  0} following [78]. The multivariate Gaussian family is also called the MultiVariate Normal family,
or MVN family for short.
    Let λ := (λv , λM ) = (µ, Σ) denote the composite (vector,matrix) parameter of an MVN. The d-dimensional
MVN density is given by
                                                                                         
                                               1              1            > −1
                         pλ (x; λ) :=        d p      exp   −   (x − λ v )  λM  (x − λ v )  ,         (221)
                                        (2π) 2 |λM |          2

where | · | denotes the matrix determinant. The natural parameters θ are also expressed using both a vector
parameter θv and a matrix parameter θM in a compound object θ = (θv , θM ). By defining the following
compound inner product on a composite (vector,matrix) object
                                                                      
                                                                 0 >
                                       hθ, θ0 i := θv> θv0 + tr θM   θM ,                             (222)

where tr(·) denotes the matrix trace, we rewrite the MVN density of Equation (221) in the canonical form
of an exponential family [84]:

                             pθ (x; θ)    :=       exp (ht(x), θi − Fθ (θ)) = pλ (x; λ(θ)),                  (223)

where                                                                                             
                                                           1                                  1 −1
                       θ = (θv , θM ) =       Σ   −1
                                                       µ, − Σ−1       = θ(λ) =       λ−1
                                                                                      M λv , − λM        ,   (224)
                                                           2                                 2
is the compound natural parameter and
                                                        t(x) = (x, −xx> )                                    (225)
is the compound sufficient statistic. The function Fθ is the strictly convex and continuously differentiable
log-normalizer defined by:                                                
                                         1                        1 > −1
                                Fθ (θ) =     d log π − log |θM | + θv θM θv ,                         (226)
                                         2                        2
    The log-normalizer can be expressed using the ordinary parameters, λ = (µ, Σ), as:
                                                  1 > −1                           
                              Fλ (λ)      =         λv λM λv + log |λM | + d log 2π ,                        (227)
                                                  2
                                                  1 > −1                         
                                          =         µ Σ µ + log |Σ| + d log 2π .                             (228)
                                                  2
    The moment/expectation parameters [8] are

                                         η = (ηv , ηM ) = E[t(x)] = ∇F (θ).                                  (229)

   We report the conversion formula between the three types of coordinate systems (namely the ordinary
parameter λ, the natural parameter θ and the moment parameter η) as follows:

                           θv (λ) = λ−1       −1                      −1
                                                         λv (θ) = 12 θM
                                                     
                                     M λv = Σ    µ
                                                   ⇔
                                                                         θv = µ
                                                                                                 (230)
                                     1 −1   1 −1                    1 −1
                           θM (λ) = 2 λM = 2 Σ           λM (θ) = 2 θM = Σ
                        −1
           ηv (θ) = 21 θM                                θv (η) = −(ηM + ηv ηv> )−1 ηv
                                                     
                           θv
                        1 −1     1 −1      −1      ⇔                                             (231)
           ηM (θ) = − 2 θM − 4 (θM θv )(θM θv )  >       θM (η) = − 21 (ηM + ηv ηv> )−1
                                                     
                       λv (η) = ηv = µ                   ηv (λ) = λv = µ
                                                   ⇔                                             (232)
                       λM (η) = −ηM − ηv ηv> = Σ         ηM (λ) = −λM − λv λ>   v = −Σ − µµ
                                                                                           >




                                                               46
    The dual Legendre convex conjugate [8] is
                                      1              −1
                        Fη∗ (η) = −     log(1 + ηv> ηM
                                                                                           
                                                        ηv ) + log | − ηM | + d(1 + log 2π) ,               (233)
                                      2
and θ = ∇η Fη∗ (η).
   We check the Fenchel-Young equality when η = ∇F (θ) and θ = ∇F ∗ (η):

                                             Fθ (θ) + Fη∗ (η) − hθ, ηi = 0.                                 (234)

   The KullbackLeibler divergence between two d-dimensional Gaussians distributions p(µ1 ,Σ1 ) and p(µ2 ,Σ2 )
(with ∆µ = µ2 − µ1 ) is
                                                                               
                                       1       −1        > −1          |Σ2 |
        KL(p(µ1 ,Σ1 ) : p(µ2 ,Σ2 ) ) =     tr(Σ2 Σ1 ) + ∆µ Σ2 ∆µ + log       − d = KL(pλ1 : pλ2 ).   (235)
                                       2                               |Σ1 |

   We check that KL(p(µ,Σ) : p(µ,Σ) ) = 0 since ∆µ = 0 and tr(Σ−1 Σ) = tr(I) = d. Notice that when
Σ1 = Σ2 = Σ, we have
                                                          1 > −1   1 2
                             KL(p(µ1 ,Σ) : p(µ2 ,Σ) ) =    ∆ Σ ∆µ = DΣ −1 (µ1 , µ2 ),                       (236)
                                                          2 µ      2
that is half the squared Mahalanobis distance for the precision matrix Σ−1 (a positive-definite matrix:
Σ−1  0), where the Mahalanobis distance is defined for any positive matrix Q  0 as follows:
                                               q
                                DQ (p1 : p2 ) = (p1 − p2 )> Q(p1 − p2 ).                         (237)

    The KullbackLeibler divergence between two probability densities of the same exponential families amount
to a Bregman divergence [8]:

                     KL(p(µ1 ,Σ1 ) : p(µ2 ,Σ2 ) ) = KL(pλ1 : pλ2 ) = BF (θ2 : θ1 ) = BF ∗ (η1 : η2 ),       (238)

where the Bregman divergence is defined by

                                  BF (θ : θ0 ) := F (θ) − F (θ0 ) − hθ − θ0 , ∇F (θ0 )i,                    (239)

with η 0 = ∇F (θ0 ). Define the canonical divergence [8]

                            AF (θ1 : η2 ) = F (θ1 ) + F ∗ (η2 ) − hθ1 , η2 i = AF ∗ (η2 : θ1 ),             (240)

since F ∗ ∗ = F . We have BF (θ1 : θ2 ) = AF (θ1 : η2 ).


Notations

Below is a list of notations we used in this document:
 [D]                       [D] := {1, . . . , D}
 h·, ·i                    inner product                            qP
 MQ (u, v) = ku − vkQ                                                           i   i   j   j
                           Mahalanobis distance MQ (u, v) =               i,j (u − v )(u − v )Qij , Q  0
          0
 D(θ : θ )                 parameter divergence
 D[p(x) : p0 (x)]          statistical divergence
 D, D∗                     Divergence and dual (reverse) divergence



                                                            47
                                                               0
                                                    PD         θ
Csiszár divergence If             If (θ : θ0 ) := i=1 θi f θii with f (1) = 0
Bregman divergence BF              BF (θ : θ0 ) := F (θ) − F (θ0 ) − (θ − θ0 )> ∇F (θ0 )
Canonical divergence AF,F ∗        AF,F ∗ (θ : η 0 ) = F (θ)     ∗ 0
                                                                         θ> η0
                                                           R + F α(η ) −1−α
Bhattacharyya distance             Bα [p1 : p2 ] = − log x∈X p1 (x)p2 (x)dµ(x)
                                     (α)
Jensen/Burbea-Rao divergence       JF (θ1 : θ2 ) = αF (θ1 ) + (1 −  R α)F (θ2 ) −1−α
                                                                                   F (θ1 + (1 − α)θ2 )
Chernoff information               C[P1 , P2 ] = − log minα∈(0,1) x∈X pα   1 (x)p 2   (x)dµ(x)
F, F∗                              Potential functions related by Legendre-Fenchel transformation
                                                                          R1
Dρ (p, q)                          Riemannian distance Dρ (p, q) := 0 kγ 0 (t)kγ(t) dt

B, B ∗                             basis, reciprocal basis
B = {e1 = ∂1 , . . . , eD = ∂D }   natural basis
{dxi }i                            covector basis (one-forms)
(v)B := (v i )                     contravariant components of vector v
(v)B ∗ := (vi )                    covariant components of vector v
u⊥v p                              vector u is perpendicular to vector v (hu, vi = 0)
kvk = hv, vi                       induced norm, length of a vector v

M, S                               Manifold, submanifold
Tp                                 tangent plane at p
TM                                 Tangent bundle T M = ∪p Tp = {(p, v), p ∈ M, v ∈ Tp }
F(M )                              space of smooth functions on M
X(M ) = Γ(T M )                    space of smooth vector fields on M
vf                                 direction derivative of f with respect to vector v
X, Y, Z ∈ X(M )                    Vector fields
  Σ
g = gij dxi ⊗ dxj                  metric tensor (field)
(U, x)                             local coordinates x in a chat U
         ∂
∂i :=: ∂x  i
                                   natural basis vector
         ∂
∂ i :=: ∂x i                       natural reciprocal basis vector

∇                                  affine connection
∇X Y                               covariant derivative
Q∇
                                   parallel transport of vectors along a smooth curve c
Qc∇
   c v                             Parallel transport of v ∈ Tc(0) along a smooth curve c
γ, γ∇                              geodesic, geodesic with respect to connection ∇
Γij,l                              Christoffel symbols of the first kind (functions)
Γkij                               Christoffel symbols of the second kind (functions)
R                                  Riemann-Christoffel curvature tensor
[X, Y ]                            Lie bracket [X, Y ](f ) = X(Y (f )) − Y (X(f )), ∀f ∈ F(M )
∇-projection                       PS = arg minQ∈S D(θ(P ) : θ(Q))
∇∗ -projection                     PS∗ = arg minQ∈S D(θ(Q) : θ(P ))
C                                  Amari-Chentsov totally symmetric cubic 3-covariant tensor

P = {pθ (x)}θi nΘ                  parametric family of probability distributions
E, M, ∆D                           exponential family, mixture family, probability simplex
P I(θ) Fisher information matrix
P I(θ)                             Fisher Information Matrix (FIM) for a parametric family P




                                                   48
Pg                               Fisher information metric tensor field
exponential connection eP ∇      e
                                 P ∇ := Eθ [(∂i ∂j l)(∂k l)]
mixture connection m
                   P∇
                                 m
                                 P ∇ := Eθ [(∂i ∂j l + ∂i l∂j l)(∂k l)]
expected skewness tensor Cijk    Cijk := Eθ [∂i l∂j l∂k l]
                                    αk       1+α
                                                             ∂i ∂j l + 1−α
                                                                                        
expected α-connections           P Γ ij := − 2 Cijk = Eθ                2 ∂i l∂j l (∂k l)
≡                                equivalence of geometric structures

References
 [1] P-A Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds.
     Princeton University Press, 2009.
 [2] Pierre-Antoine Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix
     manifolds. Princeton University Press, 2009.
 [3] Maks Aizikovich Akivis and Boris Abramovich Rosenfeld. Élie Cartan (1869-1951), volume 123.
     American Mathematical Society, 2011.
 [4] Shun-ichi Amari. Theory of information spaces: A differential geometrical foundation of statistics.
     Post RAAG Reports, 1980.
 [5] Shun-ichi Amari. Differential-geometrical methods in statistics. Lecture Notes on Statistics, 28, 1985.
     second edition in 1990.
 [6] Shun-ichi Amari. Natural gradient works efficiently in learning. Neural computation, 10(2):251–276,
     1998.
 [7] Shun-ichi Amari. New developments of information geometry. Saiensu’sha, Tokyo, 2014. Jouhou
     kikagaku no shintenkai (in Japanese).
 [8] Shun-ichi Amari. Information Geometry and Its Applications. Applied Mathematical Sciences. Springer
     Japan, 2016.
 [9] Shun-ichi Amari and Hiroshi Nagaoka. Methods of Information Geometry. American Mathematical
     Society, 2007.
[10] Khadiga A. Arwini and Christopher Terence John Dodson. Information Geometry: Near Randomness
     and Near Independance. Springer, 2008.
[11] John Ashburner and Karl J. Friston. Diffeomorphic registration using geodesic shooting and Gauss-
     Newton optimisation. NeuroImage, 55(3):954–967, 2011.
[12] Nihat Ay and Shun-ichi Amari. A novel approach to canonical divergences within information geometry.
     Entropy, 17(12):8111–8129, 2015.
[13] Nihat Ay, Paolo Gibilisco, and Frantisek Matús. Information Geometry and its Applications: On the
     Occasion of Shun-ichi Amari’s 80th Birthday, volume 252 of Springer Proceedings in Mathematics &
     Statistics. Springer, 2018. following the June 2016 event at Liblice, Czech Republic.
[14] Katy S Azoury and Manfred K Warmuth. Relative loss bounds for on-line density estimation with the
     exponential family of distributions. Machine Learning, 43(3):211–246, 2001.
[15] John C. Baez and Derek K. Wise. Teleparallel gravity as a higher gauge theory. Communications in
     Mathematical Physics, 333(1):153–186, 2015.
[16] Arindam Banerjee, Srujana Merugu, Inderjit S Dhillon, and Joydeep Ghosh. Clustering with Bregman
     divergences. Journal of machine learning research, 6(Oct):1705–1749, 2005.


                                                      49
[17] Ole E. Barndorff-Nielsen, David Roxbee Cox, and Nancy Reid. The role of differential geometry in
     statistical theory. International Statistical Review, pages 83–96, 1986.
[18] Arnaud Berny. Selection and reinforcement learning for combinatorial optimization. In International
     Conference on Parallel Problem Solving from Nature, pages 601–610. Springer, 2000.

[19] Hans-Georg Beyer and Hans-Paul Schwefel. Evolution strategies–a comprehensive introduction. Nat-
     ural computing, 1(1):3–52, 2002.
[20] Jean-Daniel Boissonnat, Frank Nielsen, and Richard Nock. Bregman Voronoi diagrams. Discrete &
     Computational Geometry, 44(2):281–307, 2010.
[21] Silvère Bonnabel. Stochastic gradient descent on Riemannian manifolds. IEEE Transactions on Au-
     tomatic Control, 58(9):2217–2229, 2013.
[22] Jean-Pierre Bourguignon. Ricci curvature and measures. Japanese Journal of Mathematics, 4(1):27–45,
     2009.
[23] Lev M. Bregman. The relaxation method of finding the common point of convex sets and its appli-
     cation to the solution of problems in convex programming. USSR computational mathematics and
     mathematical physics, 7(3):200–217, 1967.
[24] Sébastien Bubeck et al. Convex optimization: Algorithms and complexity. Foundations and Trends R
     in Machine Learning, 8(3-4):231–357, 2015.
[25] Ovidiu Calin and Constantin Udriste. Geometric Modeling in Probability and Statistics. Mathematics
     and Statistics. Springer International Publishing, 2014.
[26] L. Lorne Campbell. An extended Čencov characterization of the information metric. Proceedings of
     the American Mathematical Society, 98(1):135–141, 1986.
[27] Elie Joseph Cartan. On manifolds with an affine connection and the theory of general relativity.
     Bibliopolis, 1986.
[28] AL Cauchy. Methode générale pour la résolution des systèmes d’équations simultanées. Comptes
     Rendus de l’Académie des Sciences, 25:536–538, 1847.
[29] Nikolai N. Chentsov. Statistical decision rules and optimal inference. Monographs, American Mathe-
     matical Society, Providence, RI, 1982.

[30] JM Corcuera and Federica Giummolè. A characterization of monotone and regular divergences. Annals
     of the Institute of Statistical Mathematics, 50(3):433–450, 1998.
[31] Grenville J Croll. The natural philosophy of Kazuo Kondo. arXiv preprint arXiv:0712.0641, 2007.
[32] Jean-Pierre Crouzeix. A relationship between the second derivatives of a convex function and of its
     conjugate. Mathematical Programming, 13(1):364–365, 1977.
[33] Imre Csiszár. Information-type measures of difference of probability distributions and indirect obser-
     vation. studia scientiarum Mathematicarum Hungarica, 2:229–318, 1967.
[34] Imre Csiszár and Paul C Shields. Information theory and statistics: A tutorial. Foundations and
     Trends R in Communications and Information Theory, 1(4):417–528, 2004.
[35] Haskell B Curry. The method of steepest descent for non-linear minimization problems. Quarterly of
     Applied Mathematics, 2(3):258–261, 1944.
[36] Anand Ganesh Dabak. A geometry for detection theory. PhD thesis, Rice University, 1993.


                                                    50
[37] Stephen Della Pietra, Vincent Della Pietra, and John Lafferty. Inducing features of random fields.
     IEEE transactions on pattern analysis and machine intelligence, 19(4):380–393, 1997.
[38] Manfredo P Do Carmo. Differential geometry of curves and surfaces: revised and updated second
     edition. Courier Dover Publications, 2016.
[39] Christopher Terence John Dodson, editor. Geometrization of statistical theory. ULDM Publications,
     1987. University of Lancaster, Department of Mathematics.
[40] D. C. Dowson and Basil V. Landau. The Fréchet distance between multivariate normal distributions.
     Journal of multivariate analysis, 12(3):450–455, 1982.
[41] Bradley Efron et al. Defining the curvature of a statistical problem (with applications to second order
     efficiency). The Annals of Statistics, 3(6):1189–1242, 1975.
[42] Shinto Eguchi. Second order efficiency of minimum contrast estimators in a curved exponential family.
     The Annals of Statistics, pages 793–803, 1983.
[43] Shinto Eguchi et al. A differential geometric approach to statistical inference on the basis of contrast
     functionals. Hiroshima mathematical journal, 15(2):341–391, 1985.
[44] Bent Fuglede and Flemming Topsøe. Jensen-Shannon divergence and Hilbert space embedding. In
     International Symposium on Information Theory (ISIT), page 31. IEEE, 2004.
[45] Akio Fujiwara. Foundations of Information Geometry. Makino Shoten, Tokyo, 2015. Jouhou kikagaku
     no kisou (in Japanese).
[46] Hitoshi Furuhata. Hypersurfaces in statistical manifolds. Differential Geometry and its Applications,
     27(3):420–429, 2009.
[47] Paolo Gibilisco, Eva Riccomagno, Maria Piera Rogantin, and Henry P. Wynn, editors. Algebraic and
     Geometric Methods in Statistics. Cambridge University Press, 2009.
[48] Evarist Giné and Richard Nickl. Mathematical foundations of infinite-dimensional statistical models,
     volume 40. Cambridge University Press, 2015.
[49] Erika Gomes-Gonçalves, Henryk Gzyl, and Frank Nielsen. Geometry and fixed-rate quantization in
     riemannian metric spaces induced by separable Bregman divergences. In Frank Nielsen and Frédéric
     Barbaresco, editors, Geometric Science of Information - 4th International Conference, GSI 2019,
     Toulouse, France, August 27-29, 2019, Proceedings, volume 11712 of Lecture Notes in Computer Sci-
     ence, pages 351–358. Springer, 2019.
[50] G. J. Gordon. Approximate solutions to Markov decision processes. PhD thesis, Department of Com-
     puter Science, Carnegie Mellon University, 1999.
[51] Masahito Hayashi. Quantum information. Springer, 2006.
[52] Jean-Baptiste Hiriart-Urruty and Claude Lemaréchal. Fundamentals of convex analysis. Springer
     Science & Business Media, 2012.
[53] Harold Hotelling. Spaces of statistical parameters. Bulletin of the American Mathematical Society
     (AMS), 36:191, 1930.
[54] Shun ichi Amari and Hiroshi Nagaoka. Methods of Information Geometry. Iwanami Shoten, Japan,
     1993. Jouhou kika no houhou (in Japanese).
[55] Harold Jeffreys. An invariant form for the prior probability in estimation problems. Proc. R. Soc.
     Lond. A, 186(1007):453–461, 1946.


                                                    51
[56] Jiantao Jiao, Thomas A Courtade, Albert No, Kartik Venkat, and Tsachy Weissman. Information mea-
     sures: the curious case of the binary alphabet. IEEE Transactions on Information Theory, 60(12):7616–
     7626, 2014.
[57] Satoshi Kakihara, Atsumi Ohara, and Takashi Tsuchiya. Information geometry and interior-point al-
     gorithms in semidefinite programs and symmetric cone programs. J. Optim. Theory Appl., 157(3):749–
     780, 2013.
[58] Robert E. Kass and Paul W. Vos.          Geometrical Foundations of Asymptotic Inference.         Wiley-
     Interscience, 07 1997.
[59] Michiaki Kawaguchi. An introduction to the theory of higher order spaces I. the theory of Kawaguchi
     spaces. RAAG Memoirs, 3:718–734, 1960.
[60] Robert W Keener. Theoretical statistics: Topics for a core course. Springer, 2011.
[61] Takashi Kurose. On the divergences of 1-conformally flat statistical manifolds. Tohoku Mathematical
     Journal, Second Series, 46(3):427–433, 1994.

[62] Stefan L. Lauritzen. Statistical manifolds. Differential geometry in statistical inference, 10:163–216,
     1987.
[63] Luigi Malagò and Giovanni Pistone. Information geometry of the Gaussian distribution in view of
     stochastic optimization. In Proceedings of the 2015 ACM Conference on Foundations of Genetic Al-
     gorithms XIII, pages 150–162, 2015.

[64] Ann F. S. Mitchell. Statistical manifolds of univariate elliptic distributions. International Statistical
     Review/Revue Internationale de Statistique, pages 1–16, 1988.
[65] Ann F. S. Mitchell. The information matrix, skewness tensor and α-connections for the general mul-
     tivariate elliptic distribution. Annals of the Institute of Statistical Mathematics, 41(2):289–304, 1989.

[66] Uwe Mühlich. Fundamentals of tensor calculus for engineers with a primer on smooth manifolds,
     volume 230. Springer, 2017.
[67] Michael Murray and John Rice. Differential geometry and statistics. Number 48 in Monographs on
     Statistics and Applied Probability. Chapman and Hall, 1993.
[68] Hiroshi Nagaoka and Shun-ichi Amari. Differential geometry of smooth families of probability distri-
     butions. Technical report, University of Tokyo, 1982. METR 82-7.
[69] Jan Naudts and Jun Zhang. Rho–tau embedding and gauge freedom in information geometry. Infor-
     mation Geometry, Aug 2018.
[70] Frank Nielsen. Legendre transformation and information geometry, 2010.

[71] Frank Nielsen. Cramér-Rao lower bound and information geometry. In Connected at Infinity II, pages
     18–37. Springer, 2013.
[72] Frank Nielsen. Hypothesis testing, information divergence and computational geometry. In GSI, pages
     241–248, 2013.

[73] Frank Nielsen. An information-geometric characterization of Chernoff information.           IEEE SPL,
     20(3):269–272, 2013.
[74] Frank Nielsen. Geometric Theory of Information. Springer, 2014.
[75] Frank Nielsen. Geometric Structures of Information. Springer, 2018.


                                                     52
[76] Frank Nielsen. What is... an information projection? Notices of the AMS, 65(3)(10):321–324, 2018.
[77] Frank Nielsen. On geodesic triangles with right angles in a dually flat space.       arXiv preprint
     arXiv:1910.03935, 2019.
[78] Frank Nielsen. On the Jensen–Shannon symmetrization of distances relying on abstract means. En-
     tropy, 21(5):485, 2019.
[79] Frank Nielsen. On Voronoi diagrams on the information-geometric Cauchy manifolds. Entropy,
     22(7):713, 2020.
[80] Frank Nielsen and Frédéric Barbaresco, editors. Geometric Science of Information, volume 8085 of
     Lecture Notes in Computer Science. Springer, 2013.
[81] Frank Nielsen and Frédéric Barbaresco, editors. Geometric Science of Information, volume 9389 of
     Lecture Notes in Computer Science. Springer, 2015.
[82] Frank Nielsen and Frédéric Barbaresco, editors. Geometric Science of Information, volume 10589 of
     Lecture Notes in Computer Science. Springer, 2017.
[83] Frank Nielsen and Sylvain Boltz. The Burbea-Rao and Bhattacharyya centroids. IEEE Transactions
     on Information Theory, 57(8), 2011.
[84] Frank Nielsen and Vincent Garcia. Statistical exponential families: A digest with flash cards. arXiv
     preprint arXiv:0911.4863, 2009.
[85] Frank Nielsen and Gaëtan Hadjeres. Monte Carlo information geometry: The dually flat case. CoRR,
     abs/1803.07225, 2018.
[86] Frank Nielsen and Gaëtan Hadjeres. Monte Carlo information-geometric structures. In Geometric
     Structures of Information, pages 69–103. Springer, 2019.
[87] Frank Nielsen and Richard Nock. Sided and symmetrized Bregman centroids. IEEE transactions on
     Information Theory, 55(6), 2009.
[88] Frank Nielsen and Richard Nock. Entropies and cross-entropies of exponential families. In 2010 IEEE
     International Conference on Image Processing, pages 3621–3624. IEEE, 2010.
[89] Frank Nielsen and Richard Nock. Hyperbolic Voronoi diagrams made easy. In International Conference
     on Computational Science and Its Applications (ICCSA), pages 74–80. IEEE, 2010.
[90] Frank Nielsen and Richard Nock. On the chi square and higher-order chi distances for approximating
     f -divergences. IEEE Signal Processing Letters, 21(1):10–13, 2013.
[91] Frank Nielsen and Richard Nock. Total Jensen divergences: Definition, properties and clustering. In
     ICASSP, pages 2016–2020, 2015.
[92] Frank Nielsen and Richard Nock. Patch matching with polynomial exponential families and projective
     divergences. In SISAP, pages 109–116, 2016.
[93] Frank Nielsen and Richard Nock. On the geometry of mixtures of prescribed distributions. In 2018
     IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2861–
     2865. IEEE, 2018.
[94] Frank Nielsen and Richard Nock.             Cumulant-free closed-form formulas for some common
     (dis)similarities between densities of an exponential family. arXiv preprint arXiv:2003.02469, 2020.
[95] Frank Nielsen and Ke Sun. Guaranteed bounds on information-theoretic measures of univariate mix-
     tures using piecewise log-sum-exp inequalities. Entropy, 18(12):442, 2016.


                                                  53
 [96] Frank Nielsen and Ke Sun. Guaranteed bounds on the Kullback–Leibler divergence of univariate
      mixtures. IEEE Signal Processing Letters, 23(11):1543–1546, 2016.
 [97] Frank Nielsen, Ke Sun, and Stéphane Marchand-Maillet. On Hölder projective divergences. Entropy,
      19(3):122, 2017.
 [98] Richard Nock, Frank Nielsen, and Shun-ichi. Amari. On conformal divergences and their population
      minimizers. IEEE TIT, 62(1):527–538, 2016.
 [99] Katsumi Nomizu, Nomizu Katsumi, and Takeshi Sasaki. Affine differential geometry: Geometry of
      affine immersions. Cambridge university press, 1994.
[100] Aleksandr Petrovich Norden. On pairs of conjugate parallel displacements in multidimensional spaces.
      Doklady Akademii nauk SSSR, 49(9):1345–1347, 1945. Kazan State University, Comptes rendus de
      l’Académie des sciences de l’URSS.
[101] Atsumi Ohara and Takashi Tsuchiya. An information geometric approach to polynomial-time interior-
      point algorithms: Complexity bound via curvature integral. The Institute of Statistical Mathematics,
      1055, 2007. Research Memorandum.
[102] Marı́a del Carmen Pardo and Igor Vajda. About distances of discrete distributions satisfying the data
      processing theorem of information theory. IEEE transactions on information theory, 43(4):1288–1293,
      1997.
[103] Charles Sanders Peirce. Chance, love, and logic: Philosophical essays. U of Nebraska Press, 1998.
[104] Linyu Peng, Huafei Sun, and Lin Jiu. The geometric structure of the Pareto distribution. Boletın de
      la Asociación Matemática Venezolana, 14(1-2):5–13, 2007.
[105] Tongzhu Li Linyu Peng and Huafei Sun. The geometric structure of the inverse gamma distribution.
      Contributions to Algebra and Geometry, 49(1):217–225, 2008.
[106] Gia-Thuy Pham, Rémy Boyer, and Frank Nielsen. Computational information geometry for binary
      classification of high-dimensional random tensors. Entropy, 20(3):203, 2018.
[107] Giovanni Pistone. Nonparametric information geometry. In Geometric Science of Information, pages
      5–36. Springer, 2013.
[108] Yu Qiao and Nobuaki Minematsu. A study on invariance of f -divergence and its application to speech
      recognition. IEEE Transactions on Signal Processing, 58(7):3884–3890, 2010.
[109] Hernando Quevedo. Geometrothermodynamics. Journal of Mathematical Physics, 48(1):013506, 2007.
[110] C. Radhakrishna Rao. Information and the accuracy attainable in the estimation of statistical param-
      eters. In Breakthroughs in statistics, pages 235–247. Springer, 1992.
[111] Radhakrishna C. Rao. Information and the accuracy attainable in the estimation of statistical param-
      eters. Bulletin of the Calcutta Mathematical Society, 37:81–91, 1945.
[112] Garvesh Raskutti and Sayan Mukherjee. The information geometry of mirror descent. IEEE Trans-
      actions on Information Theory, 61(3):1451–1457, 2015.
[113] Ferran Reverter and Josep M. Oller. Computing the Rao distance for Gamma distributions. Journal
      of computational and applied mathematics, 157(1):155–167, 2003.
[114] Yoshiharu Sato, Kazuaki Sugawa, and Michiaki Kawaguchi. The geometrical structure of the parameter
      space of the two-dimensional normal distribution. Reports on Mathematical Physics, 16(1):111–119,
      1979.


                                                    54
[115] Gerhard Schurz. Patterns of abduction. Synthese, 164(2):201–234, 2008.
[116] R. N. Sen. On parallelism in Riemannian space I. Bull. Calcutta Math. Soc, 36:102–107, 1944.
[117] R. N. Sen. On parallelism in Riemannian space II. Bull. Calcutta Math. Soc, 37:153–159, 1944.
[118] R. N. Sen. On parallelism in Riemannian space III. Bull. Calcutta Math. Soc, 38:161–167, 1946.
[119] Claude Elwood Shannon. A mathematical theory of communication. Bell Syst. Tech. J., 27:623–656,
      1948.
[120] Hirohiko Shima. The geometry of Hessian structures. World Scientific, 2007.
[121] Lene Theil Skovgaard. A Riemannian geometry of the multivariate normal model. Scandinavian
      journal of statistics, pages 211–223, 1984.
[122] Anuj Srivastava, Wei Wu, Sebastian Kurtek, Eric Klassen, and James Stephen Marron. Registration
      of Functional Data Using Fisher-Rao Metric. ArXiv e-prints, 03 2011.
[123] Stephen M. Stigler. The epic story of maximum likelihood. Statistical Science, pages 598–620, 2007.
[124] Ke Sun and Frank Nielsen. Relative Fisher information and natural gradient for learning large modular
      models. In ICML, pages 3289–3298, 2017.
[125] Asuka Takatsu. Wasserstein geometry of Gaussian measures.            Osaka Journal of Mathematics,
      48(4):1005–1026, 2011.
[126] Matus Telgarsky and Sanjoy Dasgupta. Agglomerative Bregman clustering. In Proceedings of the
      29th International Coference on International Conference on Machine Learning, pages 1011–1018.
      Omnipress, 2012.
[127] Igor Vajda. On metric divergences of probability measures. Kybernetika, 45(6):885–900, 2009.
[128] Hông Vân Lê. Statistical manifolds are statistical models. Journal of Geometry, 84(1-2):83–93, 2006.
[129] Hông Vân Lê. The uniqueness of the Fisher metric as information metric. Annals of the Institute of
      Statistical Mathematics, 69(4):879–896, 2017.
[130] Cédric Villani. Optimal transport: old and new, volume 338. Springer Science & Business Media, 2008.
[131] Abraham Wald. Statistical decision functions. The Annals of Mathematical Statistics, pages 165–205,
      1949.
[132] Abraham Wald. Statistical decision functions. Wiley, 1950.
[133] MI Wanas. Absolute parallelism geometry: Developments, applications and problems. arXiv preprint
      gr-qc/0209050, 2002.
[134] Shao-Wen Wei, Yu-Xiao Liu, and Robert B Mann. Ruppeiner geometry, phase transitions, and the
      microstructure of charged AdS black holes. Physical Review D, 100(12):124033, 2019.
[135] Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, and Jürgen Schmidhuber.
      Natural evolution strategies. The Journal of Machine Learning Research, 15(1):949–980, 2014.
[136] Shintaro Yoshizawa and Kunio Tanabe. Dual differential geometry associated with kullback-leibler
      information on the gaussian distributions and its 2-parameter deformations. SUT Journal of Mathe-
      matics, 35(1):113–137, 1999.
[137] Guodong Zhang, Shengyang Sun, David Duvenaud, and Roger Grosse. Noisy natural gradient as
      variational inference. In International Conference on Machine Learning, pages 5852–5861, 2018.


                                                     55
[138] Jun Zhang. Divergence functions and geometric structures they induce on a manifold. In Frank Nielsen,
      editor, Geometric Theory of Information, pages 1–30. Springer, 2014.
[139] Jun Zhang. On monotone embedding in information geometry. Entropy, 17(7):4485–4499, 2015.
[140] Jun Zhang. Reference duality and representation duality in information geometry. AIP Conference
      Proceedings, 1641(1):130–146, 2015.
[141] Zhenning Zhang, Huafei Sun, and Fengwei Zhong. Information geometry of the power inverse Gaussian
      distribution. Applied Sciences, 9, 2007.
[142] Fengwei Zhong, Huafei Sun, and Zhenning Zhang. The geometry of the Dirichlet manifold. Journal of
      the Korean Mathematical Society, 45(3):859–870, 2008.




                                                    56

"""Pure-standard-library scope and geometric-envelope checks before CUDA import."""
import math

MODEL_KEYS=("geometry", "grid", "viscosity", "dt", "n_steps", "chunk_steps",
            "window_start_step", "mean_inlet", "reference_diameter", "alpha_max",
            "epsilon", "advection_scheme")


def envelope_report(c):
    """A conservative whole-box certificate; includes smoothing support.

    Each Fourier f has zero angular mean, so its area log-normalisation is
    nonnegative (Jensen). Thus R exp(sum(abs(coefficients))) bounds every
    boundary radius. Independent nonoverlapping supports also imply exact
    additivity of geometric areas and no material union-interface ambiguity.
    """
    geom=c["geometry"]; scales=geom["design_scales"]
    radii=[b["radius"]*math.exp(sum(abs(v) for v in b["base_coefficients"])+1.05*sum(scales))
           +c["epsilon"] for b in geom["bodies"]]
    walls=[]; gaps=[]
    for i,b in enumerate(geom["bodies"]):
        x,y=b["center"]; walls.append(min(x,c["grid"]["lx"]-x,y,c["grid"]["ly"]-y)-radii[i])
        for j in range(i):
            a=geom["bodies"][j]
            gaps.append(math.dist(a["center"],b["center"])-radii[i]-radii[j])
    return dict(support_radius_bounds=radii,minimum_support_wall_gap=min(walls),
                minimum_support_pair_gap=min(gaps))


def validate_config(c):
    if 'wall_shift' in c or 'normalize_level_set' in c:
        raise ValueError('wall corrections require their own registered entry and passed CPU prerequisites')
    if c.get("schema")!="multibody-wake-development-1" or c.get("paper_eligible") is not False:
        raise ValueError("explicit multibody development configuration required")
    g=c["grid"]
    if not (8<=g["nx"]<=440 and 5<=g["ny"]<=82 and g["lx"]==2.2 and g["ly"]==.41):
        raise ValueError("fixed channel and at most 440x82 cells")
    if not (1<=c["n_steps"]<=48000 and 1<=c["chunk_steps"]<=200):
        raise ValueError("bounded trajectory/checkpoint storage required")
    if not (0<=c["window_start_step"]<c["n_steps"] and 0<c["dt"]<=.0005):
        raise ValueError("invalid time step or objective window")
    if c["advection_scheme"]!="central" or not (0<c["alpha_max"]<=500):
        raise ValueError("invalid pilot flow model")
    if not all(math.isfinite(c[k]) and c[k]>0 for k in ("epsilon","viscosity","mean_inlet","reference_diameter")):
        raise ValueError("finite positive physical parameters required")
    if len(c["theta"])!=12 or any(not math.isfinite(t) or abs(t)>1.05 for t in c["theta"]):
        raise ValueError("12 bounded dimensionless native design variables required")
    if not 0<=c.get("max_optimization_steps",0)<=8:
        raise ValueError("at most eight optimization updates per segment")
    if "continuation" in c:
        resume=c["continuation"]
        if (not isinstance(resume,dict) or set(resume)!={"path","sha256"}
                or not isinstance(resume["path"],str) or not resume["path"]
                or not isinstance(resume["sha256"],str) or len(resume["sha256"])!=64
                or any(ch not in "0123456789abcdef" for ch in resume["sha256"])
                or not 1<=c.get("max_optimization_steps",0)<=4
                or any(abs(t)>1 for t in c["theta"])):
            raise ValueError("continuation needs a pinned parent and at most four updates")
    geo=c["geometry"]
    if geo["kind"]!="fixed-area-lobed-array-v1" or geo["modes"]!=[2,3,4] or len(geo["bodies"])!=3:
        raise ValueError("frozen three-body, twelve-variable geometry model required")
    if len(geo["design_scales"])!=4 or any(not 0<s<=.2 for s in geo["design_scales"]):
        raise ValueError("bounded parameter scales required")
    for b in geo["bodies"]:
        if len(b["center"])!=2 or len(b["base_coefficients"])!=6 or not (0<b["radius"]<=.08):
            raise ValueError("invalid native body")
        if not all(math.isfinite(v) for v in b["center"]+b["base_coefficients"]):
            raise ValueError("nonfinite native geometry")
    report=envelope_report(c)
    if report["minimum_support_wall_gap"]<.02 or report["minimum_support_pair_gap"]<.04:
        raise ValueError("whole design box violates wall/pair clearance")
    if c["epsilon"]/max(g["lx"]/g["nx"],g["ly"]/g["ny"])<1.4:
        raise ValueError("interface half-width insufficiently resolved for pilot")

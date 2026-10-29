"""Route our HBV's runoff through DDR, on the Juniata.

This joins the two halves of the project. DDR routes per-divide runoff through
the MERIT network at an hourly step; our model produces one lumped daily series
per basin. The Juniata gauge (01567000, 8,657 km²) is in both — row 208 of our
2,003-basin set and the single gauge of DDR's sample bundle — so the comparison
can be made without reconciling anything else.

Four series are scored against the same observations, over DDR's own test
period:

    their Q', no routing     the bundle's per-divide runoff, summed
    their Q', routed         what DDR reports as its result
    our HBV, no routing      our generated runoff, summed
    our HBV, routed          our runoff through their network

Our runoff enters through ``lumped_qprime.LumpedToDivides``: mm/day to m³/s by
area, each day repeated 24 times as DDR's own reader does for daily stores, and
split across the 213 divides. The split uses the bundle's own long-run spatial
pattern rather than bare area, which was measured to route slightly better.

What to read
------------
Our runoff is about 17% low in the mean against this gauge before any routing
happens, so a bias in the routed result is inherited, not caused. The useful
comparison is therefore **the change routing makes to each**, not the levels.

The KAN is left untrained unless a checkpoint is passed, so both runs see the
same channel parameters and the comparison isolates the runoff.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from ddr import dmc, kan, streamflow
from ddr.scripts_utils import tau_trim_and_downsample
from ddr.validation import Metrics

from examples.juniata.lumped_qprime import LumpedToDivides, share_from_reference
from examples.juniata.train_and_test import _eval_dataset_and_batch, make_config

OUR_ATTR = "/scratch/data/Farshid/dataset/attr2003_mswep_03122024"
FWD_RUN = "/scratch/m0rahm31/dphydro_out/FWD2003"
GAUGE = 1567000
OUR_START = np.datetime64("1981-01-01")     # first day of the forward run


def our_daily_runoff(start: str, end: str):
    """Our HBV's generated runoff for the Juniata, mm/day, over a date range."""
    a = np.load(OUR_ATTR + ".npy")
    nm = json.load(open(OUR_ATTR + "_name.json"))
    if isinstance(nm, dict):
        nm = list(nm.values())[0] if len(nm) == 1 else list(nm)
    gid = np.round(a[:, nm.index("site_no_int")]).astype(np.int64)
    row = int(np.where(gid == GAUGE)[0][0])
    area = float(a[row, nm.index("DRAIN_SQKM")])

    d = sorted(glob.glob(f"{FWD_RUN}/*/*/ts*/"))[0]
    gen = np.squeeze(np.load(d + "flow_sim_no_rout.npy")).astype(float)[:, row]
    dates = OUR_START + np.arange(len(gen))
    m = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    return np.nan_to_num(gen[m], nan=0.0), area, int(m.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="examples/juniata/data")
    ap.add_argument("--ckpt", default=None, help="trained KAN checkpoint, optional")
    ap.add_argument("--test-period", nargs=2, default=["1995/10/01", "2010/09/30"])
    args = ap.parse_args()

    cfg = make_config(Path(args.bundle))
    eval_cfg, dataset, batch = _eval_dataset_and_batch(cfg, tuple(args.test_period))

    # ---- the two runoff fields -------------------------------------------
    their_q = streamflow(eval_cfg)(routing_dataclass=batch)          # (T_h, 213)
    n_hours, n_div = their_q.shape
    n_days = n_hours // 24
    print(f"DDR test window: {n_days} days, {n_div} divides, {n_hours} hours")

    start = args.test_period[0].replace("/", "-")
    end = args.test_period[1].replace("/", "-")
    ours_mm, area, got = our_daily_runoff(start, end)
    print(f"our HBV over the same window: {got} days, area {area:,.0f} km2")
    if got < n_days:
        raise SystemExit(f"our record is short: {got} days against {n_days} needed")
    ours_mm = ours_mm[:n_days]

    share = share_from_reference(their_q)          # the bundle's own spatial pattern
    adapter = LumpedToDivides(area_km2=area, share=share)
    our_q = adapter(torch.tensor(ours_mm, dtype=torch.float64))      # (T_h, 213)
    print(f"our q_prime: {tuple(our_q.shape)}, "
          f"mean basin flow {our_q.sum(1).mean():.1f} m3/s "
          f"against theirs {their_q.sum(1).mean():.1f}")

    # ---- channel parameters, shared between both runs ---------------------
    nn = kan(
        input_var_names=eval_cfg.kan.input_var_names,
        learnable_parameters=eval_cfg.kan.learnable_parameters,
        hidden_size=eval_cfg.kan.hidden_size,
        num_hidden_layers=eval_cfg.kan.num_hidden_layers,
        grid=eval_cfg.kan.grid,
        k=eval_cfg.kan.k,
        seed=eval_cfg.seed,
        device=eval_cfg.device,
    )
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=eval_cfg.device)
        nn.load_state_dict(state["model_state_dict"])
        print(f"loaded trained KAN from {args.ckpt}")
    else:
        print("KAN untrained — both runs see identical channel parameters")

    routing = dmc(cfg=eval_cfg, device=eval_cfg.device)
    routing.set_progress_info(epoch=0, mini_batch=0)
    with torch.no_grad():
        params = nn(inputs=batch.normalized_spatial_attributes.to(eval_cfg.device))

    obs = batch.observations.streamflow.values[:, :-2]

    def summed(q):
        d = tau_trim_and_downsample(q.sum(1, keepdim=True).T, tau=eval_cfg.params.tau)
        return d.cpu().numpy()

    def routed(q):
        with torch.no_grad():
            out = routing(routing_dataclass=batch, spatial_parameters=params, streamflow=q)
            d = tau_trim_and_downsample(out["runoff"], eval_cfg.params.tau).cpu().numpy()
        routing.routing_engine.q_prime = None
        routing.routing_engine._discharge_t = None
        return d

    def line(label, pred):
        m = Metrics(pred=pred, target=obs)
        nse, kge = float(np.nanmedian(m.nse)), float(np.nanmedian(m.kge))
        bias = 100 * (np.nanmean(pred) - np.nanmean(obs)) / np.nanmean(obs)
        print(f"  {label:<26} {nse:>8.4f} {kge:>8.4f} {bias:>+9.1f}")
        return nse

    print(f"\n  {'':<26} {'NSE':>8} {'KGE':>8} {'bias %':>9}")
    t_none = line("their Q', no routing", summed(their_q))
    t_rout = line("their Q', routed", routed(their_q))
    o_none = line("our HBV, no routing", summed(our_q))
    o_rout = line("our HBV, routed", routed(our_q))

    print(f"\n  routing changes their Q' by {t_rout - t_none:+.4f} NSE")
    print(f"  routing changes our HBV  by {o_rout - o_none:+.4f} NSE")
    print("""
Read the two changes, not the levels: our runoff carries a mean bias against
this gauge before routing touches it, so the absolute scores are not comparable
between the two runoff sources. What is comparable is how much each gains from
being routed, since both pass through the same network with the same channel
parameters.""")

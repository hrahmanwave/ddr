"""Does the spatial pattern of runoff matter, or would an area split do?

DDR's Q' gives each of the 213 divides its own runoff series. A lumped model
gives one number for the whole basin, which would have to be split by area.
This asks what that substitution costs: route their real Q', then route an
area-split version of the same daily total, and compare both against the gauge.

If the two scores are close, plugging a basin-level HBV into the seam is a
fair approximation. If they diverge, the runoff model has to be run per divide.
"""
from pathlib import Path
import numpy as np, torch, xarray as xr
from examples.juniata.train_and_test import make_config, _eval_dataset_and_batch
from ddr import streamflow
from ddr.scripts_utils import tau_trim_and_downsample
from ddr.validation import Metrics

cfg = make_config(Path("examples/juniata/data"))
cfg2, ds, batch = _eval_dataset_and_batch(cfg, ("1995/10/01", "2010/09/30"))
q = streamflow(cfg2)(routing_dataclass=batch)          # (T_hours, 213)
print("q_prime:", tuple(q.shape))

# each divide's long-run share of the total — the stand-in for an area weight
share = (q.mean(0) / q.mean(0).sum()).unsqueeze(0)     # (1, 213)
total = q.sum(1, keepdim=True)                         # (T, 1) basin total
q_split = total * share                                # (T, 213) area-split

obs = batch.observations.streamflow.values[:, :-2]

# Summing is blind to how the runoff is distributed: both fields have the same
# basin total by construction, so the no-routing score cannot separate them.
# The question only becomes answerable once the water is routed, because then
# which reach the water entered at decides how long it takes to arrive.
from ddr import dmc, kan
from ddr.validation.enums import Mode

cfg2.mode = Mode.TESTING
routing = dmc(cfg=cfg2, device=cfg2.device)
routing.set_progress_info(epoch=0, mini_batch=0)
nn = kan(
    input_var_names=cfg2.kan.input_var_names,
    learnable_parameters=cfg2.kan.learnable_parameters,
    hidden_size=cfg2.kan.hidden_size,
    num_hidden_layers=cfg2.kan.num_hidden_layers,
    grid=cfg2.kan.grid,
    k=cfg2.kan.k,
    seed=cfg2.seed,
    device=cfg2.device,
)
with torch.no_grad():
    spatial = nn(inputs=batch.normalized_spatial_attributes.to(cfg2.device))

def score_routed(x, label):
    with torch.no_grad():
        out = routing(routing_dataclass=batch, spatial_parameters=spatial, streamflow=x)
        pred = out["runoff"]
        d = tau_trim_and_downsample(pred, tau=0).cpu().numpy()
    m = Metrics(pred=d, target=obs)
    print(f"  {label:<20} {float(np.nanmedian(m.nse)):>8.4f} {float(np.nanmedian(m.kge)):>8.4f}", flush=True)
    routing.routing_engine.q_prime = None
    routing.routing_engine._discharge_t = None

def score_summed(x, label):
    d = tau_trim_and_downsample(x.sum(1, keepdim=True).T, tau=0).cpu().numpy()
    m = Metrics(pred=d, target=obs)
    print(f"  {label:<20} {float(np.nanmedian(m.nse)):>8.4f} {float(np.nanmedian(m.kge)):>8.4f}")

print(f"\n{'':<22} {'NSE':>8} {'KGE':>8}")
print("no routing — identical by construction, the totals are the same:")
score_summed(q, "their Q'")
score_summed(q_split, "area-split Q'")
print("\nrouted — this is where the spatial pattern can show:")
score_routed(q, "their Q'")
score_routed(q_split, "area-split Q'")

diff = (q - q_split).abs().sum() / q.abs().sum()
print(f"\nthe two runoff fields differ by {diff.item():.1%} of total volume")

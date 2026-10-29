"""Feed a basin-level runoff series into DDR's q_prime seam.

DDR expects runoff per MERIT divide: a gradient-capable hourly (T, N) tensor in
m³/s, one column per reach. A lumped model like ours produces one number for the
whole basin, daily, in mm/day. This bridges the two.

Three steps, each of which is a modelling choice worth stating:

  **mm/day to m³/s** — multiply by the drainage area. Exact.

  **daily to hourly** — repeat each day 24 times, which is what DDR's own reader
  does for its daily store (``readers.py`` line 517: *"Daily store:
  nearest-neighbor to hourly is just repeat(24)"*). It adds no information; it
  just lets the routing solve at a step fine enough to resolve a few hours of
  travel. That is the whole reason routing works here and did not in the daily
  runs.

  **one basin to N divides** — split by each divide's long-run share of the
  total. Measured on the Juniata: routing the real per-divide Q' gave NSE 0.672
  and routing an area-split version of the same totals gave 0.710, so the
  simplification is not what limits the result there.

The output keeps gradients, so an upstream HBV can be trained through the
routing once the two are joined.

Usage, replacing the reader at the seam:

    from lumped_qprime import LumpedToDivides
    adapter = LumpedToDivides(area_km2=8657.0, share=share_vector)
    q_prime = adapter(hbv_daily_mm)          # (T_days,) -> (T_days*24, N)
"""

import torch


class LumpedToDivides(torch.nn.Module):
    """Turn one basin's daily runoff into DDR's per-divide hourly tensor.

    Parameters
    ----------
    area_km2 : float
        Drainage area of the lumped basin, used for the depth-to-rate conversion.
    share : torch.Tensor
        Each divide's fraction of the basin total, shape (N,), summing to 1.
        Area fractions are the natural choice; a long-run share measured from a
        reference Q' field works too and is what was validated here.
    repeat : int, optional
        Sub-steps per day, 24 by default. DDR's reader uses the same.
    """

    def __init__(self, area_km2: float, share: torch.Tensor, repeat: int = 24):
        super().__init__()
        s = share.double()
        total = s.sum()
        if not torch.isclose(total, torch.tensor(1.0, dtype=torch.float64), atol=1e-6):
            s = s / total                       # normalise rather than fail
        self.register_buffer("share", s)
        self.area_km2 = float(area_km2)
        self.repeat = int(repeat)
        # mm over the basin per day -> m³/s
        self.mm_per_day_to_cms = area_km2 * 1e6 / 1000.0 / 86400.0

    def forward(self, daily_mm: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        """
        daily_mm : (T,) or (T, 1) runoff depth in mm/day, gradients preserved.
        Returns  : (T*repeat, N) discharge in m³/s.
        """
        q = daily_mm.squeeze()
        if q.dim() != 1:
            raise ValueError(f"expected a 1-D daily series, got {tuple(daily_mm.shape)}")
        cms = q.double() * self.mm_per_day_to_cms                  # (T,)
        hourly = cms.repeat_interleave(self.repeat)                # (T*24,)
        return (hourly.unsqueeze(1) * self.share.unsqueeze(0)).to(dtype)   # (T*24, N)


def share_from_areas(areas: torch.Tensor) -> torch.Tensor:
    """Divide shares from catchment areas — the obvious weighting."""
    a = areas.double().clamp(min=0)
    return a / a.sum()


def share_from_reference(q_ref: torch.Tensor) -> torch.Tensor:
    """Divide shares from a reference runoff field's long-run means.

    Useful when a per-divide Q' already exists and you want the split to match
    its spatial pattern rather than bare area.
    """
    m = q_ref.double().mean(dim=0).clamp(min=0)
    return m / m.sum()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, T = 213, 365

    print("1. shape and units")
    share = share_from_areas(torch.rand(N) * 100)
    ad = LumpedToDivides(area_km2=8657.0, share=share)
    daily = torch.full((T,), 2.0, dtype=torch.float64)      # 2 mm/day everywhere
    q = ad(daily)
    print(f"   in  {tuple(daily.shape)} mm/day  ->  out {tuple(q.shape)} m3/s")
    expected = 2.0 * 8657e6 / 1000 / 86400
    print(f"   basin total {q[0].sum().item():.2f} m3/s, expected {expected:.2f}")
    assert abs(q[0].sum().item() - expected) < 0.05

    print("\n2. volume is conserved across the split and the repeat")
    daily = torch.rand(T, dtype=torch.float64) * 5
    q = ad(daily)
    v_in = (daily * ad.mm_per_day_to_cms * 86400).sum().item()      # m³ over the period
    v_out = (q.sum(1).double() * 3600).sum().item()
    print(f"   in {v_in:,.0f} m3, out {v_out:,.0f} m3, difference {100*(v_out-v_in)/v_in:+.3f}%")
    assert abs(v_out - v_in) / v_in < 1e-5

    print("\n3. each day becomes 24 identical hours")
    q = ad(torch.tensor([1.0, 7.0], dtype=torch.float64))
    tot = q.sum(1)
    print(f"   day 1 hours 0-23 all equal: {bool((tot[:24] - tot[0]).abs().max() < 1e-6)}")
    print(f"   day 2 is 7x day 1: {tot[24].item() / tot[0].item():.2f}")
    assert abs(tot[24].item() / tot[0].item() - 7.0) < 1e-4

    print("\n4. shares sum to 1 and are respected")
    print(f"   share sum {ad.share.sum().item():.6f}")
    q = ad(torch.full((2,), 3.0, dtype=torch.float64))
    frac = q[0] / q[0].sum()
    print(f"   max deviation from the given share: {(frac.double() - ad.share).abs().max().item():.2e}")

    print("\n5. gradients pass through to the runoff")
    daily = torch.rand(T, dtype=torch.float64, requires_grad=True)
    ad(daily).sum().backward()
    print(f"   dq/d(daily) non-zero: {bool(daily.grad.abs().sum() > 0)}, "
          f"magnitude {daily.grad.abs().mean().item():.4f}")
    assert daily.grad.abs().sum() > 0

    print("\n6. a reference field can set the shares instead of area")
    q_ref = torch.rand(100, N) * torch.rand(N)
    s2 = share_from_reference(q_ref)
    print(f"   sum {s2.sum().item():.6f}, min {s2.min().item():.2e}, max {s2.max().item():.4f}")

    print("\nall checks passed")

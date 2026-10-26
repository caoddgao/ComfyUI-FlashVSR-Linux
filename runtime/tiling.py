"""Spatial tiling for FlashVSR.

FlashVSR needs output sizes that are multiples of 128 (8x8 windows of 2x2
patches of the 8x VAE latent). Large frames are processed as overlapping
tiles that each run the whole temporal pipeline, then blended back with a
linear ramp across the overlap.
"""

import math

import torch

ALIGN = 128


def round_up(x, m=ALIGN):
    return (x + m - 1) // m * m


def axis_tiles(length, tile, overlap):
    """Start offsets of tiles of size ``tile`` covering ``length`` with at least ``overlap`` overlap."""
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    count = math.ceil((length - tile) / step) + 1
    # spread the tiles evenly; spacing <= step, so neighbours overlap by >= overlap.
    # Only the tile size has to be a multiple of 128, not the offset.
    return [round(i * (length - tile) / (count - 1)) for i in range(count)]


def plan(out_h, out_w, tile_edge_out, overlap_out):
    """Return ([(y0, x0, th, tw), ...], overlap) in output pixels.

    out_h/out_w must already be multiples of 128 (padded size). The overlap is
    rounded up to 8 pixels and capped at half a tile.
    """
    tile = max(ALIGN, round_up(int(tile_edge_out)))
    th = min(tile, out_h)
    tw = min(tile, out_w)
    overlap = min(round_up(int(overlap_out), 8), min(th, tw) // 2)
    tiles = []
    for y0 in axis_tiles(out_h, th, overlap):
        for x0 in axis_tiles(out_w, tw, overlap):
            tiles.append((y0, x0, th, tw))
    return tiles, (overlap if len(tiles) > 1 else 0)


def blend_weight(th, tw, y0, x0, out_h, out_w, overlap, device=None):
    """[th, tw] weight: 1 inside, linear ramp on edges that touch another tile."""
    def ramp(n, at_start, at_end):
        w = torch.ones(n, device=device)
        r = max(1, min(overlap, n // 2))
        if overlap > 0:
            edge = (torch.arange(r, device=device, dtype=torch.float32) + 0.5) / r
            if not at_start:
                w[:r] = edge
            if not at_end:
                w[n - r:] = edge.flip(0)
        return w

    wy = ramp(th, y0 == 0, y0 + th >= out_h)
    wx = ramp(tw, x0 == 0, x0 + tw >= out_w)
    return wy[:, None] * wx[None, :]

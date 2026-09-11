# Stage A Summary

Date: 2026-09-02  
Status: **FAIL**

The existing Core27 -> HumanML3D-263 bridge has substantial round-trip bias:
FID 14.9845 and paired embedding cosine 0.8807. It recomputes rotations from
positions and does not preserve ARDY/SMPL rotation and twist semantics.
Official HumanML3D learned metrics are therefore blocked. The machine-readable
measurements and fallback policy are in `A_HML_BRIDGE_RESULTS.json`.

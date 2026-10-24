# 0.2.23 — Infernux 0.4.3-v5 Player shutdown fix

- Rebuild the Web Player and both host shader-tool payloads from frozen engine commit `de7cc8c8b0ecbfbb7afd8553f256a1fa236f0aaa`.
- Unload native scenes before releasing asset catalogs so component shutdown retains font and other asset references.
- Keep the package engine requirement `>=0.4.3` and exact matching precompiled Player release.

# 0.2.22 — Infernux 0.4.3-v5

- Rebuild the Web Player and both Windows/Linux shader-tool payloads against Infernux 0.4.3-v5.
- Build from frozen engine commit `f683bfa678eb680748c20be91badb9fe668ec553`.
- Keep the package engine requirement `>=0.4.3`; precompiled Players require the exact matching engine release.

# 0.2.18 — Infernux 0.4.3

- Rebuild the complete Player payload against the final 0.4.3 engine source, including asset import-failure publication.
- Keep the engine requirement >=0.4.3.

# 0.2.17 — Infernux 0.4.3

- Rebuild the complete Web Player and both desktop shader-tool payloads with Infernux 0.4.3.
- Include complete split scene bindings, layered cameras and per-camera particles.
- Render textured flipbook and soft particles, and preserve shader texture/uniform bindings.
- Use pinned engine dependency submodules and shared texture decoding.
- Keep the package engine requirement >=0.4.3; native payloads require the exact engine release.

- Derive the runtime version from the package/engine identity instead of hard-coding a previous release.

# 0.2.15 — Infernux 0.4.2 revision 2

- Publish the exporter and precompiled Player payload built with Infernux 0.4.2-v2.
- Preserve the exact engine-release check for precompiled runtimes.
- Keep the package engine requirement >=0.4.2.
- Publish new immutable artifacts; earlier package releases remain unchanged.

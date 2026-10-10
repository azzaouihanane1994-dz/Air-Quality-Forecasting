# Publishing to GitHub

This directory is prepared for the repository URL referenced by the manuscript:
`https://github.com/azzaouihanane1994-dz/Air-Quality-Forecasting`.

Create an empty repository with that name under the intended GitHub account, then push from this directory:

```bash
git remote add origin https://github.com/azzaouihanane1994-dz/Air-Quality-Forecasting.git
git push -u origin main
```

If the remote already contains commits, inspect and reconcile them before pushing. No force push is needed for an empty repository.

Before adding future results, run `python scripts/check_repository.py`. The 50 MB limit applies to every working-tree file. Oversized original paths are ignored explicitly. Keep larger artifacts outside this Git repository.

The prepared local repository includes MIT licensing, citation metadata, scientific dependencies, documentation, source scripts and eligible recorded outputs. It has not been uploaded by this preparation process.
